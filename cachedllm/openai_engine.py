from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Dict,
    Tuple,
    Optional,
    TypeVar,
    Union,
)

import enum
import abc
import os
import openai
import time
import itertools
import json

from .query_tools import LLMEngineBase

class OpenAIBaseEngine(LLMEngineBase, abc.ABC):
    class OpenAIMode(enum.Enum):
        COMPLETION = "completion"
        CHAT = "chat"

    def __init__(self, engine_name: str, mode: OpenAIMode):
        if os.environ.get("OPENAI_ORG") is not None:
            client = openai.OpenAI(
                api_key= os.environ["OPENAI_KEY"],
                organization=os.environ["OPENAI_ORG"],
            )
        else:
            client = openai.OpenAI(
                api_key=os.environ["OPENAI_KEY"],
            )
        self.cliet = client
        self.engine_name = engine_name
        self.mode = mode
        self.MAX_ATTEMPTS = 10
        self.RATE_WAITTIME = 10
        self.ERROR_WAITTIME = 10
        self.total_usage = 0

    def reset_usage(self):
        self.total_usage = 0

    def get_usage(self):
        return self.total_usage

    def model_args_to_str(self) -> str:
        return f"MODEL={self.engine_name}\n"

    def query_args_to_str(self,
                        max_tokens: int,
                        temperature: float,
                        top_p: float,
                        n: int,
                        logprobs: Optional[int],
                        stop_tokens: Optional[Union[str, List[str]]],
                        echo_prompt: bool,
                        **kwargs) -> str:
        return self._default_query_args_to_str(max_tokens, temperature, top_p, n, logprobs, stop_tokens, echo_prompt)

    def _gpt_safe_completion(self,
                             prompts: List[str],
                             max_tokens: int,
                             temperature: float,
                             top_p: float,
                             n: int,
                             logprobs: Optional[int],
                             stop_tokens: Optional[Union[str, List[str]]],
                             echo_prompt: bool,
                             **kwargs) -> List[Any]:
        assert not (temperature > 0.0 and top_p < 1.0)
        args_dict = {
            "model": self.engine_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "logprobs": logprobs,
            "echo": echo_prompt,
            "stop": stop_tokens,
        }
        if top_p == 1.0: args_dict.pop("top_p")
        if logprobs is None: args_dict.pop("logprobs")
        if stop_tokens is None: args_dict.pop("stop")
        if echo_prompt is False: args_dict.pop("echo")

        last_exc = None
        for _ in range(self.MAX_ATTEMPTS):
            try:
                if self.mode == OpenAIBaseEngine.OpenAIMode.COMPLETION:
                    response = self.cliet.completions.create(prompt=prompts, **args_dict)
                elif self.mode == OpenAIBaseEngine.OpenAIMode.CHAT:
                    assert len(prompts) == 1
                    response = self.cliet.chat.completions.create(messages=prompts[0], **args_dict)
                response = json.loads(response.model_dump_json())
                self.total_usage += response["usage"]["total_tokens"]
                return response["choices"]
            # wait longer for rate limit errors
            # except openai.error.RateLimitError as e:
            except openai.RateLimitError as e:
                last_exc = e
                time.sleep(self.RATE_WAITTIME)
            # invalid request errors are fatal
            except openai.BadRequestError as e:
                raise e
            except openai.OpenAIError as e:
                last_exc = e
                time.sleep(self.ERROR_WAITTIME)

        if isinstance(last_exc, openai.RateLimitError):
            raise RuntimeError("Consistently hit rate limit error")

        # make fake choices
        if self.mode == OpenAIBaseEngine.OpenAIMode.COMPLETION:
            fake_choices = [
                [{
                    "text": " OPENAI Error:" + str(last_exc),
                    "logprobs": None,
                    "index": 0,
                    "finish_reason": "api_error"
                }] * n
                for p in prompts
            ]
        elif self.mode == OpenAIBaseEngine.OpenAIMode.CHAT:
            fake_choices = [
                [{
                    "message": {"role": "assistant", "content": " OPENAI Error:" + str(last_exc)},
                    "index": 0,
                    "finish_reason": "api_error"
                }] * n
                for p in prompts
            ]
        fake_choices = itertools.chain(*fake_choices)
        return fake_choices

    def complete_batch(self, prompts: List[Any],
                             max_tokens: int,
                             temperature: float,
                             top_p: float,
                             n: int,
                             logprobs: Optional[int],
                             stop_tokens: Optional[Union[str, List[str]]],
                             echo_prompt: bool,
                             **kwargs) -> List[Any]:
        all_choices = self._gpt_safe_completion(prompts, max_tokens, temperature, top_p, n, logprobs, stop_tokens, echo_prompt, **kwargs)
        choices_by_prompt = [all_choices[(i * n):(i * n + n)] for i in range(len(prompts))]
        responses = []
        for prompt, choices in zip(prompts, choices_by_prompt):
            meta_response = self.construct_meta_response(prompt, choices, logprobs is not None, echo_prompt)
            meta_response["model"] = self.engine_name
            responses.append(meta_response)

        return responses

    @abc.abstractmethod
    def construct_meta_response(self, prompt: Any, choices: List[Any], include_logprobs: bool, echo_prompt: bool) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAICompletionEngine(OpenAIBaseEngine):
    def __init__(self, engine_name: str):
        super().__init__(engine_name, self.OpenAIMode.COMPLETION)
        # config some engine-specific parameters
        if engine_name.startswith("code-"):
            self.MAX_ATTEMPTS = 30

    def prompt_to_hashable_str(self, prompt: Any) -> str:
        assert isinstance(prompt, str)
        return prompt

    def construct_meta_response(self, prompt: Any, choices: List[Any], include_logprobs: bool, echo_prompt: bool) -> Dict[str, Any]:
        error_flag = choices[0]["finish_reason"] == "api_error"
        if error_flag:
            return {"prompt": {"text": prompt, "logprobs": None}, "completions": choices}

        if not include_logprobs:
            if echo_prompt:
                for c in choices:
                    c["text"] = c["text"][len(prompt):]

            return {"prompt": {"text": prompt, "logprobs": None}, "completions": choices}

        # make prompt logprobs and choices logprobs
        if echo_prompt:
            completion_offset = len(prompt)
            anchor_logporbs = choices[0]["logprobs"]
            split_point = next(filter(lambda i: anchor_logporbs["text_offset"][i] >= completion_offset, range(len(anchor_logporbs["text_offset"]))), len(anchor_logporbs["text_offset"]))

            prompt_logprobs = {k: v[:split_point] for k, v in anchor_logporbs.items()}
            for c in choices:
                c["text"] = c["text"][completion_offset:]
                logprobs = c["logprobs"]
                for k, v in logprobs.items():
                    logprobs[k] = v[split_point:]
                c["logprobs"] = logprobs
            return {"prompt": {"text": prompt, "logprobs": prompt_logprobs}, "completions": choices}
        else:
            # strip logprobs, the first item is usually the last token of the prompt
            for c in choices:
                logprobs = c["logprobs"]
                if len(logprobs["text_offset"]) > 0 and logprobs["text_offset"][1] == len(prompt):
                    for k, v in logprobs.items():
                        logprobs[k] = v[1:]
                    c["logprobs"] = logprobs
            return {"prompt": {"text": prompt, "logprobs": None}, "completions": choices}


class OpenAIChatEngine(OpenAIBaseEngine):
    def __init__(self, engine_name: str):
        super().__init__(engine_name, self.OpenAIMode.CHAT)

    def prompt_to_hashable_str(self, prompt: Any) -> str:
        assert isinstance(prompt, list)
        return str(prompt)

    def complete_batch(self, prompts: List[Any],
                             max_tokens: int,
                             temperature: float,
                             top_p: float,
                             n: int,
                             logprobs: Optional[int],
                             stop_tokens: Optional[Union[str, List[str]]],
                             echo_prompt: bool,
                             **kwargs) -> List[Any]:
        # does not support echo back prompts and get logprobs for now
        if echo_prompt or logprobs is not None:
            raise RuntimeError("OpenAI chat model does not echo_prompt or logprobs")
        if len(prompts) > 1:
            raise RuntimeError("OpenAI chat model does not support batched requests")

        return super().complete_batch(prompts, max_tokens, temperature, top_p, n, logprobs, stop_tokens, echo_prompt, **kwargs)

    def construct_meta_response(self, prompt: Any, choices: List[Any], include_logprobs: bool, echo_prompt: bool) -> Dict[str, Any]:
        error_flag = choices[0]["finish_reason"] == "api_error"
        if error_flag:
            return {"prompt": {"message": prompt,}, "completions": choices}
        assert not include_logprobs and not echo_prompt
        return {"prompt": {"message": prompt,}, "completions": choices}
