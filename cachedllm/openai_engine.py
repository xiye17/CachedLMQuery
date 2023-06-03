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

import os
import openai
import time
import itertools
import json

from .query_tools import LLMEngineBase

class OpenAICompletionEngine(LLMEngineBase):
    MAX_ATTEMPTS = 30
    RATE_WAITTIME = 10
    ERROR_WAITTIME = 1

    def __init__(self, engine_name: str):
        openai.api_key = os.environ["OPENAI_KEY"]
        self.engine_name = engine_name

    def prompt_to_hashable_str(self, prompt: Any) -> str:
        assert isinstance(prompt, str)
        return prompt

    def _gpt_safe_completion(self, prompts: List[str],
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
            "engine": self.engine_name,
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

        last_exc = None
        for i in range(self.MAX_ATTEMPTS):
            try:
                response = openai.Completion.create(prompt=prompts, **args_dict)
                response = json.loads(json.dumps(response))
                return response["choices"]
            # wait longer for rate limit errors
            except openai.error.RateLimitError as e:
                last_exc = e
                time.sleep(self.RATE_WAITTIME)
            except openai.error.OpenAIError as e:
                last_exc = e
                time.sleep(self.ERROR_WAITTIME)

        # make fake choices
        fake_choices = [
            [{
                "text": " OPENAI Error:" + str(last_exc),
                "logprobs": None,
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
            responses.append(meta_response)

        return responses

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

    def hash_query_request(self, prompt: Any,
                           max_tokens: int,
                            temperature: float,
                            top_p: float,
                            n: int,
                            logprobs: Optional[int],
                            stop_tokens: Optional[Union[str, List[str]]],
                            echo_prompt: bool,
                            **kwargs) -> str:
        return self._default_query_request_hash_func(prompt, max_tokens, temperature, top_p, n, logprobs, stop_tokens, echo_prompt, **kwargs)
