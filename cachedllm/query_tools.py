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

import abc
import hashlib
import json


from tqdm import tqdm
from sqlitedict import SqliteDict


class LLMEngineBase(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def complete_batch(self, prompts: List[Any],
                           max_tokens: int,
                            temperature: float,
                            top_p: float,
                            n: int,
                            logprobs: Optional[int],
                            stop_tokens: Optional[Union[str, List[str]]],
                            echo_prompt: bool,
                            **kwargs) -> List[Any]:
        raise NotImplementedError

    def hash_query_request(self, prompt: Any,
                           max_tokens: int,
                            temperature: float,
                            top_p: float,
                            n: int,
                            logprobs: Optional[int],
                            stop_tokens: Optional[Union[str, List[str]]],
                            echo_prompt: bool,
                            **kwargs) -> str:
        hash_str = self.model_args_to_str() + self.query_args_to_str(
            max_tokens, temperature, top_p, n, logprobs, stop_tokens, echo_prompt, **kwargs
            ) + self.prompt_to_hashable_str(prompt)
        hash_key = self.hash_of_string(hash_str)
        return hash_key

    @abc.abstractmethod
    def model_args_to_str(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def query_args_to_str(self,
                          max_tokens: int,
                            temperature: float,
                            top_p: float,
                            n: int,
                            logprobs: Optional[int],
                            stop_tokens: Optional[Union[str, List[str]]],
                            echo_prompt: bool,
                            **kwargs) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def prompt_to_hashable_str(self, prompt: Any) -> str:
        raise NotImplementedError

    def hash_of_string(self, s: str) -> str:
        return hashlib.sha1(s.encode('utf-8')).hexdigest()

    def _default_query_args_to_str(self,
                          max_tokens: int,
                            temperature: float,
                            top_p: float,
                            n: int,
                            logprobs: Optional[int],
                            stop_tokens: Optional[Union[str, List[str]]],
                            echo_prompt: bool,
                            **kwargs) -> str:
        return (f"MAX_TOKENS={max_tokens}\n"
                f"TEMPREATURE={float(temperature)}\n"
                f"TOP_P={float(top_p)}\n"
                f"N={n}\n"
                f"LOGPROBS={logprobs}\n"
                f"STOP_TOKENS={repr(stop_tokens)}\n"
                f"ECHO_PROMPT={echo_prompt}\n")


class Cachebase(abc.ABC):
    def lookup(self, key: str) -> Any:
        raise NotImplementedError

    def write(self, key: str, value: Any) -> None:
        raise NotImplementedError


class SqliteCache(Cachebase):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.cache = SqliteDict(cache_dir, autocommit=True)

    def lookup(self, key: str) -> Any:
        if key in self.cache:
            return self.cache[key]
        else:
            return None

    def write(self, key: str, value: Any) -> None:
        self.cache[key] = value


# class FileCache(Cachebase): TODO: FileSystem Based Cache for storing large files


class CachedQueryInterface(abc.ABC):
    def __init__(self, engine: LLMEngineBase, cache_dir: str):
        self.engine = engine
        self.cache_dir = cache_dir
        self.cache = SqliteCache(cache_dir)

    def complete_prompts(self,
                         # the parameters above controls the query results, hence controlling the information to be cached
                         prompts: List[Any],
                         max_tokens: int = 0,
                         temperature: float = 0.0,
                         top_p: float = 1.0,
                         n: int = 1,
                         logprobs: Optional[int] = None,
                         stop_tokens: Optional[Union[str, List[str]]] = None,
                         echo_prompt: bool = False,
                         # the parameters below controls query speed
                         batch_size: int = 1,
                         # other parameters
                         **kwargs) -> List[Any]:
        # bachify for speeding up
        query_pool = []
        responses = []
        for i, prompt in enumerate(prompts):
            hash_key, respone = self.response_lookup(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n,
                                                    logprobs=logprobs, stop_tokens=stop_tokens, echo_prompt=echo_prompt, **kwargs)
            if respone is None:
                query_pool.append((i, hash_key, prompt))
            responses.append(respone)

        num_batches = (len(query_pool) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), total=num_batches, desc="querying"):
            batch_start = batch_idx * batch_size
            batch_end = min(len(query_pool), batch_start + batch_size)
            batch = query_pool[batch_start:batch_end]
            batch_indices, batch_hash_keys, batch_prompts = zip(*batch)

            batch_responses = self.engine.complete_batch(batch_prompts, max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n,
                                                         logprobs=logprobs, stop_tokens=stop_tokens, echo_prompt=echo_prompt, **kwargs)

            for i, hash_key, response in zip(batch_indices, batch_hash_keys, batch_responses):
                self.response_writeback(hash_key, response)
                responses[i] = response

        return responses

    def response_lookup(self, prompt: Any,
                        max_tokens: int,
                        temperature: float,
                        top_p: float,
                        n: int,
                        logprobs: Optional[int],
                        stop_tokens: Optional[Union[str, List[str]]],
                        echo_prompt: bool,
                        **kwargs) -> Tuple[str, Any]:
        hash_key = self.engine.hash_query_request(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n,
                                                  logprobs=logprobs, stop_tokens=stop_tokens, echo_prompt=echo_prompt, **kwargs)
        result = self.cache.lookup(hash_key)
        return hash_key, result

    def response_writeback(self, hash_key: str, response: Any):
        self.cache.write(hash_key, response)

