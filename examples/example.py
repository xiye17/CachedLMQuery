import os
import openai
import json

from jinja2 import Environment, BaseLoader
from cachedllm.query_tools import *
from cachedllm.prompt_tools import *
from cachedllm.openai_engine import *

def read_json(fname):
    with open(fname) as f:
        return json.load(f)

def dump_json(obj, fname, indent=None):
    with open(fname, 'w', encoding='utf-8') as f:
        return json.dump(obj, f, indent=indent)

def example_openai_text():
    data = read_json("demo_data.json")
    with open("demo_template.tpl") as f:
        prompt_tpl = f.read()

    prompt_tpl = Environment(loader=BaseLoader).from_string(prompt_tpl)
    prompts = [prompt_tpl.render(**d) for d in data]

    query_interface = CachedQueryInterface(
        OpenAICompletionEngine("code-davinci-002"),
        cache_dir="gsm.sqlite",
    )

    responses = query_interface.complete_prompts(
        prompts,
        max_tokens=160,
        temperature=0.0,
        n=2,
        # logprobs=1,
        stop_tokens="\n\n",
        # echo_prompt=True,
        batch_size=2,
    )

    for r in responses:
        print(json.dumps(r, indent=2))


def example_openai_chat():
    data = read_json("demo_data.json")
    with open("demo_template.tpl") as f:
        prompt_tpl = f.read()

    prompt_tpl = Environment(loader=BaseLoader).from_string(prompt_tpl)
    prompts = [prompt_tpl.render(**d) for d in data]

    query_interface = CachedQueryInterface(
        OpenAIChatEngine("gpt-3.5-turbo"),
        # OpenAIChatEngine("gpt-4"),
        cache_dir="gsm.sqlite",
    )

    system_message = "You are a helpful assistant that answers the math questions. Example input-output will be provided."
    messages = [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": p}
        ]
        for p in prompts
    ]

    responses = query_interface.complete_prompts(
        messages,
        max_tokens=320,
        temperature=0.5,
        n=1,
        logprobs=None,
        echo_prompt=False,
    )

    for r in responses:
        print(json.dumps(r, indent=2))


example_openai_text()
example_openai_chat()

