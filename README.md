# CachedLMQuery
Lightweight toolkit for constructing prompting, quering LLMs, and caching the responses.

## Setup
```sh
pip install -e .
export OPENAI_KEY=YOURKEY
```

## Usage

**Examples**
```sh
cd examples
python example.py
```

**Currently Supporting**
* Query and cache OpenAI vanilla Completion engines (e.g., `text-davinci-003`). Also support caching logprobs of both prompts and completions for more sophisticated usage.
* Query and cache OpenAI Chat Completion engines (e.g., `gpt-3.5-turbo`).
* TODO: huggingface models and prompting tools

**Cache System**

SqliteDict for OpenAI queries


