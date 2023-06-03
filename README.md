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
* Query and cache OpenAI vanilla completion engines (e.g., `text-davinci-001`). Also support caching Logprobs of both prompts and completions can be cached for more sophisticated usage.
* TODO: prompting tools
* TODO: chat series models and huggingface models

**Cache System**

SqliteDict for OpenAI queries


