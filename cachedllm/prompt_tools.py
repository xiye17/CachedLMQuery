from typing import (
    Any,
    List,
    Dict,
    Tuple,
    Optional,
    Union,
)

import re

from abc import ABC, abstractmethod
from jinja2 import Environment, BaseLoader, Template

class Jinja2PromptTemplate(ABC):
    """Base class for prompt templates."""
    def __init__(self, template: Template):
        self.template = template

    @abstractmethod
    def render(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Render the template with the given data."""
        pass

    @classmethod
    def from_file(cls, filename: str) -> "TextPromptTemplate":
        """Load a template from a file."""
        with open(filename, 'r') as f:
            text = f.read()
        template = Environment(loader=BaseLoader, keep_trailing_newline=True, trim_blocks=True).from_string(text)
        return cls(template)

    @classmethod
    def from_string(cls, text: str) -> "TextPromptTemplate":
        """Load a template from a string."""
        template = Environment(loader=BaseLoader, keep_trailing_newline=True, trim_blocks=True).from_string(text)
        return cls(template)

class TextPromptTemplate(Jinja2PromptTemplate):
    """Template for text prompts."""

    def render(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        if data is not None:
            return self.template.render(**data, **kwargs)
        return self.template.render(**kwargs)


class OAIChatPromptTemplate(Jinja2PromptTemplate):
    """Template for messages prompts."""

    # MEG_BLOCK_START = '[MSGBLOCK]'
    # MEG_BLOCK_END = '[/MSGBLOCK]'
    # trailing newline of the end block will be removed
    pattern = re.compile(r'\[MSGBLOCK\](.*?)\n\[/MSGBLOCK\]', re.DOTALL)
    # role can be letters and digits but has to start with a letter
    role_pattern = re.compile(r'-[a-zA-Z][a-zA-Z0-9]*')

    def render(self, data: Optional[Dict[str, Any]] = None, **kwargs) -> List[Dict]:
        if data is not None:
            text = self.template.render(**data, **kwargs)
        else:
            text = self.template.render(**kwargs)
        # split into blocks
        matches = self.pattern.findall(text)

        # parse into openai format
        messages = []
        for match in matches:
            assert self.role_pattern.match(match)
            role, msg = match.split('\n', 1)
            role = role[1:]
            messages.append({'role': role, 'content': msg})

        assert len(messages) > 0 and messages[0]['role'] == 'system'

        return messages

    @staticmethod
    def pretty_print(messages: List[Dict]) -> str:
        """Pretty print messages."""
        return '\n===========================\n'.join([f"{m['role']}:\n{m['content']}" for m in messages])
