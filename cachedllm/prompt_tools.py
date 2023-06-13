from typing import (
    Any,
    List,
    Dict,
    Tuple,
    Optional,
    Union,
)

from jinja2 import Environment, BaseLoader, Template

class PromptTemplate:
    pass

class TextPromptTemplate(PromptTemplate):
    def __init__(self, template: Template):
        self.template = template

    def render(self, data: Dict[str, Any], **kwargs) -> str:
        return self.template.render(**data, **kwargs)

    @classmethod
    def from_file(cls, filename: str) -> "TextPromptTemplate":
        with open(filename, 'r') as f:
            text = f.read()
        template = Environment(loader=BaseLoader, keep_trailing_newline=True, trim_blocks=True).from_string(text)
        return cls(template)

    @classmethod
    def from_string(cls, text: str) -> "TextPromptTemplate":
        template = Environment(loader=BaseLoader, keep_trailing_newline=True, trim_blocks=True).from_string(text)
        return cls(template)
