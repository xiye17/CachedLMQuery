from setuptools import setup, find_packages

setup(
    name='cachedllm',
    version='0.2',
    description='LM Interface with caching',
    packages=find_packages(
        include=["cachedllm", "cachedllm.*"],
        exclude=["examples"],
    ),
    install_requires=[
        'openai>=1.5.0',
        'sqlitedict',
        'jinja2',
    ],
)
