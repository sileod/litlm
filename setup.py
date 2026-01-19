from setuptools import setup

setup(
    name="litlm",
    version="0.2.0",
    description="Minimalist litellm wrapper for simpler requests and better openrouter support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Damien Sileo",
    url="https://github.com/sileod/litlm",
    py_modules=["litlm"],
    install_requires=[
        "litellm",
        "tqdm",          # Required for progress bars
        "requests",      # Required for fetching OpenRouter models
        "appdirs",       # Required for cross-platform cache paths
        "nest_asyncio",  # Required for Jupyter/Loop patching
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)