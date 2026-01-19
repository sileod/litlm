# litlm

**litlm** is a minimalist, lazy-programmer wrapper around [LiteLLM](https://github.com/BerriAI/litellm). It removes the boilerplate from LLM API calls, handles async loops automatically (even in Jupyter), adds smart OpenRouter model resolution, and integrates natively with Pandas.

## Installation

```bash
pip install litlm

```

## Quick Start

```python
from litlm import complete

# 1. Simple String (Synchronous feel, but Async under the hood)
res = complete("What is 2+2?")
print(res) 
# > "4"

# 2. Batch Processing (Parallel execution + Progress Bar)
questions = ["Meaning of life?", "Capital of France?", "Who is TURING?"]
answers = complete(questions)
# > [Completing] 100%|██████████| 3/3 [00:01<00:00, 2.15it/s]

```

## Key Features

### 1. The "Text" Object

The result acts exactly like a string, but carries all metadata (usage, cost, raw response) and history.

```python
res = complete("Write a haiku")

print(res)              # It prints the content directly
print(res.usage)        # Access token usage
print(res.call_id)      # Unique ID for the call
print(res.reasoning)    # Access reasoning content (e.g. for DeepSeek-R1 / o1)

```

### 2. Smart OpenRouter Resolution

Stop typing `openrouter/openai/gpt-4o...`. **litlm** fuzzy-matches your model name against the OpenRouter list and finds the shortest valid model ID.

```python
# Automatically finds 'openrouter/openai/gpt-4o-mini'
complete("Hello", model="gpt-4o-mini") 

# Automatically finds 'openrouter/anthropic/claude-3.5-sonnet'
complete("Hello", model="sonnet")

```

### 3. Pandas & Numpy Support

Pass DataFrames, Series, or Numpy arrays directly.

```python
import pandas as pd

df = pd.DataFrame({"prompts": ["Joke about cats", "Joke about dogs"]})

# Returns a list of Text objects, keeping order
df["results"] = complete(df["prompts"]) 

```

### 4. Built-in Caching & Robustness

Avoid re-running expensive prompts during development.

```python
# Caches result to disk (user_cache_dir)
# If you run this again, it returns instantly without API cost.
res = complete("Complex query...", caching=True)

# Built-in retries and timeout handling
res = complete("Flaky API...", num_retries=5)

```

### 5. Session History

Access your past generation without cluttering your variables.

```python
from litlm import get_history

# Get the last result
last_res = get_history()

# Get a specific result by index
first_res = get_history(0)

```

## Advanced Configuration

You can pass any standard `litellm` argument (temperature, max_tokens, etc.).

```python
complete(
    "Hello", 
    model="deepseek/deepseek-chat", 
    temperature=0.7, 
    max_tokens=500,
    api_key="sk-..." # Optional if env var is set
)

```

## License

MIT

```

```
