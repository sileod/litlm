# litlm

**litlm** is a minimalist, lazy-programmer wrapper around [LiteLLM](https://github.com/BerriAI/litellm). It removes the boilerplate from LLM API calls, handles async loops automatically (even in Jupyter), adds smart OpenRouter model resolution, and integrates natively with Pandas.

## Installation

```bash
pip install litlm

```

## Quick Start

```python
import os
from litlm import complete, cost_breakdown

os.environ["OPENROUTER_API_KEY"] = "sk-or-..." 

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
print(res.model_used)   # See which fallback route actually answered
print(res.cost)         # LiteLLM/OpenRouter cost when available
print(res.call_id)      # Unique ID for the call
print(res.reasoning)    # Access reasoning content (e.g. for DeepSeek-R1 / o1)

```

### 2. Smart Provider Fallback

Stop typing provider prefixes. **litlm** tries the cheapest useful route first: NVIDIA NIM, then OpenRouter `:free`, then paid OpenRouter. It fuzzy-matches names against the OpenRouter list and automatically adds `:free` when needed.

```python
# Automatically finds 'openrouter/openai/gpt-4o-mini'
complete("Hello", model="gpt-4o-mini") 

# Automatically finds 'openrouter/anthropic/claude-3.5-sonnet'
complete("Hello", model="sonnet")

# Also works when you already know the provider/model slug
complete("Hello", model="meta-llama/llama-3.3-70b-instruct")

# Prefer "latest" variants when fuzzy matching, e.g. "haiku" -> haiku-latest
complete("Hello", model="haiku")

```

### 3. Small Conveniences

```python
# System prompt shortcut
complete("Summarize this", system="Be concise.")

# JSON mode returns parsed JSON
data = complete("Return {'topic': string} as JSON", json=True)

# Lightweight in-memory spend summary for this Python session
cost_breakdown("day")          # cost by model over the last day
cost_breakdown("week", by="day")

```

### 4. Pandas & Numpy Support

Pass DataFrames, Series, or Numpy arrays directly.

```python
import pandas as pd

df = pd.DataFrame({"prompts": ["Joke about cats", "Joke about dogs"]})

# Returns a list of Text objects, keeping order
df["results"] = complete(df["prompts"]) 

```

### 5. Prompt Caching & Robustness

Use OpenRouter prompt caching for long repeated context, or opt into LiteLLM's local response cache during development.

```python
# OpenRouter prompt caching: top-level cache_control passthrough
res = complete("Question over a long stable context...", prompt_cache=True)

# 1-hour TTL for providers that support it
res = complete("Question over a long stable context...", prompt_cache="1h")

# Full passthrough when you want exact control
res = complete("Question...", cache_control={"type": "ephemeral", "ttl": "1h"})

# Old LiteLLM response cache remains available, but is off by default.
# If you run this again, it returns instantly without API cost.
res = complete("Complex query...", caching=True)

# Built-in retries and timeout handling
res = complete("Flaky API...", num_retries=5)

```

### 6. Session History

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
