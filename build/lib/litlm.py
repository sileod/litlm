import time
import requests
from litellm import completion
import pandas as pd
import os
import json
import hashlib
from appdirs import user_cache_dir
import re

# Fetch models
OPENROUTER_MODELS = list(pd.DataFrame(requests.get("https://openrouter.ai/api/v1/models").json()['data']).id)

# Cache directory
CACHE_DIR = os.path.join(user_cache_dir(), 'litlm')
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_dir(subdir=None):
    path = os.path.join(CACHE_DIR, subdir or 'cache')
    os.makedirs(path, exist_ok=True)
    return path

def get_cache_file(cache_dir, key):
    return os.path.join(cache_dir, f"{hashlib.md5(key.encode()).hexdigest()}.json")

def complete(prompt, model, max_tokens=64, max_retries=5, 
             silent_exception=False, free_sleep=3, cache=None, **kwargs):
    m = sorted([x for x in OPENROUTER_MODELS if model in x], key=len)[0]
    provider = f"openrouter/{m}"
    
    if cache:
        cache_dir = get_cache_dir(cache)
        cache_key = f"{m}:{prompt}"
        cache_file = get_cache_file(cache_dir, cache_key)
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)['response']

    if ":free" in m:
        time.sleep(free_sleep)
    
    try:
        response = completion(provider, [{"role": "user", "content": prompt}], max_tokens=max_tokens, max_retries=max_retries, **kwargs)
        result = response.choices[0].message.content.strip()
        
        if cache:
            with open(cache_file, 'w') as f:
                json.dump({'model': m, 'prompt': prompt, 'response': result}, f)
        return result
    except Exception as e:
        if not silent_exception:
            print(f"❗ {e}")
        return ""

def complete_m(m, **kwargs):
    return lambda x: complete(x, m, **kwargs)

def read_cache(subdir=None):
    cache_dir = get_cache_dir(subdir)
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        return pd.DataFrame(columns=['model', 'prompt', 'response'])
    
    data = [json.load(open(os.path.join(cache_dir, f), 'r')) 
            for f in os.listdir(cache_dir) if f.endswith('.json')]
    return pd.DataFrame(data)


def extract_answer(x,tag="answer", strip=True):
    """extract answer from tags"""
    #result = re.sub(r"```[a-z]+\n", "```", s)

    y=x.split(f'<{tag}>',-1)[-1].split(f'</{tag}>')[0]
    if strip:
        y=y.strip().strip('`').strip()
    return y