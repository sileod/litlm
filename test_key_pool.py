import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch

import litellm
import litlm
import litlm_cli


def response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))],
                           usage=None, model='test', _hidden_params={})


def test_keys_are_balanced_without_mutating_environment(monkeypatch):
    monkeypatch.setenv('POOL_A', 'first')
    monkeypatch.setenv('POOL_B', 'second')
    seen = []
    async def request(**kwargs):
        seen.append(kwargs['api_key'])
        return response()
    with patch.object(litlm, 'acompletion', request):
        batch = litlm.complete(['q'] * 6, model='albert/test', api_key_envs=['POOL_A', 'POOL_B'], show_progress=False)
    assert batch == ['ok'] * 6
    assert seen.count('first') == seen.count('second') == 3
    assert {item.key_env for item in batch} == {'POOL_A', 'POOL_B'}
    assert batch._resume_options['api_key_envs'] == ['POOL_A', 'POOL_B']


def test_exhausted_key_fails_over_within_exact_provider(monkeypatch):
    monkeypatch.setenv('POOL_A', 'empty')
    monkeypatch.setenv('POOL_B', 'working')
    seen = []
    async def request(**kwargs):
        seen.append(kwargs['api_key'])
        if kwargs['api_key'] == 'empty':
            raise litellm.RateLimitError('insufficient quota', model='test', llm_provider='openai')
        return response()
    with patch.object(litlm, 'acompletion', request):
        answer = litlm.complete('q', model='albert/test', api_key_envs=['POOL_A', 'POOL_B'], num_retries=0, show_progress=False)
    assert answer == 'ok'
    assert seen == ['empty', 'working']


def test_all_exhausted_keys_stop(monkeypatch):
    monkeypatch.setenv('POOL_A', 'empty-a')
    monkeypatch.setenv('POOL_B', 'empty-b')
    seen = []
    async def request(**kwargs):
        seen.append(kwargs['api_key'])
        raise litellm.AuthenticationError('invalid key', model='test', llm_provider='openai')
    with patch.object(litlm, 'acompletion', request):
        batch = litlm.complete(['q', 'r'], model='albert/test', api_key_envs=['POOL_A', 'POOL_B'], show_progress=False)
    assert all(item.failed for item in batch)
    assert len(seen) == 2


def test_per_key_slots_are_independent(monkeypatch):
    monkeypatch.setenv('POOL_A', 'first')
    monkeypatch.setenv('POOL_B', 'second')
    async def run():
        pool = litlm._KeyPool(['POOL_A', 'POOL_B'], rpm=1200)
        started = time.monotonic()
        slots = []
        async def reserve():
            index, _ = await pool.acquire()
            slots.append((index, time.monotonic() - started))
        await asyncio.gather(*(reserve() for _ in range(4)))
        return slots
    slots = asyncio.run(run())
    for index in [0, 1]:
        times = [elapsed for key, elapsed in slots if key == index]
        assert len(times) == 2
        assert times[1] - times[0] >= 0.04


def test_cli_passes_pool_options(monkeypatch, tmp_path):
    monkeypatch.setattr(litlm_cli.sys, 'stdin', __import__('io').StringIO('question'))
    with patch.object(litlm_cli, 'complete', return_value='ok') as complete:
        assert litlm_cli.main(['--model', 'albert/test', '--api-key-envs', 'POOL_A,POOL_B', '--per-key-rpm', '40']) == 0
    assert complete.call_args.kwargs['api_key_envs'] == ['POOL_A', 'POOL_B']
    assert complete.call_args.kwargs['per_key_rpm'] == 40


def test_checkpoint_tracks_model_but_not_key_selection():
    parser = litlm_cli._parser()
    first = parser.parse_args(['--model', 'albert/first', '--api-key-envs', 'POOL_A'])
    rotated = parser.parse_args(['--model', 'albert/first', '--api-key-envs', 'POOL_B'])
    changed = parser.parse_args(['--model', 'albert/second', '--api-key-envs', 'POOL_A'])
    assert litlm_cli._key('question', first) == litlm_cli._key('question', rotated)
    assert litlm_cli._key('question', first) != litlm_cli._key('question', changed)


def test_background_progress_reports_eta(capsys):
    async def request(**kwargs):
        await asyncio.sleep(0.02)
        return response()
    with patch.object(litlm, 'acompletion', request):
        result = litlm.complete(['q', 'r'], model='albert/test', show_progress=False, progress_interval=0.001)
    assert result == ['ok', 'ok']
    assert 'ETA ' in capsys.readouterr().err
