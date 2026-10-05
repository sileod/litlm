import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import litellm
import pytest
import litlm


def response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))],
                           usage=None, model='test', _hidden_params={})


def test_controller_grows_within_ceiling_and_ignores_old_congestion():
    controller = litlm._AdaptiveConcurrency(20)
    for _ in range(8): controller.observe(None, 0)
    assert controller.limit == 16
    for _ in range(16): controller.observe(None, 0)
    assert controller.limit == 20
    error = litellm.RateLimitError('too many requests',model='test',llm_provider='openai')
    controller.observe(error, 0)
    assert controller.limit == 10
    for _ in range(20): controller.observe(error, 0)
    assert controller.limit == 10  # failures from the previous wave
    controller.observe(error, 1)
    assert controller.limit == 5


def test_repeated_timeouts_back_off_but_unrelated_errors_do_not():
    controller = litlm._AdaptiveConcurrency(32)
    for _ in range(6): controller.observe(None, 0)
    for _ in range(2): controller.observe(TimeoutError('timeout'), 0)
    assert controller.limit == 4
    quota = litellm.RateLimitError('insufficient quota',model='test',llm_provider='openai')
    for error in [ValueError('bad format'), quota]: controller.observe(error, 1)
    assert controller.limit == 4


def test_scheduler_adapts_without_dropping_callbacks_or_crossing_ceiling(monkeypatch):
    monkeypatch.setenv('KEY', 'secret')
    active = peak = 0
    async def request(**kwargs):
        nonlocal active, peak
        assert 'adaptive_concurrency' not in kwargs
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.001)
        active -= 1
        return response()
    settled = []
    with patch.object(litlm, 'acompletion', request):
        batch = litlm.complete(['q']*80,model='albert/test',api_key_envs=['KEY'],
            adaptive_concurrency=True,max_concurrency=16,show_progress=False,
            on_result=lambda index,result: settled.append(index))
    assert 8 < peak <= 16
    assert sorted(settled) == list(range(80))
    assert batch == ['ok']*80
    assert batch.tuning['concurrency'] == 16
    assert batch._resume_options['adaptive_concurrency'] is True


def test_adaptive_mode_still_respects_key_pacing(monkeypatch):
    monkeypatch.setenv('KEY_A', 'a')
    monkeypatch.setenv('KEY_B', 'b')
    starts = {'a':[], 'b':[]}
    async def request(**kwargs):
        starts[kwargs['api_key']].append(asyncio.get_running_loop().time())
        return response()
    with patch.object(litlm, 'acompletion', request):
        batch = litlm.complete(['q']*8,model='albert/test',api_key_envs=['KEY_A','KEY_B'],
            per_key_rpm=3000,adaptive_concurrency=True,max_concurrency=16,show_progress=False)
    assert batch == ['ok']*8
    for times in starts.values():
        assert len(times) == 4
        assert all(right-left >= .017 for left,right in zip(times,times[1:]))


def test_server_errors_back_off_and_resume_only_failures():
    calls=0
    async def request(**kwargs):
        nonlocal calls
        calls += 1
        failed = calls <= 8
        await asyncio.sleep(.001)
        if failed:
            raise litellm.InternalServerError('overloaded',model='test',llm_provider='openai')
        return response()
    with patch.object(litlm,'acompletion',request):
        batch=litlm.complete(['q']*80,model='direct/openai/test',adaptive_concurrency=True,
                             max_concurrency=16,num_retries=0,show_progress=False)
        assert len(batch.failures) == 8
        assert batch.tuning['decreases'] == 1
        batch.resume()
    assert calls == 88
    assert batch == ['ok']*80
    assert 'concurrency=' in batch.summary()


@pytest.mark.parametrize('ceiling',[None,0,-1,1.5,True])
def test_adaptive_mode_requires_finite_positive_ceiling(ceiling):
    with pytest.raises(ValueError,match='ceiling'):
        litlm.complete(['q'],adaptive_concurrency=True,max_concurrency=ceiling,show_progress=False)


def test_cli_exposes_adaptive_option_and_reuses_checkpoints(monkeypatch,tmp_path):
    import litlm_cli
    source=tmp_path/'in.jsonl'; source.write_text('"q"\n"r"\n')
    output=tmp_path/'out.jsonl'
    seen=[]
    async def request(**kwargs):
        seen.append(kwargs);return response()
    arguments=['-i',str(source),'-o',str(output),'-m','direct/openai/test','--no-progress',
               '--adaptive-concurrency','--max-concurrency','16']
    with patch.object(litlm,'acompletion',request):
        assert litlm_cli.main(arguments) == 0
        assert litlm_cli.main([a for a in arguments if a!='--adaptive-concurrency']) == 0
    assert len(seen) == 2
