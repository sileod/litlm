import json
from unittest.mock import patch

import httpx
import pytest
import litlm
import litlm_cli
import litlm_decisions

QUESTIONS={'eligible':{'type':'choice','instructions':'Is this usable training data?',
                       'criteria':{'yes':None,'no':None,'unknown':None}}}


def transport(monkeypatch, handler):
    original=httpx.AsyncClient
    monkeypatch.setattr(litlm_decisions.httpx,'AsyncClient',lambda **kwargs:
        original(transport=httpx.MockTransport(handler),**kwargs))
    monkeypatch.setenv('OPENROUTER_API_KEY','private-test-key')


def response(request):
    return httpx.Response(200,json={'model':'dated-model','usage':{'cost':0.001,'input_tokens':10},
        'answers':{'eligible':{'type':'choice','choice':'yes','probabilities':{'yes':0.8,'no':0.1,'unknown':0.1}}}})


def test_decisions_preserve_states_distributions_usage_and_route(monkeypatch):
    calls=[]
    def handle(request):calls.append(request);return response(request)
    transport(monkeypatch,handle)
    states=[{'text':'source one','images':[{'bytes':'encoded-image'}]},'source two']
    out=litlm.complete(states,decisions=QUESTIONS,model='openrouter/perplexity/pplx-decider-v1.1-27b',show_progress=False)
    assert [json.loads(r.content)['state'] for r in calls]==states
    assert all(str(r.url)=='https://openrouter.ai/api/alpha/decisions' for r in calls)
    assert all(r.headers['Authorization']=='Bearer private-test-key' for r in calls)
    assert out[0].data['answers']['eligible']['probabilities']['unknown']==0.1
    assert out[0].data['model']=='dated-model' and out[0].cost==0.001
    assert out[0].model_used=='openrouter/perplexity/pplx-decider-v1.1-27b'
    assert 'private-test-key' not in json.dumps(litlm_cli._record(out[0]))


def test_decision_failure_resume_retries_only_failed_state(monkeypatch):
    calls=[];fail=[True]
    def handle(request):
        state=json.loads(request.content)['state'];calls.append(state)
        if state=='bad' and fail[0]:return httpx.Response(200,json={'answers':{}})
        return response(request)
    transport(monkeypatch,handle)
    out=litlm.complete(['good','bad'],decisions=QUESTIONS,model='openrouter/test',show_progress=False)
    assert len(out.failures)==1
    fail[0]=False;out.resume()
    assert not out.failures and calls==['good','bad','bad']


def test_decisions_cache_reuses_success_but_questions_change_key(tmp_path,monkeypatch):
    calls=[]
    def handle(request):calls.append(request);return response(request)
    transport(monkeypatch,handle);monkeypatch.setattr(litlm,'CACHE_DIR',tmp_path)
    kwargs=dict(decisions=QUESTIONS,model='openrouter/test',show_progress=False,caching=True)
    litlm.complete('source',**kwargs);litlm.complete('source',**kwargs)
    assert len(calls)==1
    changed={'eligible':dict(QUESTIONS['eligible'],instructions='A different question')}
    litlm.complete('source',**dict(kwargs,decisions=changed))
    assert len(calls)==2
    assert 'private-test-key' not in ''.join(p.read_text() for p in (tmp_path/'decisions').glob('*.json'))


@pytest.mark.parametrize('bad',[
    {'answers':{'eligible':{'type':'choice','probabilities':{'yes':1,'no':0}}}},
    {'answers':{'eligible':{'type':'choice','probabilities':{'yes':0.5,'no':0.5,'unknown':0.5}}}},
    {'answers':{'eligible':{'type':'noul','noul':0.5}}},
])
def test_decisions_reject_invalid_distributions(bad):
    with pytest.raises(ValueError):litlm_decisions.validate_answers(bad,QUESTIONS)


def test_decisions_validate_noul_and_score():
    questions={'n':{'type':'noul','instructions':'Yes?'},'s':{'type':'score','instructions':'Quality?', 'criteria':['low','high']}}
    litlm_decisions.validate_questions(questions)
    data={'answers':{'n':{'type':'noul','noul':0.4},'s':{'type':'score','probabilities':{'0':0.2,'1':0.8},'legend':{'0':'low','1':'high'}}}}
    litlm_decisions.validate_answers(data,questions)
    data['answers']['s']['legend']={'0':'high','1':'low'}
    with pytest.raises(ValueError):litlm_decisions.validate_answers(data,questions)


def test_decisions_cli_checkpoint_includes_questions_and_json_states(tmp_path,monkeypatch,capsys):
    calls=[]
    def handle(request):calls.append(request);return response(request)
    transport(monkeypatch,handle)
    inputs=tmp_path/'states.jsonl';inputs.write_text('{"source":"one"}\n{"source":"two"}\n')
    questions=tmp_path/'questions.json';questions.write_text(json.dumps(QUESTIONS));output=tmp_path/'out.jsonl'
    args=['-i',str(inputs),'--decisions',str(questions),'-m','openrouter/test','-o',str(output),'--num-retries','0']
    assert litlm_cli.main(args)==0 and len(calls)==2
    assert litlm_cli.main(args)==0 and len(calls)==2
    assert [json.loads(r.content)['state'] for r in calls]==[{'source':'one'},{'source':'two'}]
    changed={'eligible':dict(QUESTIONS['eligible'],instructions='A different question')}
    questions.write_text(json.dumps(changed))
    assert litlm_cli.main(args)==0 and len(calls)==4
    assert all('answers' in json.loads(l)['data'] for l in output.read_text().splitlines())


def test_decisions_refuse_chat_options_and_implicit_routing():
    with pytest.raises(ValueError):litlm.complete('state',model='test',decisions=QUESTIONS)
    with pytest.raises(ValueError):litlm.complete('state',model='openrouter/test',decisions=QUESTIONS,system='Chat system')


def test_native_state_question_bundles_pack_different_questions(monkeypatch):
    calls=[]
    def handle(request):
        data=json.loads(request.content);calls.append(data)
        return httpx.Response(200,json={'answers':{name:{'type':'noul','noul':0.7} for name in data['questions']}})
    transport(monkeypatch,handle)
    bundles=[{'state':'shared source A','questions':{'train':{'type':'noul','instructions':'Has train?'},
              'data':{'type':'noul','instructions':'Has data?'}}},
             {'state':{'source':'B'},'questions':{'fit':{'type':'noul','instructions':'Fits?'}}}]
    out=litlm.complete(bundles,decisions=True,model='openrouter/test',show_progress=False)
    assert calls[0]['state']=='shared source A' and len(calls[0]['questions'])==2
    assert calls[1]['state']=={'source':'B'} and set(out[1].data['answers'])=={'fit'}


def test_ensemble_retains_members_weights_and_disagreement(monkeypatch):
    def handle(request):
        model=json.loads(request.content)['model']
        data=response(request).json()
        data['model']=model
        if model=='two':data['answers']['eligible']['probabilities']={'yes':0.2,'no':0.7,'unknown':0.1}
        return httpx.Response(200,json=data)
    transport(monkeypatch,handle)
    out=litlm.ensemble('source',['openrouter/one','openrouter/two'],QUESTIONS,weights=[1,3],show_progress=False)
    assert out.data['answers']['eligible']['probabilities']['yes']==pytest.approx(0.35)
    assert out.data['agreement']['eligible'] is False and len(out.data['members'])==2
    assert out.cost==0.002
    assert [v.model_used for v in out.member_results]==['openrouter/one','openrouter/two']


def test_incomplete_ensemble_does_not_silently_average_remaining_members(monkeypatch):
    def handle(request):
        if json.loads(request.content)['model']=='two':return httpx.Response(401,json={'error':'Invalid key'})
        return response(request)
    transport(monkeypatch,handle)
    out=litlm.ensemble('source',['openrouter/one','openrouter/two'],QUESTIONS,num_retries=0,show_progress=False)
    assert out.failed and not out.member_results[0].failed and out.member_results[1].failed


def test_decisions_cli_accepts_packed_bundles(tmp_path,monkeypatch):
    calls=[]
    def handle(request):calls.append(json.loads(request.content));return response(request)
    transport(monkeypatch,handle)
    path=tmp_path/'bundles.jsonl';path.write_text(json.dumps({'state':{'source':'one'},'questions':QUESTIONS})+'\n')
    out=tmp_path/'out.jsonl'
    assert litlm_cli.main(['-i',str(path),'--decisions','-m','openrouter/test','-o',str(out)])==0
    assert calls==[{'model':'test','state':{'source':'one'},'questions':QUESTIONS}]


def test_invalid_native_answer_is_preserved_separately_from_accepted_cache(tmp_path,monkeypatch):
    bad={'model':'dated-model','usage':{'cost':0.001},'answers':{'eligible':{'type':'choice','probabilities':{'yes':0.5,'no':0.5,'unknown':0.5}}}}
    transport(monkeypatch,lambda request:httpx.Response(200,json=bad))
    monkeypatch.setattr(litlm,'CACHE_DIR',tmp_path)
    result=litlm.complete(['source','another source'],decisions=QUESTIONS,model='openrouter/test',caching=True,show_progress=False)[0]
    record=litlm_cli._record(result)
    assert record['failed'] and record['raw_response']==bad
    assert '1.5' in record['error']['message'] and 'eligible' in record['error']['message']
    assert 'private-test-key' not in json.dumps(record)
    assert not list((tmp_path/'decisions').glob('*.json'))
