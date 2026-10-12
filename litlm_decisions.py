"""OpenRouter typed Decisions transport; scheduling/results stay in litlm."""
import asyncio
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import uuid

import httpx
import litellm


def validate_questions(questions):
    if not isinstance(questions, dict) or not questions:
        raise ValueError('decisions must be a nonempty named question mapping')
    for name, question in questions.items():
        if not isinstance(name,str) or not isinstance(question,dict) or question.get('type') not in ('noul','choice','score'):
            raise ValueError('Each decision needs a name and type noul, choice or score')
        if not isinstance(question.get('instructions'),str) or not question['instructions'].strip():
            raise ValueError('Each decision needs instructions')
        criteria=question.get('criteria')
        if question['type']=='choice' and (not isinstance(criteria,dict) or len(criteria)<2):
            raise ValueError('choice needs at least two named criteria')
        if question['type']=='score' and (not isinstance(criteria,list) or len(criteria)<2):
            raise ValueError('score needs an ordered criteria list')


def validate_answers(data, questions, *, sum_tolerance=None):
    answers=data.get('answers')
    if not isinstance(answers,dict) or set(answers)!=set(questions):
        raise ValueError('Decisions response has missing or unexpected questions')
    for name,question in questions.items():
        answer=answers[name]
        if answer.get('type')!=question['type']:raise ValueError('Decision answer type mismatch')
        if question['type']=='noul':values=[answer.get('noul')]
        else:
            keys=list(question['criteria']) if question['type']=='choice' else [str(i) for i in range(len(question['criteria']))]
            probabilities=answer.get('probabilities',{})
            if set(probabilities)!=set(keys):raise ValueError('Decision criteria mismatch')
            values=[probabilities[k] for k in keys]
            if question['type']=='score' and answer.get('legend') and answer['legend']!={str(i):v for i,v in enumerate(question['criteria'])}:
                raise ValueError('Decision score legend mismatch')
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError('Invalid decision probabilities')
        # Native routes may quantize probabilities to hundredths; retain their values.
        rounded=all(abs(v-round(v,2))<1e-12 for v in values)
        tolerance=sum_tolerance if sum_tolerance is not None else min(0.02,0.005*len(values)) if rounded else 0.001
        if question['type']!='noul' and abs(sum(values)-1)>tolerance+1e-12:
            raise ValueError(f'Decision {name!r} probabilities sum to {sum(values):.8g}, expected one')


async def adecision_completion(*, model, state, questions, api_key=None, timeout=60, num_retries=0,
                               cache_dir=None, api_base=None, base_url=None, extra_headers=None, **kwargs):
    if kwargs:raise ValueError('Unsupported Decisions parameters: '+', '.join(sorted(kwargs)))
    if not api_key:raise ValueError('An OpenRouter API key is required for decisions')
    base=(base_url or api_base or 'https://openrouter.ai').rstrip('/')
    if base.endswith('/api/v1'):base=base[:-7]
    elif base.endswith('/api'):base=base[:-4]
    url=base+'/api/alpha/decisions'
    payload=dict(model=model.removeprefix('openrouter/'),state=state,questions=questions)
    path=None
    if cache_dir:
        key=hashlib.sha256((url+json.dumps(payload,sort_keys=True,ensure_ascii=False)).encode()).hexdigest()
        path=Path(cache_dir)/'decisions'/(key+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    cache_hit=bool(path and path.exists())
    if cache_hit:data=json.loads(path.read_text())
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(num_retries+1):
                response=await client.post(url,json=payload,headers={**(extra_headers or {}),'Authorization':'Bearer '+api_key})
                if response.status_code in (429,500,502,503,504) and attempt<num_retries:
                    await asyncio.sleep(min(2**attempt,30));continue
                if response.is_error:
                    message=f'Decisions HTTP {response.status_code}: '+response.text[:500].replace(api_key,'[REDACTED]')
                    error=litellm.AuthenticationError if response.status_code in (401,403) else litellm.RateLimitError if response.status_code in (402,429) else litellm.BadRequestError
                    raise error(message=message,model=model,llm_provider='openrouter',response=response)
                data=response.json();break
        try:validate_answers(data,questions)
        except ValueError as error:
            error.raw_response=data
            raise
        if path:
            temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            temporary.write_text(json.dumps(data,ensure_ascii=False));temporary.replace(path)
    validate_answers(data,questions)
    return SimpleNamespace(model=data.get('model',model),usage=data.get('usage',{}),_litlm_decisions=data,
        _hidden_params={'response_cost':0.0} if cache_hit else {},
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data,ensure_ascii=False)),finish_reason='stop')])


async def aensemble(inputs, models, decisions, weights=None, **kwargs):
    """Typed decision ensemble; retain members and flag disagreement, never hide failures.

    Each model uses litlm's scheduler; model batches run sequentially to avoid
    independent key pools competing for the same quota. Weights are averaging
    weights, not a claim of calibrated or independent probabilities.
    """
    from litlm import acomplete, Failure, Text
    if isinstance(models,str) or len(models)<2 or len(set(models))!=len(models):
        raise ValueError('An ensemble needs at least two distinct model routes')
    weights=list(weights) if weights is not None else [1.0]*len(models)
    if len(weights)!=len(models) or any(not math.isfinite(w) or w<=0 for w in weights):
        raise ValueError('Each ensemble member needs a finite positive weight')
    weights=[w/sum(weights) for w in weights]
    states=inputs if isinstance(inputs,list) else [inputs]
    if not states:return []
    # A row failure stays attached to that row; successful member calls are retained.
    members=[]
    for model in models:
        try:members.append(await acomplete(states,model=model,decisions=decisions,**kwargs))
        except Exception as error:members.append([Failure(error,None,state,model) for state in states])
    results=[]
    for i,state in enumerate(states):
        values=[batch[i] for batch in members]
        if any(v.failed for v in values):
            result=Failure(RuntimeError('Incomplete decision ensemble; inspect member_results'),None,state,'ensemble')
        else:
            questions=state['questions'] if decisions is True else decisions
            answers={};agreement={}
            for name,question in questions.items():
                entries=[v.data['answers'][name] for v in values]
                if question['type']=='noul':
                    probabilities=[e['noul'] for e in entries]
                    answers[name]=dict(type='noul',noul=sum(w*p for w,p in zip(weights,probabilities)))
                    agreement[name]=len({p>=0.5 for p in probabilities})==1
                else:
                    keys=list(question['criteria']) if question['type']=='choice' else [str(j) for j in range(len(question['criteria']))]
                    probabilities={k:sum(w*e['probabilities'][k] for w,e in zip(weights,entries)) for k in keys}
                    answers[name]=dict(type=question['type'],probabilities=probabilities)
                    if question['type']=='choice':answers[name]['choice']=max(keys,key=probabilities.get)
                    else:answers[name]['legend']={str(j):v for j,v in enumerate(question['criteria'])}
                    agreement[name]=len({max(keys,key=e['probabilities'].get) for e in entries})==1
            data=dict(model='ensemble',answers=answers,members=[v.data for v in values],agreement=agreement,
                      aggregation=dict(method='weighted_mean',models=list(models),weights=weights),
                      usage=dict(cost=sum(float(v.cost or 0) for v in values)))
            # A weighted mean preserves the bounded rounding errors of validated members.
            validate_answers(data,questions,sum_tolerance=0.02)
            response=SimpleNamespace(model='ensemble',usage=data['usage'],
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)),finish_reason='stop')])
            result=Text(json.dumps(data),response,None,state);result.data=data
        result.member_results=values;results.append(result)
    return results if isinstance(inputs,list) else results[0]


def ensemble(inputs, models, decisions, weights=None, **kwargs):
    """Synchronous counterpart to aensemble()."""
    from litlm import _run_sync
    return _run_sync(aensemble(inputs,models,decisions,weights,**kwargs))
