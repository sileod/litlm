import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import litlm_cli


class Result(str):
    def __new__(cls, text="OK", failed=False):
        obj = super().__new__(cls, text)
        obj.failed = failed
        obj.model_used = "provider/model"
        obj.cost = 0.001
        obj.usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1)
        obj.reasoning = None
        if failed:
            obj.error = RuntimeError("limited")
        return obj


def test_cli_scalar_text(capsys):
    with patch.object(litlm_cli, "complete", return_value=Result("hello")) as complete:
        code = litlm_cli.main(["hello", "world", "--model", "test-model"])

    assert code == 0
    assert capsys.readouterr().out == "hello\n"
    assert complete.call_args.args == ("hello world",)
    assert complete.call_args.kwargs["model"] == "test-model"


def test_cli_json_output_has_stable_metadata(capsys):
    with patch.object(litlm_cli, "complete", return_value=Result("hello")):
        code = litlm_cli.main(["hello", "--output", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload == {
        "text": "hello",
        "model": "provider/model",
        "cost": 0.001,
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        "reasoning": None,
        "failed": False,
    }


def test_cli_reads_stdin(monkeypatch, capsys):
    monkeypatch.setattr(litlm_cli.sys, "stdin", io.StringIO("from stdin"))
    with patch.object(litlm_cli, "complete", return_value=Result("OK")) as complete:
        code = litlm_cli.main([])

    assert code == 0
    assert capsys.readouterr().out == "OK\n"
    assert complete.call_args.args == ("from stdin",)


def test_cli_jsonl_batch_defaults_to_jsonl(monkeypatch, capsys):
    monkeypatch.setattr(litlm_cli.sys, "stdin", io.StringIO('"a"\n"b"\n'))
    with patch.object(litlm_cli, "complete", return_value=[Result("A"), Result("B")]) as complete:
        code = litlm_cli.main(["--input-jsonl"])

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0
    assert [line["text"] for line in lines] == ["A", "B"]
    assert complete.call_args.args == (["a", "b"],)


def test_cli_jsonl_message_objects_become_batch_conversations(monkeypatch, capsys):
    source = '{"role":"user","content":"a"}\n{"role":"user","content":"b"}\n'
    monkeypatch.setattr(litlm_cli.sys, "stdin", io.StringIO(source))
    with patch.object(litlm_cli, "complete", return_value=[Result("A"), Result("B")]) as complete:
        code = litlm_cli.main(["--input-jsonl"])

    assert code == 0
    capsys.readouterr()
    assert complete.call_args.args == ([
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}],
    ],)


def test_cli_failure_sets_nonzero_exit(capsys):
    with patch.object(litlm_cli, "complete", return_value=Result("", failed=True)):
        code = litlm_cli.main(["hello", "--output", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["failed"] is True
    assert payload["error"] == {"type": "RuntimeError", "message": "limited"}


def test_cli_param_parses_json_values(capsys):
    with patch.object(litlm_cli, "complete", return_value=Result()) as complete:
        code = litlm_cli.main(["hello", "--param", "top_p=0.8", "--param", "seed=42"])

    assert code == 0
    assert capsys.readouterr().out == "OK\n"
    assert complete.call_args.kwargs["top_p"] == 0.8
    assert complete.call_args.kwargs["seed"] == 42
