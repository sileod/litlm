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


def test_cli_param_cannot_duplicate_explicit_option():
    with patch.object(litlm_cli, "complete"):
        try:
            litlm_cli.main(["hello", "--param", "model=other"])
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("expected argparse error")


# --- agent-oriented features, run end to end against a stubbed provider ---
import asyncio

import litlm


def _stub(reply):
    async def acompletion(**kwargs):
        await asyncio.sleep(0)
        message = SimpleNamespace(content=reply(kwargs["messages"][-1]["content"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None,
                               model="m", _hidden_params={"response_cost": 0.25})
    return acompletion


def test_cli_lines_input_file_with_fields(tmp_path, capsys):
    source = tmp_path / "prompts.txt"
    source.write_text("a\n\nb\n")
    with patch.object(litlm, "acompletion", _stub(str.upper)):
        code = litlm_cli.main(["--lines", "-i", str(source), "-m", "openrouter/x", "--fields", "text"])

    captured = capsys.readouterr()
    assert code == 0
    assert [json.loads(l) for l in captured.out.splitlines()] == [
        {"index": 0, "text": "A"}, {"index": 1, "text": "B"}]
    assert "litlm: 2/2 ok" in captured.err


def test_cli_template_rows_and_choices(tmp_path, capsys):
    source = tmp_path / "rows.jsonl"
    source.write_text('{"t": "great"}\n{"t": "awful"}\n')
    seen = []

    def reply(content):
        seen.append(content)
        return "Positive!" if "great" in content else "negative"

    with patch.object(litlm, "acompletion", _stub(reply)):
        code = litlm_cli.main(["-i", str(source), "-t", "Review: {t}", "--choices", "positive,negative",
                               "-m", "openrouter/x", "--output", "text"])

    assert code == 0
    assert capsys.readouterr().out == "positive\nnegative\n"
    assert seen[0].startswith("Review: great")


def test_cli_out_checkpoint_resumes_only_unfinished(tmp_path, capsys):
    source = tmp_path / "prompts.txt"
    source.write_text("a\nb\nc\n")
    out = tmp_path / "out.jsonl"
    calls = []

    def flaky(content):
        calls.append(content)
        if content == "b" and calls.count("b") == 1:
            return "not json"
        return json.dumps({"p": content})

    args = ["--lines", "-i", str(source), "-o", str(out), "--json", "-m", "openrouter/x"]
    with patch.object(litlm, "acompletion", _stub(flaky)):
        first = litlm_cli.main(args)
        first_out = capsys.readouterr().out
        second = litlm_cli.main(args)
        second_out = capsys.readouterr().out

    assert first == 1 and first_out.startswith("2/3 ok (0 reused), 1 failed")
    assert second == 0 and second_out.startswith("3/3 ok (2 reused), 0 failed")
    assert sorted(calls) == ["a", "b", "b", "c"]
    records = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["index"] for r in records] == [0, 1, 2]
    assert [r["data"] for r in records] == [{"p": "a"}, {"p": "b"}, {"p": "c"}]


def test_cli_out_checkpoint_ignores_changed_inputs(tmp_path, capsys):
    source = tmp_path / "prompts.txt"
    out = tmp_path / "out.jsonl"
    with patch.object(litlm, "acompletion", _stub(str.upper)):
        source.write_text("a\nb\n")
        litlm_cli.main(["--lines", "-i", str(source), "-o", str(out), "-m", "openrouter/x"])
        source.write_text("a\nz\n")
        litlm_cli.main(["--lines", "-i", str(source), "-o", str(out), "-m", "openrouter/x"])

    assert capsys.readouterr().out.splitlines()[-1].startswith("2/2 ok (1 reused)")
    assert [json.loads(l)["text"] for l in out.read_text().splitlines()] == ["A", "Z"]


def test_cli_routes_and_doctor(monkeypatch, capsys):
    assert litlm_cli.main(["--routes", "-m", "openrouter/a/b"]) == 0
    assert capsys.readouterr().out == "openrouter/a/b\n"
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")
    assert litlm_cli.main(["--doctor"]) == 0
    out = capsys.readouterr().out
    assert "openrouter: set" in out and "secret-value" not in out
