import argparse
import hashlib
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

from litlm import complete, doctor, routes


DEFAULT_MODEL = "openrouter/openai/gpt-4.1-nano"


def _value(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def _params(values):
    out = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--param expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("--param key cannot be empty")
        out[key] = _value(value)
    return out


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    for name in ("model_dump", "dict"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                return _jsonable(method())
            except Exception:
                pass
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _record(result):
    failed = bool(getattr(result, "failed", False))
    data = {
        "text": str(result) if isinstance(result, str) else None,
        "model": getattr(result, "model_used", None),
        "cost": _jsonable(getattr(result, "cost", None)),
        "usage": _jsonable(getattr(result, "usage", None)),
        "reasoning": _jsonable(getattr(result, "reasoning", None)),
        "failed": failed,
    }
    if failed:
        error = getattr(result, "error", None)
        data["error"] = {
            "type": type(error).__name__ if error is not None else None,
            "message": str(error) if error is not None else None,
        }
    elif isinstance(result, (dict, list)):
        data["data"] = _jsonable(result)
    else:
        attrs = getattr(result, "__dict__", {})
        if attrs.get("key_env"):
            data["key_env"] = attrs['key_env']
            data["latency_s"] = attrs.get('latency_s')
        if "data" in attrs:
            # Text settled with json=True (seen by on_result before parsing is unwrapped).
            data["data"] = _jsonable(attrs["data"])
        if attrs.get("tool_calls"):
            data["tool_calls"] = _jsonable(attrs["tool_calls"])
    return data


def _select(record, fields):
    if not fields:
        return record
    return {k: v for k, v in record.items() if k in fields or k == "index"}


def _read_batch(args):
    if args.input and args.input != "-":
        return Path(args.input).read_text(encoding="utf-8")
    return sys.stdin.read()


def _inputs(args, parser):
    if args.lines and args.input_jsonl:
        parser.error("--lines and --input-jsonl are mutually exclusive")
    if args.input or args.input_jsonl or args.lines:
        if args.prompt:
            parser.error("batch input cannot be combined with a positional prompt")
        try:
            lines = [line for line in _read_batch(args).splitlines() if line.strip()]
        except OSError as error:
            parser.error(f"cannot read --input: {error}")
        if not lines:
            parser.error("batch input is empty")
        if args.lines:
            return lines, True
        try:
            values = [json.loads(line) for line in lines]
        except json.JSONDecodeError as error:
            parser.error(f"invalid JSONL input: {error}")
        if all(isinstance(value, str) for value in values):
            return values, True
        if args.template:
            if all(isinstance(value, (str, dict)) for value in values):
                return values, True
            parser.error("with --template, JSONL lines must be strings or objects")
        if all(isinstance(value, (dict, list)) for value in values):
            conversations = [value if isinstance(value, list) else [value] for value in values]
            if all(all(isinstance(message, dict) for message in conversation) for conversation in conversations):
                return conversations, True
        parser.error("JSONL lines must all be strings or message conversations")

    if args.prompt:
        return " ".join(args.prompt), False
    if not sys.stdin.isatty():
        prompt = sys.stdin.read()
        if prompt:
            return prompt, False
    parser.error("provide a prompt or pipe one on stdin")


def _parser():
    parser = argparse.ArgumentParser(
        prog="litlm",
        description="Small command-line interface to litlm.complete(). "
        "Answers go to stdout; progress, summaries, and errors go to stderr.",
        epilog="examples:\n"
        "  litlm 'Capital of France?'\n"
        "  litlm --lines --input prompts.txt --out answers.jsonl   # checkpointed, rerun to resume\n"
        "  litlm --input rows.jsonl --template 'Review: {text}' --choices pos,neg --fields text\n"
        "  litlm --routes -m deepseek-v4-flash\n"
        "  litlm --doctor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="prompt text; stdin is used when omitted")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("--system")
    parser.add_argument("--json", action="store_true", help="request and parse JSON output")
    parser.add_argument("--output", choices=("text", "json", "jsonl"))
    parser.add_argument("--input-jsonl", action="store_true", help="read one JSON input per stdin line")
    parser.add_argument("-i", "--input", metavar="PATH", help="read a batch from PATH (JSONL, or lines with --lines)")
    parser.add_argument("--lines", action="store_true", help="each non-empty input line is a raw prompt")
    parser.add_argument("-o", "--out", metavar="PATH",
                        help="write JSONL records to PATH as items settle; rerunning skips finished items "
                        "and prints only a one-line summary")
    parser.add_argument("-t", "--template", help="str.format template; JSONL objects fill named fields, strings fill {input}")
    parser.add_argument("--choices", type=lambda s: [c.strip() for c in s.split(",") if c.strip()],
                        metavar="A,B,...", help="normalize each answer to exactly one label")
    parser.add_argument("--fields", type=lambda s: {f.strip() for f in s.split(",") if f.strip()},
                        metavar="F,...", help="keep only these JSON record fields on stdout, e.g. text,cost")
    parser.add_argument("--routes", action="store_true", help="print the provider routes for --model and exit")
    parser.add_argument("--doctor", action="store_true", help="print which provider keys are set and exit")
    parser.add_argument("--caching", action="store_true")
    parser.add_argument("--prompt-cache", nargs="?", const=True, default=False, type=_value)
    parser.add_argument("--cache-control", type=_value, metavar="JSON")
    parser.add_argument("--num-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--attempt-timeout", type=float)
    parser.add_argument("--max-concurrency", type=int, default=64, help="0 means unbounded")
    parser.add_argument("--adaptive-concurrency", action="store_true",
                        help="adapt concurrency below --max-concurrency; keep configured RPM ceilings")
    parser.add_argument("--rpm", type=float)
    parser.add_argument("--api-key-envs", type=lambda s: [name.strip() for name in s.split(",") if name.strip()],
                        help="comma-separated environment names of interchangeable keys; requires an exact route")
    parser.add_argument("--per-key-rpm", type=float, help="maximum request starts per minute for each pooled key")
    parser.add_argument("--progress-interval", type=float, default=30,
                        help="seconds between stderr progress/ETA updates (default 30)")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--fallback", action="append", dest="fallbacks")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _print(results, output, batch, fields=None):
    values = list(results) if batch else [results]
    records = [_record(value) for value in values]
    if batch:
        records = [{"index": i, **record} for i, record in enumerate(records)]
    shown = [_select(record, fields) for record in records]

    if output == "jsonl":
        for record in shown:
            print(json.dumps(record, ensure_ascii=False))
    elif output == "json":
        payload = shown if batch else shown[0]
        print(json.dumps(payload, ensure_ascii=False))
    elif batch:
        for value in values:
            print(value)
    elif isinstance(results, (dict, list)):
        print(json.dumps(_jsonable(results), ensure_ascii=False))
    else:
        print(results)

    if batch:
        summary = getattr(results, "summary", None)
        if callable(summary):
            print(f"litlm: {summary()}", file=sys.stderr)
    return 1 if any(record["failed"] for record in records) else 0


def _key(value, args):
    spec = [value, args.template, args.choices, args.system, args.json,
            args.model, args.max_tokens, args.temperature, args.reasoning_effort, args.fallbacks,
            [param for param in args.param if param.split('=', 1)[0] not in {'api_key', 'api_key_envs'}]]
    blob = json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _load_checkpoint(path, keys):
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            index = record["index"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if (isinstance(index, int) and 0 <= index < len(keys)
                and record.get("key") == keys[index] and not record.get("failed")):
            done[index] = record
    return done


def _run_checkpointed(inputs, call, args, path):
    """Run only unfinished items, appending each record to `path` as it settles."""
    keys = [_key(value, args) for value in inputs]
    records = _load_checkpoint(path, keys)
    reused = len(records)
    pending = [i for i in range(len(inputs)) if i not in records]
    error = None
    if pending:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as sink:
            def on_result(j, result):
                i = pending[j]
                record = {"index": i, "key": keys[i], **_record(result)}
                records[i] = record
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()

            try:
                if args.debug:
                    with redirect_stdout(sys.stderr):
                        complete([inputs[i] for i in pending], on_result=on_result, **call)
                else:
                    complete([inputs[i] for i in pending], on_result=on_result, **call)
            except Exception as exc:
                error = exc

    # Compact the append log: one record per index, in input order.
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as sink:
        for i in sorted(records):
            sink.write(json.dumps(records[i], ensure_ascii=False) + "\n")
    os.replace(tmp, path)

    if error is not None:
        print(f"litlm: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    values = list(records.values())
    ok = sum(not r.get("failed") for r in values)
    failed = sum(bool(r.get("failed")) for r in values)
    missing = len(inputs) - len(values)
    cost = sum(float(r.get("cost") or 0) for r in values if not r.get("failed"))
    line = f"{ok}/{len(inputs)} ok ({reused} reused), {failed} failed"
    if missing:
        line += f", {missing} missing"
    print(f"{line}, cost=${cost:.6f} -> {path}")
    return 1 if failed or missing else 0


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.doctor:
        status = doctor()
        for name, present in status.items():
            print(f"{name}: {'set' if present else 'missing'}")
        return 0 if any(status.values()) else 1
    if args.routes:
        try:
            for route in routes(args.model, args.fallbacks):
                print(route)
        except Exception as error:
            print(f"litlm: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        return 0
    inputs, batch = _inputs(args, parser)
    output = args.output or ("jsonl" if batch else "text")

    try:
        kwargs = _params(args.param)
    except ValueError as error:
        parser.error(str(error))

    call = dict(
        model=args.model,
        system=args.system,
        json=args.json,
        show_progress=not args.no_progress and sys.stderr.isatty(),
        caching=args.caching,
        prompt_cache=args.prompt_cache,
        cache_control=args.cache_control,
        num_retries=args.num_retries,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        attempt_timeout=args.attempt_timeout,
        debug=args.debug,
        max_concurrency=args.max_concurrency,
        adaptive_concurrency=args.adaptive_concurrency,
        rpm=args.rpm,
        api_key_envs=args.api_key_envs,
        per_key_rpm=args.per_key_rpm,
        progress_interval=args.progress_interval,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        fallbacks=args.fallbacks,
        template=args.template,
        choices=args.choices,
    )
    duplicate = set(call) & set(kwargs)
    if duplicate:
        parser.error(f"--param duplicates explicit option: {sorted(duplicate)[0]}")
    call.update(kwargs)

    if args.out:
        return _run_checkpointed(inputs if batch else [inputs], call, args, Path(args.out))

    try:
        if args.debug:
            with redirect_stdout(sys.stderr):
                result = complete(inputs, **call)
        else:
            result = complete(inputs, **call)
    except Exception as error:
        print(f"litlm: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    return _print(result, output, batch, args.fields)


if __name__ == "__main__":
    raise SystemExit(main())
