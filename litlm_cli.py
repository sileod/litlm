import argparse
import json
import sys
from contextlib import redirect_stdout

from litlm import complete


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
        "cost": getattr(result, "cost", None),
        "usage": _jsonable(getattr(result, "usage", None)),
        "reasoning": getattr(result, "reasoning", None),
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
    return data


def _inputs(args, parser):
    if args.input_jsonl:
        if args.prompt:
            parser.error("--input-jsonl cannot be combined with a positional prompt")
        lines = [line for line in sys.stdin.read().splitlines() if line.strip()]
        if not lines:
            parser.error("--input-jsonl requires JSON values on stdin")
        try:
            return [json.loads(line) for line in lines], True
        except json.JSONDecodeError as error:
            parser.error(f"invalid JSONL input: {error}")

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
        description="Small command-line interface to litlm.complete().",
    )
    parser.add_argument("prompt", nargs="*", help="prompt text; stdin is used when omitted")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("--system")
    parser.add_argument("--json", action="store_true", help="request and parse JSON output")
    parser.add_argument("--output", choices=("text", "json", "jsonl"))
    parser.add_argument("--input-jsonl", action="store_true", help="read one JSON input per stdin line")
    parser.add_argument("--caching", action="store_true")
    parser.add_argument("--prompt-cache", nargs="?", const=True, default=False, type=_value)
    parser.add_argument("--cache-control", type=_value, metavar="JSON")
    parser.add_argument("--num-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--attempt-timeout", type=float)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--rpm", type=float)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--fallback", action="append", dest="fallbacks")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _print(results, output, batch):
    values = list(results) if batch else [results]
    records = [_record(value) for value in values]

    if output == "jsonl":
        for record in records:
            print(json.dumps(record, ensure_ascii=False))
    elif output == "json":
        payload = records if batch else records[0]
        print(json.dumps(payload, ensure_ascii=False))
    elif batch:
        for value in values:
            print(value)
    elif isinstance(results, (dict, list)):
        print(json.dumps(_jsonable(results), ensure_ascii=False))
    else:
        print(results)

    return 1 if any(record["failed"] for record in records) else 0


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
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
        rpm=args.rpm,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        fallbacks=args.fallbacks,
        **kwargs,
    )

    try:
        if args.debug:
            with redirect_stdout(sys.stderr):
                result = complete(inputs, **call)
        else:
            result = complete(inputs, **call)
    except Exception as error:
        print(f"litlm: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    return _print(result, output, batch)


if __name__ == "__main__":
    raise SystemExit(main())
