import asyncio
import inspect
import io
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import litlm


def _response(content="ok", cost=None):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    hidden = {"response_cost": cost} if cost is not None else {}
    return SimpleNamespace(
        choices=[choice], usage=None, model="test-model", _hidden_params=hidden
    )


class RetryTimeoutTests(unittest.TestCase):
    def setUp(self):
        litlm._HISTORY.clear()
        litlm._COSTS.clear()
        litlm._FAILURES.clear()
        litlm._MODEL_USED.clear()

    def test_wrapper_does_not_apply_timeout_to_entire_retry_operation(self):
        calls = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            await asyncio.sleep(0.02)
            return _response()

        with patch.object(litlm, "acompletion", fake_acompletion):
            result = litlm.complete(
                "hello",
                model="openrouter/test-model",
                timeout=0.01,
                num_retries=4,
                show_progress=False,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(calls[0]["timeout"], 0.01)
        self.assertEqual(calls[0]["num_retries"], 4)

    def test_timeout_error_is_normalized_after_litellm_finishes_retrying(self):
        async def fake_acompletion(**kwargs):
            raise TimeoutError("provider timeout")

        with patch.object(litlm, "acompletion", fake_acompletion):
            with self.assertRaisesRegex(
                TimeoutError,
                "litlm timed out after 7s while calling openrouter/test-model",
            ):
                litlm.complete(
                    "hello",
                    model="openrouter/test-model",
                    timeout=7,
                    num_retries=2,
                    show_progress=False,
                )

    def test_batch_failure_returns_string_and_does_not_abort_other_items(self):
        async def fake_acompletion(**kwargs):
            content = kwargs["messages"][0]["content"]
            if content == "bad":
                raise TimeoutError("provider timeout")
            return _response(content, cost=0.01)

        stderr = io.StringIO()
        with patch.object(litlm, "acompletion", fake_acompletion), redirect_stderr(stderr):
            result = litlm.complete(
                ["first", "bad", "last"],
                model="openrouter/test-model",
                show_progress=False,
                num_retries=0,
            )

        self.assertEqual([str(item) for item in result], ["first", "", "last"])
        self.assertEqual([item.failed for item in result], [False, True, False])
        self.assertIsInstance(result[1], str)
        self.assertIsInstance(result[1].error, TimeoutError)
        self.assertEqual(litlm.get_failures(result[1].call_id), [result[1]])
        self.assertIs(litlm.get_failure(result[1].call_id), result[1].error)
        self.assertIn("⚠ 1/3 failed (33.3%)", stderr.getvalue())
        self.assertIn("TimeoutError: litlm timed out after 60s", stderr.getvalue())
        self.assertIn("litlm.get_failure(0)", stderr.getvalue())

    def test_get_failure_defaults_to_latest_error(self):
        first = litlm.Failure(ValueError("first"), 0, [], "test-model")
        second = litlm.Failure(RuntimeError("second"), 1, [], "test-model")
        litlm._FAILURES.extend([first, second])

        self.assertIs(litlm.get_failure(), second.error)
        self.assertIs(litlm.get_failure(0), first.error)
        self.assertIsNone(litlm.get_failure(99))

    def test_representative_errors_are_grouped_and_cropped(self):
        failures = [
            litlm.Failure(ValueError("x" * 500), 0, [], "test-model"),
            litlm.Failure(ValueError("another message"), 0, [], "test-model"),
        ]

        lines = litlm._representative_failure_lines(failures, max_chars=80)

        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("  2× ValueError:"))
        self.assertLessEqual(len(lines[0]), 82)
        self.assertTrue(lines[0].endswith("…"))

    def test_compact_error_breakdown_is_counted_and_bounded(self):
        errors = [
            ValueError("x" * 500),
            TimeoutError("provider timeout"),
            ValueError("another message"),
        ]

        summary = litlm._compact_error_breakdown(errors, max_chars=80)

        self.assertTrue(summary.startswith("ValueError×2, TimeoutError×1 |"))
        self.assertLessEqual(len(summary), 80)
        self.assertTrue(summary.endswith("…"))

    def test_batch_resume_retries_only_failures_with_overrides(self):
        attempts = []

        async def fake_acompletion(**kwargs):
            content = kwargs["messages"][0]["content"]
            attempts.append((content, kwargs["timeout"]))
            if content == "bad" and kwargs["timeout"] < 120:
                raise TimeoutError("too short")
            return _response("recovered" if content == "bad" else content)

        with patch.object(litlm, "acompletion", fake_acompletion), redirect_stderr(io.StringIO()):
            result = litlm.complete(
                ["good", "bad"],
                model="openrouter/test-model",
                timeout=5,
                num_retries=0,
                show_progress=False,
            )
            resumed = result.resume(timeout=120)

        self.assertIsInstance(result, litlm.BatchResult)
        self.assertIs(resumed, result)
        self.assertEqual([str(item) for item in result], ["good", "recovered"])
        self.assertEqual(result.failures, [])
        self.assertEqual(attempts.count(("good", 5)), 1)
        self.assertEqual(attempts.count(("bad", 5)), 1)
        self.assertEqual(attempts.count(("bad", 120)), 1)

    def test_complete_exposes_common_typed_parameters(self):
        signature = inspect.signature(litlm.complete)

        self.assertIn("reasoning_effort", signature.parameters)
        self.assertIn("temperature", signature.parameters)
        self.assertIsNot(signature.parameters["timeout"].annotation, inspect.Parameter.empty)
        self.assertIsNot(signature.return_annotation, inspect.Signature.empty)

    def test_session_cost_includes_rows_older_than_one_day(self):
        litlm._COSTS.extend([
            {
                "time": datetime.now() - timedelta(days=2),
                "model": "old-model",
                "cost": 1.0,
                "usage": None,
            },
            {
                "time": datetime.now(),
                "model": "new-model",
                "cost": 2.0,
                "usage": None,
            },
        ])

        self.assertEqual(
            litlm.cost_breakdown("session"),
            {"old-model": 1.0, "new-model": 2.0},
        )
        self.assertEqual(litlm.cost_breakdown("day"), {"new-model": 2.0})


if __name__ == "__main__":
    unittest.main()
