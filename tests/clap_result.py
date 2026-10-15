"""Read subprocess counters without depending on vendor log ordering."""

import json
import subprocess


RESULT_MARKER = "__CLAP_DNN_RESULT__="


def assert_clap_result(
    result: subprocess.CompletedProcess,
    expected_calls: int,
    expected_fallbacks: int,
) -> None:
    captured = (
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0, captured
    records = [
        line[len(RESULT_MARKER):]
        for line in result.stdout.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    assert len(records) == 1, f"Expected exactly one CLAP result\n{captured}"
    try:
        counts = json.loads(records[0])
    except ValueError as exc:
        raise AssertionError(f"Malformed CLAP result\n{captured}") from exc
    expected = {"calls": expected_calls, "fallbacks": expected_fallbacks}
    assert (
        isinstance(counts, dict)
        and counts.keys() == expected.keys()
        and all(type(value) is int for value in counts.values())
        and counts == expected
    ), f"Expected CLAP result {expected}, got {counts!r}\n{captured}"
