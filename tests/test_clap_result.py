"""Protocol checks that run without PyTorch, CLAP, or a built extension."""

import subprocess

import pytest

from clap_result import RESULT_MARKER, assert_clap_result


VALID = RESULT_MARKER + '{"calls": 1, "fallbacks": 0}\n'


@pytest.mark.parametrize("before,after", [(True, False), (False, True), (True, True)])
def test_logs_around_result(before, after):
    output = ("[CLAP_DNN] before\n" if before else "") + VALID
    output += "[CLAP_DNN] after\n" if after else ""
    result = subprocess.CompletedProcess([], 0, output, "vendor diagnostic\n")
    assert_clap_result(result, 1, 0)


def test_fallback_result():
    result = subprocess.CompletedProcess(
        [], 0, RESULT_MARKER + '{"calls": 0, "fallbacks": 1}\n', ""
    )
    assert_clap_result(result, 0, 1)


@pytest.mark.parametrize("output", [
    "[CLAP_DNN] no result\n",
    RESULT_MARKER + "not JSON\n",
    VALID + VALID,
    VALID + RESULT_MARKER + "malformed duplicate\n",
    RESULT_MARKER + '{"calls": 0, "fallbacks": 1}\n',
    RESULT_MARKER + '{"calls": 1}\n',
    RESULT_MARKER + '{"calls": true, "fallbacks": 0}\n',
    RESULT_MARKER + '{"calls": "1", "fallbacks": 0}\n',
    RESULT_MARKER + '[]\n',
])
def test_invalid_results_include_captured_output(output):
    result = subprocess.CompletedProcess([], 0, output, "vendor diagnostic\n")
    with pytest.raises(AssertionError) as error:
        assert_clap_result(result, 1, 0)
    assert output in str(error.value)
    assert result.stderr in str(error.value)


def test_child_failure_includes_captured_output():
    result = subprocess.CompletedProcess([], 1, VALID, "child failed\n")
    with pytest.raises(AssertionError) as error:
        assert_clap_result(result, 1, 0)
    assert "returncode=1" in str(error.value)
    assert VALID in str(error.value)
    assert result.stderr in str(error.value)
