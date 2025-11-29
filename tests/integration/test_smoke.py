
import pytest
import subprocess
import sys
from pathlib import Path

# Define the scripts to test
SCRIPTS = [
    "src/exp/run_teacher.py",
    "src/exp/run_bigrec.py",
    "src/exp/run_distill.py",
    "src/exp/run_bigrec_inference.py"
]

@pytest.mark.parametrize("script_path", SCRIPTS)
def test_script_help(script_path):
    """
    Smoke test to check if the script can be executed with --help.
    This catches SyntaxErrors, ImportErrors, and NameErrors at module level.
    """
    cmd = [sys.executable, script_path, "--help"]
    
    # Run the command
    # check=True raises CalledProcessError if exit code is non-zero
    # capture_output=True hides stdout/stderr unless failure
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Script {script_path} failed with exit code {e.returncode}.\nStderr: {e.stderr}")
    except FileNotFoundError:
        pytest.fail(f"Script {script_path} not found.")

