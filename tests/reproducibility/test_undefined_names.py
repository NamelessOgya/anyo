import ast
import builtins
import pytest
from pathlib import Path

# Directories to check
SRC_DIR = Path(__file__).parent.parent.parent / "src"
DIRS_TO_CHECK = [
    SRC_DIR / "exp",
    SRC_DIR / "student",
    SRC_DIR / "teacher",
    SRC_DIR / "distill",
]

def get_python_scripts():
    scripts = []
    for directory in DIRS_TO_CHECK:
        if directory.exists():
            scripts.extend(list(directory.glob("**/*.py")))
    return scripts

@pytest.mark.parametrize("script_path", get_python_scripts())
def test_undefined_names(script_path):
    """
    Static analysis to find undefined names in scripts.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"SyntaxError in {script_path.name}: {e}")

    checker = UndefinedNameChecker()
    checker.visit(tree)
    
    if checker.undefined_names:
        errors = [f"Line {lineno}: {name}" for name, lineno in checker.undefined_names]
        pytest.fail(f"Undefined names found in {script_path.name}:\n" + "\n".join(errors))
