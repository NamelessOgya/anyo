import ast
import inspect
import pytest
from pathlib import Path
import sys

# Import the classes to check
from src.student.trainer_baseline import SASRecTrainer
from src.teacher.trainer_ilora import iLoRATrainer
from src.distill.trainer_distill import DistillationTrainer
from src.student.models import SASRec
from src.teacher.ilora_model import iLoRAModel

# Directory to check
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "src" / "exp"

# Map class names to their actual classes
CLASSES_TO_CHECK = {
    "SASRecTrainer": SASRecTrainer,
    "iLoRATrainer": iLoRATrainer,
    "DistillationTrainer": DistillationTrainer,
    "SASRec": SASRec,
    "iLoRAModel": iLoRAModel,
}

class ArgumentChecker(ast.NodeVisitor):
    def __init__(self, script_path):
        self.script_path = script_path
        self.errors = []

    def visit_Call(self, node):
        # Check for Class(...)
        if isinstance(node.func, ast.Name) and node.func.id in CLASSES_TO_CHECK:
            self._check_arguments(node, CLASSES_TO_CHECK[node.func.id], is_constructor=True)
        
        # Check for Class.load_from_checkpoint(...)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "load_from_checkpoint":
            if isinstance(node.func.value, ast.Name) and node.func.value.id in CLASSES_TO_CHECK:
                self._check_arguments(node, CLASSES_TO_CHECK[node.func.value.id], is_constructor=False)
        
        self.generic_visit(node)

    def _check_arguments(self, node, cls, is_constructor):
        try:
            sig = inspect.signature(cls.__init__)
        except ValueError:
            return # Cannot get signature

        valid_params = set(sig.parameters.keys())
        if 'self' in valid_params:
            valid_params.remove('self')
        
        # If checking load_from_checkpoint, ignore arguments specific to it
        # load_from_checkpoint(checkpoint_path, map_location=None, hparams_file=None, strict=True, **kwargs)
        # The kwargs are passed to __init__
        ignored_args = set()
        if not is_constructor:
            # Arguments consumed by load_from_checkpoint and NOT passed to __init__
            # Based on PyTorch Lightning documentation
            ignored_args = {'checkpoint_path', 'map_location', 'hparams_file', 'strict'}

        # Check keyword arguments
        for keyword in node.keywords:
            arg_name = keyword.arg
            if arg_name is None: # **kwargs
                continue
            
            if not is_constructor and arg_name in ignored_args:
                continue

            if arg_name not in valid_params:
                # Check if __init__ accepts **kwargs
                has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if not has_kwargs:
                    self.errors.append(
                        f"Line {node.lineno}: Unexpected argument '{arg_name}' for {cls.__name__}"
                    )

def get_python_scripts():
    return list(SCRIPTS_DIR.glob("*.py"))

@pytest.mark.parametrize("script_path", get_python_scripts())
def test_argument_validation(script_path):
    """
    Static analysis to validate arguments for specific classes.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"SyntaxError in {script_path.name}: {e}")

    checker = ArgumentChecker(script_path)
    checker.visit(tree)
    
    if checker.errors:
        pytest.fail(f"Argument errors found in {script_path.name}:\n" + "\n".join(checker.errors))
