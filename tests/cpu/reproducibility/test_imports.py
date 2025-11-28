import pytest
import importlib
import pkgutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def get_all_modules():
    """
    Recursively find all modules in the src directory.
    """
    src_path = PROJECT_ROOT / "src"
    modules = []
    
    for path in src_path.rglob("*.py"):
        if path.name == "__init__.py":
            continue
            
        # Convert path to module name
        # e.g. src/student/trainer_baseline.py -> src.student.trainer_baseline
        relative_path = path.relative_to(PROJECT_ROOT)
        module_name = str(relative_path).replace("/", ".").replace(".py", "")
        modules.append(module_name)
        
    return modules

@pytest.mark.parametrize("module_name", get_all_modules())
def test_can_import_module(module_name):
    """
    Try to import the module. This will catch ModuleNotFoundError and ImportError
    caused by invalid imports at the top level of the module.
    """
    try:
        importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.fail(f"Failed to import {module_name}: {e}")
    except Exception as e:
        # Other errors (e.g. runtime errors during import) are also failures
        # but we might want to be lenient if they are side effects of running code at import time
        # For now, let's fail on them too as good practice is to avoid side effects at import
        pytest.fail(f"Error during import of {module_name}: {e}")
