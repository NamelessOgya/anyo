import pytest
import os
import glob
from pathlib import Path

# Define the directory containing experiment scripts
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "src" / "exp"

def get_python_scripts():
    """Get all Python scripts in src/exp"""
    return list(SCRIPTS_DIR.glob("*.py"))

@pytest.mark.parametrize("script_path", get_python_scripts())
def test_no_direct_hydraconfig_usage(script_path):
    """
    Verify that experiment scripts do not use HydraConfig.get() directly.
    We use load_hydra_config() which uses hydra.compose(), so HydraConfig is not set.
    Accessing it causes 'ValueError: HydraConfig was not set'.
    """
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check for forbidden patterns
    forbidden_patterns = [
        "HydraConfig.get()",
        "hydra.core.hydra_config.HydraConfig.get()"
    ]
    
    for pattern in forbidden_patterns:
        if pattern in content:
            pytest.fail(
                f"Found forbidden pattern '{pattern}' in {script_path.name}. "
                "Do not use HydraConfig.get() in scripts using load_hydra_config(). "
                "Use cfg.run.dir or other config values directly."
            )


