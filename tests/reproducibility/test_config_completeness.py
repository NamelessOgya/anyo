import ast
import pytest
import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# Directories to check for scripts using 'cfg'
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "src" / "exp"
CONF_DIR = PROJECT_ROOT / "conf"

class ConfigUsageCollector(ast.NodeVisitor):
    def __init__(self, config_var_name="cfg"):
        self.config_var_name = config_var_name
        self.used_keys = set()

    def visit_Attribute(self, node):
        # Flatten the attribute chain: cfg.a.b.c
        parts = []
        curr = node
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        
        if isinstance(curr, ast.Name) and curr.id == self.config_var_name:
            # Found a chain starting with cfg
            # parts are reversed: [c, b, a] -> a.b.c
            full_key = ".".join(reversed(parts))
            self.used_keys.add(full_key)
        
        self.generic_visit(node)

def get_python_scripts():
    return list(SCRIPTS_DIR.glob("*.py"))

def get_experiment_configs():
    experiment_dir = CONF_DIR / "experiment"
    return [f.stem for f in experiment_dir.glob("*.yaml")]

@pytest.fixture(scope="module")
def collected_usages():
    """
    Collects all usages of 'cfg.key...' from scripts.
    Returns a dict mapping script path to a set of used keys.
    """
    usages = {}
    collector = ConfigUsageCollector(config_var_name="cfg")
    
    for script_path in get_python_scripts():
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            tree = ast.parse(code)
            # Reset collector for each file? Or accumulate?
            # If we want to report which file has the issue, we should do it per file.
            # But for efficiency, we can collect all unique keys used across all scripts.
            # Let's collect per file for better error messages.
            file_collector = ConfigUsageCollector(config_var_name="cfg")
            file_collector.visit(tree)
            if file_collector.used_keys:
                usages[script_path.name] = file_collector.used_keys
        except SyntaxError:
            pass # Should be caught by other tests
            
    return usages

@pytest.mark.parametrize("experiment_name", get_experiment_configs())
def test_config_completeness(experiment_name, collected_usages):
    """
    Test 39: [Config Completeness] Verify that all config keys used in scripts (e.g. cfg.train.batch_size)
    are present in the experiment configuration.
    """
    # Load the experiment config
    try:
        with hydra.initialize(version_base=None, config_path="../../conf"):
            cfg = hydra.compose(config_name="config", overrides=[f"experiment={experiment_name}"])
            OmegaConf.set_struct(cfg, True)
    except Exception as e:
        pytest.fail(f"Failed to load experiment config '{experiment_name}': {e}")

    # Check each used key against the loaded config
    for script_name, keys in collected_usages.items():
        for key in keys:
            # Check if key exists in cfg
            # key is dot-separated, e.g. "train.batch_size"
            try:
                # OmegaConf.select might be too permissive even with throw_on_missing=True?
                # Let's use direct attribute access which respects struct mode.
                parts = key.split('.')
                curr = cfg
                for part in parts:
                    curr = getattr(curr, part)
            except Exception as e:
                # If select fails, it might be missing.
                pytest.fail(f"Key '{key}' used in '{script_name}' is missing from experiment config '{experiment_name}'. Error: {e}")
