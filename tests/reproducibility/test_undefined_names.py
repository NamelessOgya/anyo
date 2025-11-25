import ast
import builtins
import pytest
from pathlib import Path

# Directory to check
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "src" / "exp"

class UndefinedNameChecker(ast.NodeVisitor):
    def __init__(self):
        self.undefined_names = []
        self.scopes = [set(dir(builtins))] # Start with builtins
        self.current_scope = self.scopes[0]

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.current_scope.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.current_scope.add(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.current_scope.add(node.name)
        # New scope for function
        new_scope = set()
        # Add arguments to new scope
        for arg in node.args.args:
            new_scope.add(arg.arg)
        for arg in node.args.kwonlyargs:
            new_scope.add(arg.arg)
        if node.args.vararg:
            new_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            new_scope.add(node.args.kwarg.arg)
            
        self.scopes.append(new_scope)
        self.current_scope = new_scope
        self.generic_visit(node)
        self.scopes.pop()
        self.current_scope = self.scopes[-1]

    def visit_ClassDef(self, node):
        self.current_scope.add(node.name)
        # New scope for class
        new_scope = set()
        self.scopes.append(new_scope)
        self.current_scope = new_scope
        self.generic_visit(node)
        self.scopes.pop()
        self.current_scope = self.scopes[-1]

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.current_scope.add(target.id)
            elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.current_scope.add(elt.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.current_scope.add(node.target.id)
        self.generic_visit(node)
        
    def visit_For(self, node):
        # Loop variables leak to enclosing scope in Python (except in comprehensions, handled separately)
        if isinstance(node.target, ast.Name):
            self.current_scope.add(node.target.id)
        elif isinstance(node.target, ast.Tuple) or isinstance(node.target, ast.List):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self.current_scope.add(elt.id)
        self.generic_visit(node)
        
    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                if isinstance(item.optional_vars, ast.Name):
                    self.current_scope.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.current_scope.add(node.name)
        self.generic_visit(node)

    def _visit_comprehension(self, node):
        # Comprehensions create a new scope
        new_scope = set()
        self.scopes.append(new_scope)
        self.current_scope = new_scope
        
        for generator in node.generators:
            self.visit(generator) # Visit generators to add targets to scope
            
        # Visit element/key/value in the new scope
        if hasattr(node, 'elt'):
            self.visit(node.elt)
        if hasattr(node, 'key'):
            self.visit(node.key)
        if hasattr(node, 'value'):
            self.visit(node.value)
            
        self.scopes.pop()
        self.current_scope = self.scopes[-1]

    def visit_ListComp(self, node):
        self._visit_comprehension(node)

    def visit_SetComp(self, node):
        self._visit_comprehension(node)

    def visit_DictComp(self, node):
        self._visit_comprehension(node)

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            # Check if defined in any scope
            defined = False
            for scope in reversed(self.scopes):
                if node.id in scope:
                    defined = True
                    break
            if not defined:
                self.undefined_names.append((node.id, node.lineno))
        elif isinstance(node.ctx, ast.Store):
            self.current_scope.add(node.id)
        # generic_visit is not needed for Name as it has no children

def get_python_scripts():
    return list(SCRIPTS_DIR.glob("*.py"))

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
