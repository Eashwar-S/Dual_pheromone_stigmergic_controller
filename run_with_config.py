import ast
import yaml
import sys
from pathlib import Path

def run_simulation(script_path, config_path="config.yaml"):
    script_path = Path(script_path)
    if not script_path.exists():
        print(f"Error: {script_path} does not exist.")
        return

    # Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Combine common and script-specific configurations
    sim_name = script_path.stem
    overrides = config.get("common", {}).copy()
    overrides.update(config.get(sim_name, {}))

    # Parse the target simulation script into an AST
    with open(script_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    tree = ast.parse(source)

    # Dynamically inject parameters from YAML by modifying the AST
    class ConfigInjector(ast.NodeTransformer):
        def visit_Assign(self, node):
            # Only override simple assignments (e.g., VARIABLE = value)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if var_name in overrides:
                    val = overrides[var_name]
                    # Make sure it's a primitive type
                    if val is None or isinstance(val, (int, float, str, bool)):
                        # ast.Constant is available in Python 3.8+
                        node.value = ast.Constant(value=val)
            return node

    injector = ConfigInjector()
    tree = injector.visit(tree)
    ast.fix_missing_locations(tree)

    # Compile the modified AST
    sys.path.insert(0, str(script_path.parent))
    code = compile(tree, filename=str(script_path), mode="exec")
    
    print(f"🚀 Running '{sim_name}' with parameters from '{config_path}':")
    for k, v in overrides.items():
        print(f"   - {k} = {v}")
    print("-" * 50)
        
    # Execute the script in the __main__ context so the 'if __name__ == "__main__":' block runs
    namespace = {'__name__': '__main__', '__file__': str(script_path)}
    exec(code, namespace)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_with_config.py <path_to_simulation_script>")
        print("Example: python run_with_config.py src/simulations/centralized_sim.py")
        sys.exit(1)
    
    # Optional second argument for a custom config file path
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    run_simulation(sys.argv[1], config_file)
