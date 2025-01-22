import yaml 
def load_meta_policy(filepath):
    """
    Load the YAML file and return a dict representing the MetaPolicy.
    Returns:
        {
          'version': int,
          'base_cost': float,
          'root': <pointer-string>,  # e.g. "0"
          'nodes': {
              "<pointer-string>": MetaPolicyNode(...),
              ...
          }
        }
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    # Basic checks
    version = data.get('version', None)
    base_cost = data.get('base_cost', 0.0)
    root_key = data.get('root', None)
    nodes_data = data.get('nodes', {})

    # Convert each node to a MetaPolicyNode object
    nodes = {}

load_meta_policy("bin/test.yaml")