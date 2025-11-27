import json
from types import SimpleNamespace

# --- 1. Recommended Method: Using SimpleNamespace ---
# This is the cleanest and easiest way to convert a dictionary into a simple object.

def dict_to_simple_object(data_dict):
    """
    Converts a flat dictionary into an object using SimpleNamespace.
    Keys become accessible as attributes (e.g., obj.key).
    
    NOTE: For nested dictionaries, you need recursion or to use the json_to_object function below.
    """
    return SimpleNamespace(**data_dict)

# --- 2. Advanced Method: Using a Custom Class for Recursion ---
# This method allows you to handle nested dictionaries and turn them into nested objects.

class ConfigObject:
    """
    A custom class that converts a dictionary into an object and handles nested dicts
    by recursively creating new ConfigObject instances.
    """
    def __init__(self, data):
        # Ensure the input is a dictionary
        if not isinstance(data, dict):
            raise TypeError("Input must be a dictionary.")

        for key, value in data.items():
            # Replace invalid attribute characters (optional, but robust)
            attr_key = key.replace('-', '_').replace('.', '_')
            
            # Recursively convert nested dictionaries
            if isinstance(value, dict):
                setattr(self, attr_key, ConfigObject(value))
            # Convert lists of dictionaries
            elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                 setattr(self, attr_key, [ConfigObject(item) for item in value])
            else:
                setattr(self, attr_key, value)
    
    def __repr__(self):
        # A nice representation for printing
        attributes = ', '.join(f"{k}={repr(v)}" for k, v in self.__dict__.items())
        return f"ConfigObject({attributes})"