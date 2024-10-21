import json
import os

class ParamManager:
    def __init__(self, file_path='params.json', defaults=None):
        self.file_path = file_path
        # If defaults are provided, initialize with them; otherwise, start with an empty dictionary
        self.params = {}
        # Create the params.json file if it doesn't exist
        self._ensure_file_exists()
        # Load parameters from file if it already exists
        self._load_params()
        # Update with defaults after loading existing params to avoid overwriting existing values
        if defaults:
            for key, value in defaults.items():
                self.params.setdefault(key, value)

    def _ensure_file_exists(self):
        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'w') as file:
                    json.dump(self.params, file, indent=4)
            except IOError:
                print(f"Error: Unable to create parameters file at {self.file_path}")

    def _load_params(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as file:
                    self.params.update(json.load(file))
            except (json.JSONDecodeError, IOError):
                # Handle the case where the JSON is corrupted or file cannot be read
                print(f"Error: Unable to load parameters from {self.file_path}, starting with default values.")

    def get_param(self, key, default=None):
        return self.params.get(key, default)

    def set_param(self, key, value, default=None):
        if key not in self.params and default is not None:
            self.params[key] = default
        else:
            self.params[key] = value
        self._save_params()

    def _save_params(self):
        try:
            with open(self.file_path, 'w') as file:
                json.dump(self.params, file, indent=4)
        except IOError:
            print(f"Error: Unable to save parameters to {self.file_path}")

    def get_all_params(self):
        return self.params


# Example usage of this class:
# param_manager = ParamManager()
# value = param_manager.get_param('param1', default=42)
# param_manager.set_param('new_param', 3.14, default=0)

# The default parameters would be accessible as globals once you import ParamManager and instantiate it.

# Save this file as 'param_manager.py'

# Now, in your main code file (e.g., 'main.py'), you can import ParamManager and use it as follows:

# from param_manager import ParamManager
# 
# param_manager = ParamManager()
# # Accessing parameters globally
# global_params = param_manager.get_all_params()
# 
# # You can now use `param_manager.get_param()` and `param_manager.set_param()` for easy access.