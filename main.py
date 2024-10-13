from components.param_manager import ParamManager


param_manager = ParamManager(defaults={'param1': 42, 'param2': 'Hello World'})
value = param_manager.get_param('param1')
print(value)
param_manager.set_param('new_param', 3.14)

global_params = param_manager.get_all_params()
print(global_params)