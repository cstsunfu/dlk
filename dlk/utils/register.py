# Copyright 2021 cstsunfu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, Any


class Single(object):
    def __init__(self, cls):
        self._instance = {}
        self.cls = cls
    def __call__(self, type_name, force_from_dict=False):
        if type_name not in self._instance:
            self._instance[type_name] = self.cls(type_name, force_from_dict) 
        return self._instance[type_name]

@Single
class Register(object):
    """Register"""
    def __init__(self, register_type_name, force_from_dict):
        self.registry:Dict[str, Dict[str, Any]] = {}
        self.register_type_name = register_type_name
        self.force_from_dict = force_from_dict

    def register(self, type_name: str, name: str)->Callable:
        """register the name: module to self.registry

        Args:
            name: the registed module name

        Returns: 
            the module

        """
        def decorator(module):
            if type_name not in self.registry:
                self.registry[type_name] = {}
            if name.strip() == "":
                raise ValueError(f'You must set a name for {module.__name__}')

            if name in self.registry[type_name]:
                raise ValueError(f'The {name} is already registed in {type_name}.')
            self.registry[type_name][name] = module
            if self.force_from_dict:
                return lambda config: module.from_dict(config)
            else:
                return module
        return decorator

    def __call__(self, type_name: str, name:str)->Callable:
        """you can directly call the object, the behavior is the same as object.register(name)
        """
        return self.register(type_name, name)

    def get(self, type_name: str, name: str)->Any:
        """get the module by name

        Args:
            name: the name should be the real name or name+@+sub_name, and the

        Returns: 
            registed module

        """
        sp_name = name.split('@')[0]
        if type_name not in self.registry:
            raise KeyError(f"There is not a registerd type named '{type_name}'")
        if sp_name not in self.registry[type_name]:
            raise KeyError(f"In '{type_name}' register, there is not a entry named '{sp_name}'")
        return self.registry[sp_name]

    def __getitem__(self, type_and_name: tuple)->Any:
        """wrap for object.get(name)
        """
        type_name, name = type_and_name
        return self.get(type_name, name)

config_register = Register("config", force_from_dict=True)
register = Register("module")
