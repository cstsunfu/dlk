# Copyright cstsunfu. All rights reserved.
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

__version__ = '0.0.17'


class AdditionalRegister(object):
    """Register"""
    def __init__(self, register_name:str):
        super(AdditionalRegister, self).__init__()
        self.register_name = register_name
        self.registry:Dict[str, Any] = {
            "multi_loss": {},
        }

    def register(self, group, name: str='')->Callable:
        """register the name: module to self.registry

        Args:
            group: the registed group name
            name: the registed function name

        Returns: 
            the module

        """
        if group not in self.registry:
            raise ValueError(f'Cannot recognize group {group}')
        register_group = self.registry[group]
        def decorator(module):
            if name.strip() == "":
                raise ValueError(f'You must set a name for {module.__name__}')

            if name in register_group:
                raise ValueError(f'The {group}/{name} is already registed in {self.register_name}.')
            register_group[name] = module
            self.registry[group] = register_group
            return module
        return decorator

    def __call__(self, group, name:str="")->Callable:
        """you can directly call the object, the behavior is the same as object.register(name)
        """
        return self.register(group, name)

    def get(self, group, name: str='')->Any:
        """get the module by name

        Args:
            group: the registed group name
            name: the name should be the real name or name+@+sub_name, and the

        Returns: 
            registed module

        """
        if group not in self.registry:
            raise KeyError(f"In '{self.register_name}' register, there is not a entry named '{group}/{name}'")
        return self.registry[group][name]

    def __getitem__(self, group, name: str='')->Any:
        """wrap for object.get(name)
        """
        return self.get(group, name)


additional_function = AdditionalRegister("additional_register")
