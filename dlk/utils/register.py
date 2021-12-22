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

class Register(object):
    """Register"""
    def __init__(self, register_name:str):
        super(Register, self).__init__()
        self.register_name = register_name
        self.registry:Dict[str, Any] = {}

    def register(self, name: str='')->Callable:
        """register the name: module to self.registry

        Args:
            name: the registed module name

        Returns: 
            the module

        """
        def decorator(module):
            if name.strip() == "":
                raise ValueError(f'You must set a name for {module.__name__}')

            if name in self.registry:
                raise ValueError(f'The {name} is already registed in {self.register_name}.')
            self.registry[name] = module
            return module
        return decorator

    def __call__(self, name:str="")->Callable:
        """you can directly call the object, the behavior is the same as object.register(name)
        """
        return self.register(name)

    def get(self, name: str='')->Any:
        """get the module by name

        Args:
            name: the name should be the real name or name+@+sub_name, and the

        Returns: 
            registed module

        """
        sp_name = name.split('@')[0]
        if sp_name not in self.registry:
            raise KeyError(f"In '{self.register_name}' register, there is not a entry named '{sp_name}'")
        return self.registry[sp_name]

    def __getitem__(self, name: str='')->Any:
        """wrap for object.get(name)
        """
        return self.get(name)
