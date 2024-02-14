# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict

PROTECTED = ["_name", "_base", "_search"]


def register_module_name(module_name: str):
    """get the real registerd instance module name

    Args:
        module_name: the configure module name

    Returns:
        instance module name
    """
    return module_name.split("-")[0]


class Register(object):
    """Register"""

    def __init__(self):
        self.registry: Dict[str, Dict[str, Any]] = {}

    def register(self, type_name: str, name: str) -> Callable:
        """register the name: module to self.registry

        Args:
            name: the registered module name

        Returns:
            the module

        """

        def decorator(module):
            if type_name not in self.registry:
                self.registry[type_name] = {}
            if name.strip() == "":
                raise ValueError(f"You must set a name for {module.__name__}")

            if name in self.registry[type_name]:
                raise ValueError(f"The {name} is already registered in {type_name}.")
            self.registry[type_name][name] = module
            return module

        return decorator

    def __call__(self, type_name: str, name: str) -> Callable:
        """you can directly call the object, the behavior is the same as object.register(name)"""
        return self.register(type_name, name)

    def get(self, type_name: str, name: str) -> Any:
        """get the module by name

        Args:
            name: the name should be the real name or name+@+sub_name, and the

        Returns:
            registered module

        """
        if type_name not in self.registry:
            raise KeyError(f"There is not a registerd type named '{type_name}'")
        if name not in self.registry[type_name]:
            raise KeyError(
                f"In '{type_name}' register, there is not a entry named '{name}'"
            )
        return self.registry[type_name][name]

    def __getitem__(self, type_and_name: tuple) -> Any:
        """wrap for object.get(name)"""
        type_name, name = type_and_name
        return self.get(type_name, name)


register = Register()
