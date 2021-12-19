"""registers"""
from typing import Callable, Dict, Any

class Register(object):
    """docstring for Register"""
    def __init__(self, register_name:str):
        super(Register, self).__init__()
        self.register_name = register_name
        self.registry:Dict[str, Any] = {}

    def register(self, name: str='')->Callable:
        """TODO: Docstring for register.
        :name: str: TODO
        :returns: TODO
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

        :name:str: TODO
        :returns: TODO

        """
        return self.register(name)

    def get(self, name: str='')->Any:
        """the name could be the real name or name+@+sub_name, and the
        :name: str: TODO
        :returns: TODO

        """
        sp_name = name.split('@')[0]
        if sp_name not in self.registry:
            raise KeyError(f"In '{self.register_name}' register, there is not a entry named '{sp_name}'")
        return self.registry[sp_name]

    def __getitem__(self, name: str='')->Any:
        """wrap for object.get(name)

        :arg1: TODO
        :returns: TODO
        """
        return self.get(name)
