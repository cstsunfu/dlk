from typing import Callable
func_registry = {}
import os

models_dir = os.path.dirname(__file__)

def register(text):
    if text in func_registry:
        raise ValueError('error')

    def decorator(func):
        func_registry[text] = func
        return func
    return decorator



@register('test')
def fun(inp: str) -> str:
    """TODO: Docstring for fun.

    :inp: str: TODO
    :returns: TODO

    """
    return inp

print(models_dir)
# print(func_registry['test']('nihao'))
