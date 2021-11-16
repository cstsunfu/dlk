"""tasks"""
import importlib
import os
from typing import Callable, Dict, Tuple, Any
from dlkit.utils.register import Register

task_config_register = Register("Task config register.")
task_register = Register("Task register.")

def import_tasks(tasks_dir, namespace):
    for file in os.listdir(tasks_dir):
        path = os.path.join(tasks_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            task_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + task_name)


# automatically import any Python files in the tasks directory
tasks_dir = os.path.dirname(__file__)
import_tasks(tasks_dir, "dlkit.tasks")
