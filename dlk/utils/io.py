import smart_open
from dlk.utils.logger import Logger
import os
logger = Logger.get_logger()


def open(path: str, *args, **kwargs):
    if len(args)>0 and 'w' in args[0] or 'w' in kwargs.get('mode', ''):
        subdir = os.path.dirname(path)
        if path.startswith("hdfs://"):
            pass
        else:
            if ':/' in subdir:
                logger.error(f"Currently dlk is only support `hdfs` and `local` file.")
                raise PermissionError
            if not os.path.exists(subdir):
                try:
                    os.makedirs(subdir, exist_ok=True)
                except Exception as e:
                    logger.error(f"Currently dlk is only support `hdfs` and `local` file.")
                    raise e
    return smart_open.open(path, *args, **kwargs)
