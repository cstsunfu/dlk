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
import smart_open
from dlk.utils.logger import Logger
from io import BytesIO
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
    # some binary data(hdfs) is not seekable
    if len(args)>0 and 'rb' in args[0] or 'rb' in kwargs.get('mode', ''):
        if len(args)>0:
            mode = args[0]
        else:
            mode = kwargs['mode']
        assert mode == 'rb', f"Currently dlk is only support 'rb' for reading binary data"
        with smart_open.open(path, mode) as f:
            b_data = f.read()
        return BytesIO(b_data)
        
    return smart_open.open(path, *args, **kwargs)
