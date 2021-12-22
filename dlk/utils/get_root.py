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

"""
Get the dlk package root path
"""
import os

ROOT = os.path.join('/')

def get_root():
    """get the dlk root

    Returns: 
        abspath of this package

    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    def check_cur_is_root(cur_dir):
        """check cur_dir has 'dlk' or not
        """
        root_sign_list = ['dlk']
        for root_sign in root_sign_list:
            if os.path.exists(os.path.join(cur_dir, root_sign)):
                return True
        return False

    while cur_dir != ROOT:
        if check_cur_is_root(cur_dir):
            break
        cur_dir = os.path.dirname(cur_dir)

    if cur_dir == ROOT:
        raise ImportError(f'{os.path.dirname(os.path.abspath(__file__))} is not a right file to find dlk root')
    else:
        return cur_dir
