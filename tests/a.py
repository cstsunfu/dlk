#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: 
# Author: Sun Fu
# Email: cstsunfu@gmail.com
# Github: https://github.com/cstsunfu
# Description: TODO

# import sys
# from get_root import get_root

# sys.path.append(get_root())

# from models.seq_label import SeqLabelConfig

# SeqLabelConfig()
import pandas as pd

a = pd.DataFrame({"nihao":[1,2,3]})
# # print(a)
# print('niha' in a)

print(a[~a['nihao'].isna()])
