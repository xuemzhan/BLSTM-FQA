"""对数据进行筛选，先筛选出满足条件的数据"""


import pandas as pd
import numpy as np


def load_data(path):
    sheet = -1
    if len(path.split('.')[-1].split('/'))>1 :
        sheet = path.split('.')[-1].split('/')[-1]
        path = path[:-len(sheet)-1]
        sheet = int(sheet)
    if path.split('.')[-1] == 'xls' or path.split('.')[-1] == 'xlsx':
        if sheet!=-1:
            df = pd.read_excel(path, sheetname=sheet)
        else:
            df = pd.read_excel(path)
    elif path.split('.')[-1] == 'csv':
        df = pd.read_csv(path)
    else:
        raise Exception('输入错误')
    return df



# find_relevant_pair()