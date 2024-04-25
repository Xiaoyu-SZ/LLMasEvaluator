# coding=utf-8
import pandas as pd
from collections import Counter

class PDtable:
    """
    .add(data, "colname")
    .to_pandas
    """
    def __init__(self):
        self.column_datas = {}
        self.column_names = []

    def add(self, data, colname):
        if colname not in self.column_datas:
            self.column_datas[colname] = []
            self.column_names.append(colname)

        self.column_datas[colname].append(data)

    def to_pandas(self):
        return pd.DataFrame(self.column_datas)[self.column_names]


def Ratio(vlist, nan=False):
    cnt = Counter(vlist)
    if not nan:
        total = float(len([t for t in vlist if not pd.isnull(t)]))
    else:
        total = float(len(vlist))
    ans = {}
    for tk in cnt:
        if not nan and pd.isnull(tk):
            continue
        ans[tk] = cnt[tk] / total
    return ans
        
    