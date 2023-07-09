import json

import pandas
import pandas as pd


def dataset_to_df(file_path: str) -> pandas.DataFrame:
    res = []
    data = json.load(open(file_path, "r"))
    for i in data:
        for j in data[i]:
            temp = {"title": f"{i} -- {j['question']}", "text": j['answer']}
            res.append(temp)
    return pd.DataFrame(res)
