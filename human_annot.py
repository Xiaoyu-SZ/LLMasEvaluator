import pandas as pd
from prompt import *
from llm_vllm import *
import numpy as np

from bs4 import BeautifulSoup

def read_explanatation_data():
    df = pd.read_pickle("data/df_explanation.pkl")

    # 提取'user_id'和'movie_id'两列
    selected_columns = ['user_id', 'movie_id','movie_title','explanation_type','explanation','persuasiveness','transparency','satisfaction','interest_accuracy']

    result_df = df[selected_columns]
    for metric in ['persuasiveness', 'transparency', 'interest_accuracy', 'satisfaction']:
        result_df[metric].fillna(3,inplace=True)

    # 打印结果
    pd.set_option('display.max_columns', None)
    print(result_df.head(1))
    result_df.to_csv('data/df_explanation_selected.csv', index=False)
    return result_df