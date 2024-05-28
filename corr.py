import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, kendalltau, pearsonr


def calculate_correlation(data_df, metric, METRIC):
    user_values = data_df[metric]
    llm_values = data_df[METRIC]
    pearsonr_correlation = pearsonr(user_values, llm_values)[0]
    spearmanr_correlation = spearmanr(user_values, llm_values)[0]
    kendalltau_correlation = kendalltau(user_values, llm_values)[0]
    return pd.Series({'pearsonr_correlation': pearsonr_correlation, 'spearmanr_correlation': spearmanr_correlation, 'kendalltau_correlation': kendalltau_correlation})


output_dir = 'output'
file_list = os.listdir(output_dir)
file_list = ['']
for file in file_list:
    # try:
    if not file.endswith('.csv'):
        continue
    FILE_NAME = 'output/' + file
    print(FILE_NAME)
    df = pd.read_csv(FILE_NAME, sep='\t')
    df['user_value'].fillna(3, inplace=True)
    df['llm_value'].fillna(0, inplace=True)
    results = {}
    for metric in ['persuasiveness', 'transparency', 'accuracy', 'satisfactory']:
        data_df = df[df['metric'] == metric]
        user_values = data_df['user_value']
        llm_values = data_df['llm_value']
        index = len(user_values)
        print(index)
        pearson_correlation = pearsonr(user_values[:index], llm_values[:index])[0]
        spearmanr_correlation = spearmanr(user_values[:index], llm_values[:index])[0]
        kendalltau_correlation = kendalltau(user_values[:index], llm_values[:index])[0]
        # mae = np.mean(np.abs(user_values -all_three_array))
        # rmse = np.sqrt(np.mean((user_values - all_three_array)**2))

        print(f"Metrics for {metric}, Dataset-Level:")
        # print(f"    Pearson correlation coefficient: {correlation_coefficient}")
        print(f"    Scipy Pearson correlation coefficient: {pearson_correlation}")
        print(f"    Spearman correlation coefficient: {spearmanr_correlation}")
        print(f"    Kendall correlation coefficient: {kendalltau_correlation}")
        
        user_correlation = data_df.groupby(['user']).apply(calculate_correlation,'user_value','llm_value').reset_index()
        
        print(f"Metrics for {metric}, User-Level:")
        print(f"   META Pearson correlation coefficient: {user_correlation['pearsonr_correlation'].mean()}")
        print(f"   META Spearman correlation coefficient: {user_correlation['spearmanr_correlation'].mean()}")
        print(f"   META Kendall correlation coefficient: {user_correlation['kendalltau_correlation'].mean()}")
        
        pair_correlation = data_df.groupby(['user','movie_id']).apply(calculate_correlation,'user_value','llm_value').reset_index()

        print(f"Metrics for {metric}, Sample-Level:")
        print(f"   META Pearson correlation coefficient: {pair_correlation['pearsonr_correlation'].mean()}")
        print(f"   META Spearman correlation coefficient: {pair_correlation['spearmanr_correlation'].mean()}")
        print(f"   META Kendall correlation coefficient: {pair_correlation['kendalltau_correlation'].mean()}")
        
        results[metric] = {'dataset':[100*pearson_correlation,100*spearmanr_correlation,100*kendalltau_correlation],
                            'user':[100*user_correlation['pearsonr_correlation'].mean(),100*user_correlation['spearmanr_correlation'].mean(),100*user_correlation['kendalltau_correlation'].mean()],
                            'pair':[100*pair_correlation['pearsonr_correlation'].mean(),100*pair_correlation['spearmanr_correlation'].mean(),100*pair_correlation['kendalltau_correlation'].mean()]
                            }

    for i in range(3):
        output = ''
        for metric in ['persuasiveness', 'transparency', 'accuracy', 'satisfactory']:
            output = output+results[metric][i]
        print(output)
