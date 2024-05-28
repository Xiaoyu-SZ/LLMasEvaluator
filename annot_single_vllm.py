import pandas as pd
from prompt import *
from llm_vllm import *
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from vllm import LLM, SamplingParams
import random

MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
CONTAIN_USER_PROFILE = False
CONTAIN_SHOT = 'None'  # All or Type or None
PERSONALIZED = '_personalized' if CONTAIN_USER_PROFILE else ''
TEMPEARTURE = 0.001
FILE_NAME = f'output/{MODEL_NAME}_T{str(TEMPEARTURE)}_{CONTAIN_SHOT}{
    PERSONALIZED}_correlation_results_split_all.csv'
LOG_FILE_NAME = f'output/{MODEL_NAME}_T{str(TEMPEARTURE)}_{CONTAIN_SHOT}{
    PERSONALIZED}_user_split_log_all.txt'


model = LLM(model=MODEL_NAME, tensor_parallel_size=2)


def form_few_shot_case(data):

    case = {
        'persuasiveness':
        f'''\
    Movie:{data['movie_title']}
    Explanation:{data['explanation']}
        -Persuasiveness : {int(data['persuasiveness'])}''',

        'transparency':
        f'''\
    Movie:{data['movie_title']}
    Explanation:{data['explanation']}
        -Transparency : {int(data['transparency'])}''',

        'accuracy':
        f'''\
    Movie:{data['movie_title']}
    Explanation:{data['explanation']}
        -Accuracy : {int(data['interest_accuracy'])}''',

        'satisfactory':
        f'''\
    Movie:{data['movie_title']}
    Explanation:{data['explanation']}
        -Satisfactory : {int(data['satisfaction'])}'''
    }
    return case


def extract_plain_text(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    return soup.get_text()


def read_explanatation_data():
    df = pd.read_pickle("data/df_explanation.pkl")

    selected_columns = ['user_id', 'movie_id', 'movie_title', 'explanation_type',
                        'explanation', 'persuasiveness', 'transparency', 'satisfaction', 'interest_accuracy']

    result_df = df[selected_columns]
    for metric in ['persuasiveness', 'transparency', 'interest_accuracy', 'satisfaction']:
        result_df[metric].fillna(3, inplace=True)

    pd.set_option('display.max_columns', None)
    print(result_df.head(1))
    result_df.to_csv('data/df_explanation_selected.csv', index=False)
    return result_df


def read_user_data():
    user_dict = {}
    user_df = pd.read_csv('data/user_demo.csv')
    for index, row in user_df.iterrows():
        user_id = str(row["user_id"])
        user_profile = generate_user_profile(row)
        user_dict[user_id] = user_profile
    return user_dict


def read_data():
    result_dict = {}
    explanatation_df = read_explanatation_data()
    for index, row in explanatation_df.iterrows():
        user_id = row['user_id']
        explanation_type = row['explanation_type']
        row['explanation'] = extract_plain_text(row['explanation'])
        if user_id not in result_dict:
            result_dict[user_id] = []
        result_dict[user_id].append(row.to_dict())
    return result_dict


def evaluate_llm(result_dict, user_dict):
    llm_data = {
        'user': [],
        'movie_id': [],
        'movie_title': [],
        'explanation_text': [],
        'explanation_type': [],
        'metric': [],
        'user_value': [],
        'llm_value': [],
        'corr': [],
    }
    result_list = list(result_dict.items())

    logger = open(LOG_FILE_NAME, 'w')

    with open(FILE_NAME, 'w') as f:

        f.write("user\tmovie_id\tmovie_title\texplanation_text\texplanation_type\tmetric\tuser_value\tllm_value\tcorr\n")

        case = {}
        first_user = result_list[0][0]
        if CONTAIN_SHOT != 'None' and (not ('Personalized' in CONTAIN_SHOT)):
            random_movie = random.choice(result_list[0][1])['movie_title']
            train_interactions = [
                item for item in result_list[0][1] if item['movie_title'] == random_movie]
            for data in train_interactions:
                case[data['explanation_type']] = form_few_shot_case(data)

        for result in tqdm(result_list):

            if 'Personal' in CONTAIN_SHOT:
                random_movie = random.choice(result[1])['movie_title']
                train_interactions = [
                    item for item in result[1] if item['movie_title'] == random_movie]
                interactions = [item for item in result[1]
                                if item['movie_title'] != random_movie]
                case = {}
                for data in train_interactions:
                    case[data['explanation_type']] = form_few_shot_case(data)
            elif CONTAIN_SHOT != 'None' and (result[0] == first_user):
                random_movie = random.choice(result[1])['movie_title']
                train_interactions = [
                    item for item in result[1] if item['movie_title'] == random_movie]
                interactions = [item for item in result[1]
                                if item['movie_title'] != random_movie]
                case = {}
                for data in train_interactions:
                    case[data['explanation_type']] = form_few_shot_case(data)
            else:
                interactions = result[1]

            user = result[0]
            first_movie = result[1][0]['movie_title']
            user_values = {
                'persuasiveness': [None]*len(interactions),
                'transparency': [None]*len(interactions),
                'accuracy': [None]*len(interactions),
                'satisfactory': [None]*len(interactions)
            }
            llm_values = {
                'persuasiveness': [None]*len(interactions),
                'transparency': [None]*len(interactions),
                'accuracy': [None]*len(interactions),
                'satisfactory': [None]*len(interactions),
            }

            llm_prompts = []
            for index, data in tqdm(enumerate(interactions)):
                if (data['movie_title'] == random_movie) and ('Personalized' in CONTAIN_SHOT):
                    assert (False)
                else:
                    user_values['persuasiveness'][index] = (
                        float(data['persuasiveness']))
                    user_values['transparency'][index] = (
                        float(data['transparency']))
                    user_values['accuracy'][index] = (
                        float(data['interest_accuracy']))
                    user_values['satisfactory'][index] = (
                        float(data['satisfaction']))
                    user_profile = {
                        'contain': CONTAIN_USER_PROFILE, 'prompt': user_dict[user]}
                    for k in user_values.keys():

                        if 'Type' in CONTAIN_SHOT:
                            if data['explanation_type'] in case.keys():
                                present_case = case[data['explanation_type']]
                            else:
                                present_case = ''
                        elif 'All' in CONTAIN_SHOT:
                            present_case = '\n'.join(
                                str(value) for value in present_case.values())
                        elif CONTAIN_SHOT == 'None':
                            present_case = ''
                        cases = {'type': CONTAIN_SHOT, 'prompt': present_case}

                        llm_prompt = prompt_evaluate_in_likert_personalize_few_shot_per_aspect(
                            data, user_profile, cases, k)
                        llm_prompts.append(llm_prompt)
                        # logger.write(llm_prompt+'\n\n')
            llm_answers, llm_results = call_local_llm_for_evaluate(
                llm_prompts, model, MODEL_NAME, mode="single", temperature=float(TEMPEARTURE))
            # logger.write(llm_answer+'\n\n')
            for index, data in tqdm(enumerate(interactions)):
                for j, k in tqdm(enumerate(user_values.keys())):
                    llm_values[k][index] = (
                        float(llm_results[index*4+j]) if llm_results[index*4+j] is not None else None)

            for index, data in tqdm(enumerate(interactions)):
                for metric in ['persuasiveness', 'transparency', 'accuracy', 'satisfactory']:
                    row = [
                        user,
                        data['movie_id'],
                        data['movie_title'],
                        data['explanation'],
                        data['explanation_type'],
                        metric,
                        user_values[metric][index],
                        llm_values[metric][index],
                    ]
                    f.write('\t'.join(map(str, row)) + '\n')
                logger.flush()
                f.flush() 

    df = pd.read_csv(FILE_NAME, sep='\t')
    df['user_value'].fillna(3, inplace=True)
    for metric in ['persuasiveness', 'transparency', 'accuracy', 'satisfactory']:
        user_values = df[df['metric'] == metric]['user_value']
        llm_values = df[df['metric'] == metric]['llm_value']
        correlation_coefficient = np.corrcoef(user_values, llm_values)[0, 1]
        print(f"Correlation coefficient for {
              metric}: {correlation_coefficient}")
        logger.write(f"Correlation coefficient for {
                     metric}: {correlation_coefficient}")

    logger.close()


result_dict = read_data()
user_dict = read_user_data()
evaluate_llm(result_dict, user_dict)
