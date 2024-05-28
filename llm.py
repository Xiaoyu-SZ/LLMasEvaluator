
import csv
import pandas as pd
from zhipuai import ZhipuAI
import time
import re
import datetime
from openai import OpenAI

ZhipuAI_API = "" # write here
OpenAI_API = "" # write here
openai_client = OpenAI(api_key=OpenAI_API)

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def call_gpt_api(prompt,model_name,max_retries=3,temperature=0.0):
    retries = 0
    while retries < max_retries:
        try:
            response = openai_client.chat.completions.create(
                model=model_name, 
                messages = [{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            time.sleep(1)
            return response.choices[0].message.content
        except Exception as e:
            print(now_time() + 'Error: ' + str(e))
            retries += 1
            print(f"Retrying in 2 seconds...")
            time.sleep(1)


def call_glm_api(prompt,model_name,max_retries=3,temperature=0.0):
    client = ZhipuAI(api_key=ZhipuAI_API) 
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.asyncCompletions.create(
                model=model_name,  
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
            )
            task_id = response.id
            task_status = ''
            get_cnt = 0
            
            while task_status != 'SUCCESS' and task_status != 'FAILED' and get_cnt <= 15:
                result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
                task_status = result_response.task_status
            
                time.sleep(1)
                get_cnt += 1

            return result_response.choices[0].message.content
        except Exception as e:
            print(f"Error: {str(e)}")
            retries += 1
            print(f"Retrying in 2 seconds...")
            time.sleep(1)
    return "0 0 0 0"

def call_llm_for_evaluate(prompt,model_name,mode="single",temperature=0.0):
    model_type = model_name[:3]
    if model_type == 'cla':
        model_type = 'gpt'
    call_llm_api = eval(f'call_{model_type}_api')
    answer = call_llm_api(prompt,model_name,temperature=temperature)
    result = parse_result(answer,mode=mode)
    cnt = 0
    while (result is None) :
        answer = call_llm_api(prompt,model_name)
        result = parse_result(answer,mode=mode)
        cnt = cnt + 1
        if (cnt == 5):
            print('Warning: repeat five time without valid form from LLM')
            if mode=="multi":
                return 'Error',[0,0,0,0]
            elif mode=="single":
                return 'Error',0
            else:
                raise NotImplementedError('Not implemented mode' + mode)
    return answer,result       



def parse_result(input_str,mode="single"):
    # print(input_str)
    if mode=="multi":
        pattern = re.compile(r'(\d\s\d\s\d\s\d)')
    elif mode=="single":
        pattern = re.compile(r'(\d)')
    else:
        raise NotImplementedError('Not implemented mode' + mode)
    match = pattern.search(input_str)

    if match:
        extracted_scores = match.group(1)
        scores_array = list(map(int, extracted_scores.split()))
        if mode == "multi":
            print("Score:", scores_array)
            return scores_array
        elif mode == "single":
            print("Score", [scores_array[-1]])
            return scores_array[-1]
        else:
            raise NotImplementedError('Not implemented mode' + mode)
    else:
        print("No Match")
        return None
