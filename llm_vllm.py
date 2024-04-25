
import csv
import pandas as pd
# from zhipuai import ZhipuAI
import time
import re
import datetime
from tqdm import tqdm
from vllm import SamplingParams
# from openai import OpenAI

# ZhipuAI_API = '211d1d2cc764e48163efb4e8e8659574.Zo3R44Rwt17EVIDe'
# OpenAI_API = 'sk-5YYwvUgv9ThdZdOR6960D37699B744A1896853E1B3DdA7Cf'
# # openai_client = OpenAI(api_key=OpenAI_API)
# import httpx

# openai_client = OpenAI(
#     base_url="https://svip.xty.app/v1", 
#     api_key="sk-5YYwvUgv9ThdZdOR6960D37699B744A1896853E1B3DdA7Cf",
#     http_client=httpx.Client(
#         base_url="https://svip.xty.app/v1",
#         follow_redirects=True,
#     ),
# )

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


# def call_gpt_api(prompt,model_name,max_retries=3,temperature=0.0):
#     retries = 0
#     while retries < max_retries:
#         try:
#             response = openai_client.chat.completions.create(
#                 model=model_name,  # 填写需要调用的模型名称
#                 messages = [{"role": "user", "content": prompt}],
#                 temperature=temperature,
#             )
#             time.sleep(1)
#             return response.choices[0].message.content
#         except Exception as e:
#             print(now_time() + 'Error: ' + str(e))
#             retries += 1
#             print(f"Retrying in 2 seconds...")
#             time.sleep(1)


# def call_glm_api(prompt,model_name,max_retries=3,temperature=0.0):
#     client = ZhipuAI(api_key=ZhipuAI_API) # 请填写您自己的APIKey
#     retries = 0
#     while retries < max_retries:
#         try:
#             response = client.chat.asyncCompletions.create(
#                 model=model_name,  # 填写需要调用的模型名称
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": prompt
#                     }
#                 ],
#                 temperature=temperature,
#             )
#             task_id = response.id
#             task_status = ''
#             get_cnt = 0
            
#             while task_status != 'SUCCESS' and task_status != 'FAILED' and get_cnt <= 15:
#                 result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
#                 task_status = result_response.task_status
            
#                 time.sleep(1)
#                 get_cnt += 1

#             return result_response.choices[0].message.content
#         except Exception as e:
#             print(f"Error: {str(e)}")
#             retries += 1
#             print(f"Retrying in 2 seconds...")
#             time.sleep(1)
#     return "0 0 0 0"

# def call_llm_for_evaluate(prompt,model_name,mode="single",temperature=0.0):
#     model_type = model_name[:3]
#     if model_type == 'cla':
#         model_type = 'gpt'
#     call_llm_api = eval(f'call_{model_type}_api')
#     answer = call_llm_api(prompt,model_name,temperature=temperature)
#     result = parse_result(answer,mode=mode)
#     cnt = 0
#     while (result is None) :
#         answer = call_llm_api(prompt,model_name)
#         result = parse_result(answer,mode=mode)
#         cnt = cnt + 1
#         if (cnt == 5):
#             print('Warning: repeat five time without valid form from LLM')
#             if mode=="multi":
#                 return 'Error',[0,0,0,0]
#             elif mode=="single":
#                 return 'Error',0
#             else:
#                 raise NotImplementedError('Not implemented mode' + mode)
#     return answer,result       

def local_chat_template(prompt,model_name):
    cov_style = {
            "Vicuna_v1.3_7B": ("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ",
                       "USER:",
                       " ASSISTANT:"),
            "Vicuna_v1.3_13B": ("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ",
                       "USER:",
                       " ASSISTANT:"),
            "meta-llama/Llama-2-13b-chat-hf": ("[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n",
                        "",
                        " [/INST]"),
            "chatglm": ("[Round 0]\n",
                        "问：",
                        "\n答："),
            "mistralai/Mistral-7B-Instruct-v0.2" : (
                "",
                "<s>[INST] ",
                " [/INST]"
            ),
            "Qwen/Qwen1.5-14B-Chat" : (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
                "<|im_start|>user\n",
                "<|im_end|>\n<|im_start|>assistant\n"
            )
        }
    sys_prompt = ""
    role_user = ""
    role_assistant = ""
    if model_name in cov_style:
        sys_prompt, role_user, role_assistant = cov_style[model_name]
    return sys_prompt + role_user + prompt + role_assistant

def call_local_llm_api(prompts,model,model_name,temperature=0.0):   
    batch_size = 40
    results = []
    sampling_params = SamplingParams(
    temperature=0,
    top_p=0.95,
    max_tokens=512,
    repetition_penalty=1.2
    )
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size if i + batch_size < len(prompts) else len(prompts)]
        batch_outputs = model.generate(batch_prompts, sampling_params)
        results.extend([o.outputs[0].text for o in batch_outputs])
   
    return results

def call_local_llm_for_evaluate(prompts,model,model_name,mode="single",temperature=0.0):
    prompts = [local_chat_template(prompt,model_name) for prompt in prompts]
    answers = call_local_llm_api(prompts,model,model_name,temperature=temperature)
    results = []
    for prompt,answer in zip(prompts, answers):
        print(prompt)
        results.append(parse_result(answer,mode=mode))

    # cnt = 0
    # while (result is None) :
    #     answer = call_local_llm_api(prompt,model,model_name,temperature=temperature)
    #     result = parse_result(answer,mode=mode)
    #     cnt = cnt + 1
    #     if (cnt == 5):
    #         print('Warning: repeat five time without valid form from LLM')
    #         if mode=="multi":
    #             return 'Error',[0,0,0,0]
    #         elif mode=="single":
    #             return 'Error',0
    #         else:
    #             raise NotImplementedError('Not implemented mode' + mode)
    return answers,results     



def parse_result(input_str,mode="single"):
    # input_str = input_str[input_str.find("ASSISTANT:"):]
    print(input_str)
    if mode=="multi":
        pattern = re.compile(r'(\d\s\d\s\d\s\d)')
        # pattern = re.compile(r'(Persuasiveness:\s\d\sTransparency:\s\d\sAccuracy:\s\d\sSatisfactory:\s\d\s)', re.I)
    elif mode=="single":
        pattern = re.compile(r'(\d)')
    else:
        raise NotImplementedError('Not implemented mode' + mode)
    match = pattern.search(input_str)
    

    # 检查是否匹配
    if match:
        extracted_scores = match.group(0)
        
        # scores_array = list(map(int, extracted_scores.split()))
        # print("提取的部分:", extracted_scores)
        if mode == "multi":
            scores_array =re.findall(r"\d+",extracted_scores)
            scores_array = list(map(int, scores_array))
            print("提取的分数:", scores_array)
            return scores_array
        elif mode == "single":
            scores_array = pattern.findall(input_str)
            scores_array = list(map(int, scores_array))
            # print(scores_array)
            print("提取的分数:", [scores_array[-1]])
            return scores_array[-1]
        else:
            raise NotImplementedError('Not implemented mode' + mode)
    else:
        print("未找到匹配的部分")
        return None
    # 定义正则表达式模式来匹配所需的格式
    # if input_str is None:
    #     return None
    # pattern = r'Result: (Yes|No)[\n\s]*Explanation: (.*)'
    # match = re.match(pattern, input_str)
    # if match:
    #     result = match.group(1)
    #     explain = match.group(2)
    #     return result, explain
    # else:
    #     return None



