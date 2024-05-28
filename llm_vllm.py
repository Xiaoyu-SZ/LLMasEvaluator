
import csv
import pandas as pd
import time
import re
import datetime
from tqdm import tqdm
from vllm import SamplingParams

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '

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
    return answers,results     



def parse_result(input_str,mode="single"):
    print(input_str)
    if mode=="multi":
        pattern = re.compile(r'(\d\s\d\s\d\s\d)')
    elif mode=="single":
        pattern = re.compile(r'(\d)')
    else:
        raise NotImplementedError('Not implemented mode' + mode)
    match = pattern.search(input_str)
    

    if match:
        extracted_scores = match.group(0)
        if mode == "multi":
            scores_array =re.findall(r"\d+",extracted_scores)
            scores_array = list(map(int, scores_array))
            print("Score:", scores_array)
            return scores_array
        elif mode == "single":
            scores_array = pattern.findall(input_str)
            scores_array = list(map(int, scores_array))
            # print(scores_array)
            print("Score:", [scores_array[-1]])
            return scores_array[-1]
        else:
            raise NotImplementedError('Not implemented mode' + mode)
    else:
        print("No Match")
        return None


