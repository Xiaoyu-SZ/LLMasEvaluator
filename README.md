# LLMasEvaluator

The offical code for Large Language Model as Evaluator for Explainable Recommendation.

## Installation

Please do check if your torch.cuda.is_available() is True for your local machine.

Besides, to use LLMasEvaluator with vllm detailed here, you need to mannually install vllm following vllm document.

if your CUDA is 12.1
```bash
pip install vllm
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
```
if your CUDA is 11.8
```bash
# Replace `cp39` with your Python version (e.g., `cp38`, `cp39`, `cp311`).
pip install https://github.com/vllm-project/vllm/releases/download/v0.2.2/vllm-0.2.2+cu118-cp39-cp39-manylinux1_x86_64.whl
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```

Then, install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Generate Annotations via APIs

To generate annotations via APIs, please give your API key in `llm.py`

Then give your parameters in `annot.py` or `annot_single.py`, in `annot.py` we generate annotations in all aspects at a time, in `annot_single.py` we only generate annotations in one aspect at a time.

For example:
```python
MODEL_NAME = 'gpt-3.5-turbo'
CONTAIN_USER_PROFILE = False
CONTAIN_SHOT = 'None' # Type or None
PERSONALIZED = '_personalized' if CONTAIN_USER_PROFILE else ''
TEMPEARTURE = 0
```

Then run `annot.py` or `annot_single.py`

### Generate Annotations via Local LLM

Please make sure your local machine has installed vllm and xformers.

Then give your parameters in `annot_vllm.py` or `annot_vllm_single.py`, in `annot_vllm.py` we generate annotations in all aspects at a time, in `annot_vllm_single.py` we only generate annotations in one aspect at a time.

Then run `annot_vllm.py` or `annot_vllm_single.py`

### Calculate the metrics

Run `corr.py`, it will calculate the correlations between the annotations and the ground truth for files in `./output/`

The output contains Pearson correlation, Spearman correlation and Kendall correlation; all of them are in Dataset-Level, User-Level and Item-Level.

## Output Format

The output is in DataFrame format. The columns are:
`user`,`movie_id`,`movie_title`,`explanation_text`,`explanation_type`,`metric`,`user_value`,`llm_value`

The `llm_value` is the value predicted by LLM, and the others are from the data.

## Dataset Information
The data of real user labels and self-explanations is from the paper "User Perception of Recommendation Explanation: Are Your Explanations What Users Need?", see `./output/df_explanation.pkl`.

```
@article{10.1145/3565480,
author = {Lu, Hongyu and Ma, Weizhi and Wang, Yifan and Zhang, Min and Wang, Xiang and Liu, Yiqun and Chua, Tat-Seng and Ma, Shaoping},
title = {User Perception of Recommendation Explanation: Are Your Explanations What Users Need?},
year = {2023},
issue_date = {April 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {41},
number = {2},
issn = {1046-8188},
url = {https://doi.org/10.1145/3565480},
doi = {10.1145/3565480},
month = jan,
articleno = {48},
numpages = {31},
keywords = {user modeling, recommendation explanation, Recommender system}
}
```

We additionally collect third-part annotations for the explanatory texts, see `./output/third_party.csv`.

## Citation

If you find our [paper](https://arxiv.org/abs/2406.03248) or the data collected useful for your research, please cite our work.

@inproceedings{10.1145/3640457.3688075,
author = {Zhang, Xiaoyu and Li, Yishan and Wang, Jiayin and Sun, Bowen and Ma, Weizhi and Sun, Peijie and Zhang, Min},
title = {Large Language Models as Evaluators for Recommendation Explanations},
year = {2024},
isbn = {9798400705052},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3640457.3688075},
doi = {10.1145/3640457.3688075},
booktitle = {Proceedings of the 18th ACM Conference on Recommender Systems},
pages = {33–42},
numpages = {10},
keywords = {Evaluation, Explainable Recommendation, Large Language Model},
location = {Bari, Italy},
series = {RecSys '24}
}

