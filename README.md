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

We additionally collect third-part annotations for the explanatory texts, see `./output/third_party_annotation.csv`.

If you use the data in `./output/df_explanation.pkl`, please cite the paper:

```bibtex
@article{UserPerceptionTois2023,
  author       = {Hongyu Lu and
                  Weizhi Ma and
                  Yifan Wang and
                  Min Zhang and
                  Xiang Wang and
                  Yiqun Liu and
                  Tat{-}Seng Chua and
                  Shaoping Ma},
  title        = {User Perception of Recommendation Explanation: Are Your Explanations
                  What Users Need?},
  journal      = {{ACM} Trans. Inf. Syst.},
  volume       = {41},
  number       = {2},
  pages        = {48:1--48:31},
  year         = {2023},
  url          = {https://doi.org/10.1145/3565480},
  doi          = {10.1145/3565480},
  timestamp    = {Sat, 27 May 2023 15:23:45 +0200},
  biburl       = {https://dblp.org/rec/journals/tois/LuMWZWLCM23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
