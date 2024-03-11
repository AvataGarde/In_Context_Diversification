# Improving Diversity of Commonsense Generation by Large Language Models via In-Context Learning

---

## **Introduction**

This document provides the code and explanations of the 2024 ARR submission for the paper Improving Diversity of Commonsense Generation by Large Language Models via In-Context Learning

## **Requirements**

Before running the code, ensure you have all the necessary dependencies installed.
```
conda create -n icd python=3.10
conda activate icd
pip install -r requirements.txt
```

Please 
1. put the `dataset` folder from the downloaded data to the `/`.  
2. `data` folder from the downloaded data to `extrinic` folder.

Also, you need to have the tokens from OpenAI and Huggingface to run the code.

## Training & Evaluation

We have already uploaded the `results` and the generated sentences from our experiments into the folder so you could run with our provided evaluation code `eval_accuracy.py` in `eval` folder.

To run the ICD method on GPT3.5-Turbo, you need to run `Intrinic/candidate.py` to generate the **default** and **diversified** sentences and then run `Intrinic/icd.py`. 

To run our experiments on the BART, set the default --method_name moree in `main.py` then run.

To run our ICD experiments on Vicuna, we follow the advice from vicuna authors and launch a RESTful API Server.
   1. cd to `Intrinic/Vicuna` folder
   2. launch the controller `python3 -m fastchat.serve.controller`
   3. launch the model worker `python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-13b-v1.5`
   4. launch the RESTful API server `python3 -m fastchat.serve.openai_api_server --host localhost --port 8000`

To try finetune on Vicuna-13b-v1.5, please run the data preprocessing script `Intrinic/Vicuna/data/finetune_format.py` and then run the `Intrinic/Vicuna/train.sh`. The generation code for finetune is in `Intrinic/Vicuna/generate.py`

```
python main.py
```

