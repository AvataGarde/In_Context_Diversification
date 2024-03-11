import sys
import os
import json
import time
import openai
from openai import OpenAI
import jsonlines
import math
import random
import re
import backoff
from tqdm import tqdm
from itertools import combinations,permutations

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

model = ""
client = OpenAI(api_key = "EMPTY",base_url="http://localhost:8000/v1/")

def instruction(inputs):
    instruction = f"Given several concepts: \"{inputs}\", write a coherent sentence as short as possible using background commonsense knowledge: "
    return instruction

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.chat.completions.create(**kwargs)

def remove_numbering_from_list(lst):
    lst = [re.sub(r'^\d+[\.)]\s*', '', sentence) for sentence in lst if len(sentence.split()) >= 4]
    return lst

def init_prompt(concept_set:str, n:int, mode:str, previous_sentences:list) -> list:
    chat_record = [{"role": "user", "content": instruction(concept_set)}]
    return chat_record

def generate_init(concept_set:str, n:int,  num_pos:str, mode:str, previous_sentences:list):
    init_record = init_prompt(concept_set=concept_set, n=1, mode=mode, previous_sentences=previous_sentences)
    sentences = gpt_fewshot_sentence(init_record, n=n)
    return init_record, sentences

def gpt_fewshot_sentence(chat_record:list, n:int) -> list:

    max_tokens = 240 if n==1 else 25

    completion = completions_with_backoff(model=model, messages=chat_record, n=n, temperature=1.1,  max_tokens=max_tokens)
    #just check where the n is applied
            
    if len (completion.choices)>1:
        sentences = [c.message.content.strip() for c in completion.choices]
    else:
        content = [c.message.content.strip() for c in completion.choices][0]
        sentences = re.split(r'\n+', content)
    print(sentences)
    return remove_numbering_from_list(sentences)

def write_sentence_to_file(src_file:str, tgt_file:str,n:int,  num_pos:str, mode:str) -> None:
    concept_sets = []
    
    
    # This is for the CommonGen dataset
    with open(src_file, 'r', encoding='utf-8') as f, open(tgt_file, "a+") as fw:
        for line in f:
            concept_set = line.strip()
            if concept_set not in concept_sets:
                concept_sets.append(concept_set)
    
        generate_set = concept_sets
        for concept_set in tqdm(generate_set):
            _ , sentences = generate_init(concept_set, n, num_pos, mode, [])
            print(concept_set, len(sentences))
            candidate ={"src":concept_set, "sentences":sentences}
            fw.write(json.dumps(candidate) + '\n')

if __name__ == '__main__':
    #test = "pan stove cook food"
    #_,sentences = generate_init(concept_set=test, n=6,  num_pos="api", mode="commongen", previous_sentences=[])
    write_sentence_to_file("icl/dataset/commongen/test.concept","commongen.test.ft.json", n=6, num_pos="api", mode="commongen")
