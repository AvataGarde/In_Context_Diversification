import sys
import os
import json
import time
import openai
from openai import OpenAI
import jsonlines
import utils
import math
import random
import re
import backoff
from tqdm import tqdm
from itertools import combinations,permutations
import icd
sys.path.append(r'')

client = OpenAI(api_key = "")


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def generate_sentence_prompt(inputs, n, mode="commongen"):
    #concepts = concepts.replace(" ",",")
    commongen = f"Given several concepts: \"{inputs}\", write {n} coherent sentences as short as possible using background commonsense knowledge: "
    dimongen = f"Given two concepts: \"{inputs}\", generate {n} coherent sentences as short as possible using background commonsense knowledge: "
    semeval = f"Given the conterfactual statement: \"{inputs}\", write {n} short and simple commonsense-making explanations for the statement: "
    
    variables = {
        "commongen": commongen,
        "dimongen": dimongen,
        "semeval": semeval,
    }
    return variables[mode]


def init_prompt(concept_set:str, n:int, mode:str, previous_sentences:list, if_icl:bool) -> list:
    prompt = ""
    with open(f"prompt/10_shot_{mode}.txt","r", encoding='utf-8') as fr:
        # Examples
        if if_icl:
            if mode == "dimongen" :
                for line in fr.readlines():
                    example = json.loads(line)
                    concepts = " ".join(example['inputs'])
                    # ref = example['labels']
                    ref = [example['labels'][0]]
                    prompt += generate_sentence_prompt(concepts,n=len(ref), mode=mode) + "\n"+"\n".join(ref) + " \n\n"
            elif mode == "semeval":
                for line in fr.readlines():
                    inputs = line.split("#####")[0]
                    #ref = line.split("#####")[1:]
                    ref = [line.split("#####")[1]]
                    prompt += generate_sentence_prompt(inputs,n=len(ref), mode=mode) + "\n".join(ref) + " \n\n"
                
            else:
                for line in fr.readlines():
                    concepts = line.split("#####")[0]
                    ref = line.split("#####")[1].strip()
                    prompt += generate_sentence_prompt(concepts,n=1, mode=mode) + ref + " \n\n"
    
    # Your task
    prompt += generate_sentence_prompt(concept_set, n=n, mode=mode)

    # Add the previous sentences, The previous sentences are filtered by diversity metrics.
    if len(previous_sentences)>0:
        previous_sentences = "You have generated the following sentences: " + "\n".join(previous_sentences) + "\n\n" + " try to provide other reasonable sentences:"+ "\n\n"
        prompt += previous_sentences

    chat_record = [{"role": "user", "content": prompt}]
    
    return chat_record


def gpt_fewshot_sentence(chat_record:list, n:int) -> list:
            # Hidden some str of the finetuned models
            # gpt-3.5-turbo
            # ft:gpt-3.5-turbo-0613:::
            # Semeval: ft:gpt-3.5-turbo-0613:::
            # DimonGen: ft:gpt-3.5-turbo-0613:::
    max_tokens = 240 if n==1 else 25

    completion = completions_with_backoff(model="gpt-3.5-turbo", messages=chat_record, n=n, temperature=1.1,  max_tokens=max_tokens)
    #just check where the n is applied
            
    if len (completion.choices)>1:
        sentences = [c.message.content.strip() for c in completion.choices]
    else:
        content = [c.message.content.strip() for c in completion.choices][0]
        sentences = re.split(r'\n+', content)
    
    return utils.remove_numbering_from_list(sentences)



# Init generate the sentences with examples
def generate_init(concept_set:str, n:int,  num_pos:str, mode:str, previous_sentences:list, if_icl:bool):
    if num_pos =="diversified":
        init_record = init_prompt(concept_set=concept_set, n=n, mode=mode, previous_sentences=previous_sentences, if_icl=if_icl)
        sentences = gpt_fewshot_sentence(init_record, n=1)
    else:
        init_record = init_prompt(concept_set=concept_set, n=1, mode=mode, previous_sentences=previous_sentences,if_icl=if_icl)
        sentences = gpt_fewshot_sentence(init_record, n=n)
    return init_record, sentences


# Regenerate the sentences with feedback
def generate_with_feedback(concept_set:str, n:int,  num_pos:str, mode:str, max_attempt:int, allow_miss:int, previous_sentences:list) -> list:
    success_candidate = []
    init_record, sentences = generate_init(concept_set, n, num_pos, mode, previous_sentences)
    #print("Initial sentences:")
    #print(str(sentences))
    #print("########")
    for sent in sentences:
        init_sent = sent
        for i in range(max_attempt):
            #concept_feedback = utils.check_missing_concepts(concept_set, init_sent, allow_miss)
            commonsense_feedback = utils.check_commonsense_score(init_sent)
            if  commonsense_feedback.lower()=="none" :
                #if i > 0:
                    #print("Original Sentences: ",sent,"Improved Sentences: ", init_sent, "Attempts: ", str(i))
                #Address the sentences
                if len(init_sent.split(": ")) >1:
                    success_candidate.append(init_sent.split(": ")[1].strip())
                else:
                    success_candidate.append(init_sent)
                break
            else:
                init_record.append({"role": "assistant", "content": init_sent})
                intr = commonsense_feedback +" Regenerate:"
                #print(intr)
                init_record.append({"role": "user", "content": intr.replace("NONE","")})
                if i < max_attempt - 1:
                    init_sent = gpt_fewshot_sentence(init_record, 1)[0]
    return success_candidate


#Random sample n-1 concepts from the concept set to increase the diversity of the generated sentences
def sample_subset(concept_set, n):
    concept_set = concept_set.split(" ")
    s = [' '.join(list(i)) for i in combinations(concept_set, len(concept_set)-1)]
    num = min(n, len(s))
    return random.sample(s, num)
    

def write_sentence_to_file(src_file:str, tgt_file:str,n:int,  num_pos:str, mode:str, feedback:bool, max_attempt:int,if_icl:bool) -> None:
    # mode is to use different prompt (alpha or order)
    # num_pos is to decide where to ask how many sentences need to generate (in prompt or in api), would have different output.
    concept_sets = []
    
    # This is for the DimonGen dataset
    with open(src_file, 'r', encoding='utf-8') as f, open(tgt_file, "a+") as fw:
        for line in f:
            concept_set = line.strip()
            if concept_set not in concept_sets:
                concept_sets.append(concept_set)
    
    
        generate_set = concept_sets
        for concept_set in tqdm(generate_set):
            _ , sentences = generate_init(concept_set, n, num_pos, mode, [], if_icl)
            print(concept_set, len(sentences))
            candidate ={"src":concept_set, "sentences":sentences}
            fw.write(json.dumps(candidate) + '\n')
    

def evaluate_format(filename , n):
    concepts = []
    predictions = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            concepts.extend([obj['src']]*n)
            if len (obj['sentences'])>=n:
                predictions.extend(random.sample(obj['sentences'],n))
            else:
                predictions.extend([random.choice(obj['sentences']) for _ in range(n)])
            print(obj['src'], len(obj['sentences']))
            assert len(concepts) == len(predictions)
    with open(filename+".concept", "w",encoding='utf-8') as wc, open(filename+".prediction", "w", encoding='utf-8') as wp:
        for c, p in zip(concepts, predictions):
            wc.write(c+"\n")
            wp.write(p.replace("\n","")+"\n")
            
            
            
def step2(src_file: str, n: int, num_pos: str, mode: str, feedback: bool, max_attempt: int,if_icl:bool) -> None:
    with jsonlines.open(src_file) as reader, open(src_file+".v2", "a+") as writer:
        for obj in tqdm(reader, desc="Processing", unit=" obj"):
            temp_sentences = []
            # Remove the sentences that are not good.
            for sent in list(obj['sentences']):
                sent = sent.split("\n")[0].strip().split(". ")[0].strip()
                #Filter those sentences that are too short
                if len(sent.split(" ")) < 5:
                    continue
                
                #Filter those sentences that are not cover all the concepts
                # if feedback:
                #     if utils.check_missing_concepts(obj['src'], sent, 0) != "NONE":
                #         print(obj['src'], sent)
                #         continue
                # Filter the sentences that could not be generated
                # if len(obj['sentences'])>1:
                temp_sentences.append(sent)
            
            # If the number of sentences is less than n, then generate more sentences.
            if len(temp_sentences) < n:
                # Check the diversity
                # temp_sentences = icd.bottom_k_similarity(temp_sentences, 3)
                need = n-len(temp_sentences)
                previous_sentences = temp_sentences
                _ , temp = generate_init(obj['src'], need, num_pos, mode, previous_sentences, if_icl=if_icl)
                  
                sentences = temp_sentences + temp
                sentences = list(set(sentences))
                print(obj['src'],len(sentences), len(temp_sentences))
                print("The new generated sentences are:",temp)
                print("The previous sentences are:",temp_sentences)
                print("########")
            else:
                sentences = temp_sentences
            
            candidate = {"src": obj['src'], "sentences": sentences}
            writer.write(json.dumps(candidate) + '\n')



if __name__ == '__main__':
    #write_sentence_to_file("dataset/dimongen/test.concept","dimongen.test.diversified.json", n=3, num_pos="diversified", mode="dimongen", feedback=False, max_attempt=6, if_icl=True)
    #write_sentence_to_file("dataset/dimongen/test.concept","dimongen.test.default.json", n=3, num_pos="api", mode="dimongen", feedback=False, max_attempt=6, if_icl=True)
    step2("dimongen.test.diversified.json", n=3, num_pos="diversified", mode="dimongen", feedback=False, max_attempt=6, if_icl=True)
