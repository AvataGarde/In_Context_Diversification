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

model = "vicuna-13b-v1.5"
client = OpenAI(api_key = "EMPTY",base_url="http://localhost:8000/v1/")

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.chat.completions.create(**kwargs)

def remove_numbering_from_list(lst):
    lst = [re.sub(r'^\d+[\.)]\s*', '', sentence) for sentence in lst if len(sentence.split()) >= 4]
    return lst

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



def init_prompt(concept_set:str, n:int, mode:str, previous_sentences:list) -> list:
    prompt = "\n\n# Examples\n\n"
    with open(f"prompt/10_shot_{mode}.txt","r", encoding='utf-8') as fr:
        
        # Examples
        if mode == "dimongen":
            for line in fr.readlines():
                example = json.loads(line)
                concepts = " ".join(example['inputs'])
                ref = example['labels']
                prompt += generate_sentence_prompt(concepts,n=len(ref), mode=mode) + "\n".join(ref) + " \n\n"
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
    prompt += "# Your Task \n\n"+ generate_sentence_prompt(concept_set, n=n, mode=mode)

    # Add the previous sentences
    if len(previous_sentences)>0:
        previous_sentences = "You have generated the following explanations: " + "\n".join(previous_sentences) + "\n\n" + " try to think other explanations:"+ "\n\n"
        prompt += previous_sentences
    # print(prompt)
    chat_record = [{"role": "user", "content": prompt}]
    return chat_record

def gpt_fewshot_sentence(chat_record:list, n:int) -> list:
    max_tokens = 240 if n==1 else 25

    completion = completions_with_backoff(model="vicuna-13b-v1.5", messages=chat_record, n=n, temperature=1.1,  max_tokens=max_tokens)
    #just check where the n is applied
            
    if len (completion.choices)>1:
        sentences = [c.message.content.strip() for c in completion.choices]
    else:
        content = [c.message.content.strip() for c in completion.choices][0]
        sentences = re.split(r'\n+', content)
    print(sentences)
    return remove_numbering_from_list(sentences)


# Init generate the sentences with examples
def generate_init(concept_set:str, n:int,  num_pos:str, mode:str, previous_sentences:list):
    if num_pos =="prompt":
        init_record = init_prompt(concept_set=concept_set, n=n, mode=mode, previous_sentences=previous_sentences)
        sentences = gpt_fewshot_sentence(init_record, n=1)
    else:
        init_record = init_prompt(concept_set=concept_set, n=1, mode=mode, previous_sentences=previous_sentences)
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


def write_sentence_to_file(src_file:str, tgt_file:str,n:int,  num_pos:str, mode:str,feedback:bool, max_attempt:int) -> None:
    # mode is to use different prompt (commongen or order)
    # num_pos is to decide where to ask how many sentences need to generate (in prompt or in api), would have different output.
    concept_sets = []
    
    
    # This is for the CommonGen dataset
    with open(src_file, 'r', encoding='utf-8') as f, open(tgt_file, "a+") as fw:
        for line in f:
            concept_set = line.strip()
            if concept_set not in concept_sets:
                concept_sets.append(concept_set)
    


        generate_set = concept_sets
        for concept_set in tqdm(generate_set):
            if feedback:
                sentences = generate_with_feedback(concept_set, n, num_pos, mode, max_attempt, 0, [])
            else:
                _ , sentences = generate_init(concept_set, n, num_pos, mode, [])
            print(concept_set, len(sentences))
            candidate ={"src":concept_set, "sentences":sentences}
            fw.write(json.dumps(candidate) + '\n')

def repair_no_generation(src_file: str, n: int, num_pos: str, mode: str, feedback: bool, max_attempt: int) -> None:
    with jsonlines.open(src_file) as reader, open(src_file+".v2", "a+") as writer:
        for obj in tqdm(reader, desc="Processing", unit=" obj"):
            temp_sentences = []
            # Remove the sentences that are not good.
            for sent in obj['sentences']:
                sent = sent.split("\n")[0].strip().split(". ")[0].strip()
                #Filter those sentences that are too short
                if len(sent.split()) < 6:
                    continue
                
                if "sorry" in sent.lower() or "language model" in sent.lower():
                    continue
                #Filter those sentences that are not cover all the concepts
                # if feedback:
                #     if utils.check_missing_concepts(obj['src'], sent, 0) != "NONE":
                #         print(obj['src'], sent)
                #         continue
                # Filter the sentences that could not be generated
                if len(obj['sentences'])>1:
                    temp_sentences.append(sent)
            
            # If the number of sentences is less than n, then generate more sentences.
            if len(temp_sentences) < n:
                need = n-len(temp_sentences)
                previous_sentences = temp_sentences
                if feedback:
                    # Some concepts are wrong because of the spell error, so we allow 1 missing concepts
                    temp = generate_with_feedback(obj['src'], need, num_pos, mode, max_attempt, 0, previous_sentences)
                else:
                    _ , temp = generate_init(obj['src'], need, num_pos, mode, previous_sentences)
                  
                sentences = temp_sentences + temp
                print(obj['src'],len(sentences), len(temp_sentences))
                print("The new generated sentences are:",temp)
                print("The previous sentences are:",temp_sentences)
                print("########")
            else:
                sentences = temp_sentences
            
            candidate = {"src": obj['src'], "sentences": sentences}
            writer.write(json.dumps(candidate) + '\n')



if __name__ == '__main__':
    write_sentence_to_file("../../../dataset/commongen/test.concept","../../../commongen.test.api2.json", n=6, num_pos="api", mode="commongen", feedback=False, max_attempt=6)
    repair_no_generation("commongen.test.api.json", n=6, num_pos="api", mode="commongen", feedback=False, max_attempt=6)
    write_sentence_to_file("../../../dataset/commongen/test.concept","../../../commongen.test.prompt2.json", n=6, num_pos="prompt", mode="commongen", feedback=False, max_attempt=6)
    repair_no_generation("commongen.test.prompt.json", n=6, num_pos="prompt", mode="commongen", feedback=False, max_attempt=6)

