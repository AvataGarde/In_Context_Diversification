import torch
import sys
import json
import itertools
import random
import logging
#import gpt_evaluation
#import transformers
from tqdm import tqdm
import jsonlines
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from bert_score import score

from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.rouge.rouge import Rouge

transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move the model to GPU

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
                if len(obj['sentences'])==0:
                    predictions.extend([""]*n)
                else:
                    predictions.extend([random.choice(obj['sentences']) for _ in range(n)])
            print(obj['src'], len(obj['sentences']))
            assert len(concepts) == len(predictions)
    with open(filename+f".concept", "w") as wc, open(filename+f".prediction", "w") as wp:
        for c, p in zip(concepts, predictions):
            wc.write(c+"\n")
            wp.write(p.split("\n")[0].strip()+"\n")


def restore_jsonl(concept_filename, prediction_filename, output_filename):
    with open(concept_filename, 'r', encoding='utf-8') as concept_file, \
         open(prediction_filename, 'r', encoding='utf-8') as prediction_file, \
         open(output_filename, 'w', encoding='utf-8') as output_file:
        
        concept_lines = concept_file.readlines()
        prediction_lines = prediction_file.readlines()

        if len(concept_lines) != len(prediction_lines):
            raise ValueError("The number of lines in concept and prediction files must be the same.")

        current_src = None
        current_obj = None

        for concept, prediction in zip(concept_lines, prediction_lines):
            concept = concept.strip()
            prediction = prediction.strip()
            
            if current_src != concept:
                if current_obj is not None:
                    output_file.write(json.dumps(current_obj) + '\n')
                current_src = concept
                current_obj = {'src': concept, 'sentences': [prediction]}
            else:
                current_obj['sentences'].append(prediction)

        # Write the last object
        if current_obj is not None:
            output_file.write(json.dumps(current_obj) + '\n')



def bottom_k_bertscore(texts, k):
    total_sbert = []
    for i in range(len(texts)):
        hyp_list, ref_list = [], []
        hyp_list.append(texts[i])
        ref_list.append((texts[:i]+texts[i+1:]))
        assert len(ref_list) == len(hyp_list)
        _, _, self_metrics = score(hyp_list, ref_list, lang="en", rescale_with_baseline=True,nthreads=10, verbose=True)
        total_sbert.append((texts[i], self_metrics))
    #print(total_sbleu)
    # Less value is better
    sorted_texts = sorted(total_sbert, key=lambda x: x[1], reverse=False)[:k]
    return [text[0] for text in sorted_texts]


def optimized_bottom_k_bertscore(texts, k):
    # Prepare hyp_list and ref_list for batch processing
    hyp_list = []
    ref_list = []
    for i in range(len(texts)):
        hyp_list.extend([texts[i]] * (len(texts) - 1))
        ref_list.extend(texts[:i] + texts[i+1:])
    
    # Calculate BERTScore in batch
    _, _, F1 = score(hyp_list, ref_list, lang="en", rescale_with_baseline=True, nthreads=10, batch_size=256)

    # Process the F1 scores to get self-BERTScore for each sentence
    self_scores = [sum(F1[i*(len(texts)-1):(i+1)*(len(texts)-1)].tolist()) / (len(texts) - 1) for i in range(len(texts))]

    # Sort the sentences based on their self-BERTScore and get the bottom k sentences
    sorted_texts = [texts[i] for i in sorted(range(len(texts)), key=lambda x: self_scores[x])[:k]]
    return sorted_texts


def optimized_reunion(source1_file, source2_file, n, k):
    res = []
    with open(source1_file, 'r', encoding="utf8") as f1, open(source2_file, 'r',encoding="utf8") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
    for api_line, prompt_line in tqdm(zip(lines1, lines2), total=len(lines1), desc="Processing pairs"):
        api = json.loads(api_line)
        prompt = json.loads(prompt_line)
        api_sentences = api['sentences']
        prompt_sentences = prompt['sentences']
        
        # Calculate self BERTScore for all API sentences and retrieve the top k sentences.
        api_candidate = optimized_bottom_k_bertscore(api_sentences, k) if k > 0 else []
        other_api = [item for item in api_sentences if item not in api_candidate]
        
        # Extend the prompt sentences with other API sentences.
        prompt_sentences_extended = prompt_sentences + other_api
        assert len(prompt_sentences_extended) == 12

        # For each API candidate, calculate BERTScore with all prompt sentences in batch.
        combinations = list(itertools.combinations(prompt_sentences_extended, n - k))
        
        all_combinations = [list(combine) + api_candidate for combine in combinations]
        assert all([len(comb) == n for comb in all_combinations]), "One or more combinations do not have the correct size."
        # Prepare hyp_list and ref_list for batch processing
        hyp_list = []
        ref_list = []
        for comb in all_combinations:
            for idx, sent in enumerate(comb):
                hyp_list.append(sent)
                ref_list.append(comb[:idx] + comb[idx+1:])
        # Calculate BERTScore in batch
        _, _, all_f1 = score(hyp_list, ref_list, lang="en", rescale_with_baseline=True, nthreads=10, batch_size=256)
        
        # Process the F1 scores to get average BERTScore for each combination
        reshaped_f1 = [all_f1[i:i+len(api_candidate)+n-k].tolist() for i in range(0, len(all_f1), len(api_candidate)+n-k)]
        average_f1 = [sum(f1_list) / len(f1_list) for f1_list in reshaped_f1]
        
        # Get the combination with the lowest average F1 score (most diversified)
        best_combination = all_combinations[average_f1.index(min(average_f1))]
        
        res.append({"src": api['src'], "sentences": best_combination})

    with open(f"test.selfbert_{str(k)}.json", 'w', encoding="utf8") as f:
        for item in res:
            f.write(json.dumps(item) + '\n')



def compute_metrics(hyp_list, ref_list, no_overlap=False):
    refs = {idx: lines.strip().split('\t') for (idx, lines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    if not no_overlap:
        scorers = [
            (Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
        ]
        for scorer, method in scorers:
            score, scores = scorer.compute_score(refs, hyps)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    ret_scores[m] = sc
            else:
                ret_scores[method] = score
        del scorers
    return ret_scores


# Calculate the top k based on cosine similarity
def bottom_k_similarity(texts, k):    
    # Tokenize input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input data to GPU
    
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities and sum them up for each sentence
    total_similarities = []
    for i, emb_i in enumerate(embeddings):
        total_similarity = sum(F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)) for j, emb_j in enumerate(embeddings) if i != j)
        total_similarities.append((texts[i], total_similarity.item()))
    
    # Sort by similarity and take the top k
    sorted_texts = sorted(total_similarities, key=lambda x: x[1], reverse=False)[:k]
    
    return [text[0] for text in sorted_texts]


# Calculate the top diversified k sentences based on self-BLEU3
def bottom_k_diversity(texts,k):
    total_sbleu = []
    for i in range(len(texts)):
        hyp_list, ref_list = [], []
        hyp_list.append(texts[i])
        ref_list.append('\t'.join(texts[:i]+texts[i+1:]))
        self_metrics = compute_metrics(hyp_list=hyp_list, ref_list=ref_list)
        self_metrics = self_metrics["bleu_3"]
        total_sbleu.append((texts[i], self_metrics))
    #print(total_sbleu)
    # Less value is better
    sorted_texts = sorted(total_sbleu, key=lambda x: x[1], reverse=False)[:k]
    return [text[0] for text in sorted_texts]


def calculate_diversity(texts):
    hyp_list, ref_list = [], []
    for i in range(len(texts)):
        hyp_list.append(texts[i])
        ref_list.append('\t'.join(texts[:i]+texts[i+1:]))
    self_metrics = compute_metrics(hyp_list=hyp_list, ref_list=ref_list)
    return self_metrics["bleu_3"]


def calculate_cosine(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move input data to GPU
    
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities and sum them up for each sentence
    total_similarities = 0
    num_comparisons = 0
    for i, emb_i in enumerate(embeddings):
        for j, emb_j in enumerate(embeddings):
            if i != j:
                total_similarity = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0))
                total_similarities += total_similarity.item()
                num_comparisons += 1

    # Take the average similarity
    average_similarity = total_similarities / num_comparisons
    return average_similarity


# Use gpt to evaluate the sentence diversity
def combine_withgpt(source_file1, source_file2, n, k):
    with open(source_file1, 'r', encoding="utf8") as f1, open(source_file2, 'r', encoding="utf8") as f2, open(f"test.gpt_{str(k)}.json", 'a', encoding="utf8") as outfile:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for api_line, prompt_line in tqdm(zip(lines1, lines2), total=len(lines1), desc="Processing pairs"):
            api = json.loads(api_line)
            prompt = json.loads(prompt_line)
            api_sentences = api['sentences']
            prompt_sentences = prompt['sentences']

            # Retrieve the n/2 sentences from api because api has the best quality
            api_candidate = gpt_evaluation.get_top_sentences(api_sentences, k, 3) if k > 0 else []
            
            other_api = [item for item in api_sentences if item not in api_candidate]
            
            if k == 0:
                prompt_sentences = prompt_sentences + other_api
                prompt_sentences = list(set(prompt_sentences))
                sorted_res = gpt_evaluation.get_top_sentences(prompt_sentences, n - k, 2)
            else:
                print("Finish calculating the subset from api")
                print(api_candidate)
                prompt_sentences = list(set(prompt_sentences + api_candidate))
                sorted_res = gpt_evaluation.get_top_sentences(prompt_sentences, n, 3)
                
            candidate = {"src": api['src'], "sentences": sorted_res}
            print(candidate)
            # Write the candidate to the file immediately after it's generated
            outfile.write(json.dumps(candidate) + '\n')
                


# n means how many sentences are 
def reunion(source1_file, source2_file, n, k):
    res = []
    with open(source1_file, 'r', encoding="utf8") as f1, open(source2_file, 'r',encoding="utf8") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for api_line, prompt_line in tqdm(zip(lines1, lines2), total=len(lines1), desc="Processing pairs"):
            api = json.loads(api_line)
            prompt = json.loads(prompt_line)
            api_sentences = api['sentences']
            prompt_sentences = prompt['sentences']
            #Retrieve the n/2 sentences from api because api has the best quality 
            api_candidate = bottom_k_diversity(api_sentences, k) if k > 0 else []
            
            print("Finish calculating the subset from api")
            other_api = [item for item in api_sentences if item not in api_candidate]
            prompt_sentences = prompt_sentences + other_api
            total_sbleu = []
            for combine in itertools.combinations(prompt_sentences, n - k):
            #for combine in itertools.combinations(prompt_sentences, n):
                temp_prompt = list(combine)
                temp_prompt.extend(api_candidate)
                
                temp_score = calculate_diversity(temp_prompt)
                total_sbleu.append((temp_prompt, temp_score))
            #print(total_sbleu)
            sorted_selfb = sorted(total_sbleu, key=lambda x: x[1], reverse=False)[:1]
            candidate ={"src":api['src'], "sentences":sorted_selfb[0][0]}
            res.append(candidate)
    with open(f"dimongen.selfb_{str(k)}.3.json", 'w', encoding="utf8") as f:
        for item in res:
            f.write(json.dumps(item) + '\n')
            

def split_list(input_list, n, num):
    sublists = [[] for _ in range(n)]
    if len(input_list) ==0:
        return [[""]*num for _ in range(n)]
    
    for i, elem in enumerate(input_list):
        sublists[i % n].append(elem)

    for sublist in sublists:
        while len(sublist) < num:
            random_element = random.choice(input_list)
            sublist.append(random_element)
    
    return sublists
       
def retrieval_format(src_file, num_of_experts, method):
    # Two methods: random or retrieval model
    res = []
    with jsonlines.open(src_file) as reader:
        for obj in tqdm(reader, desc="Processing", unit=" obj"):
            if method in ("random"):
                sublists = split_list(obj['sentences'], num_of_experts, 3)
            elif method in ("retrieval"):
                sublists = list(set(obj['sentences']))
            res.append(sublists)
    src_file = src_file.split('/')[-1]
    with open(f"{method}/"+src_file, "w") as fw:
        for r in res:
            fw.write(json.dumps(r) + '\n')        
            
def deduplication(src_file):
    res = []
    with jsonlines.open(src_file) as reader:
        for obj in tqdm(reader, desc="Processing", unit=" obj"):
            if obj not in res:
                res.append(obj)
    with open(src_file, "w") as fw:
        for r in res:
            fw.write(json.dumps(r) + '\n')

if __name__ == '__main__':
    reunion("dimongen.test.api.3.json","dimongen.test.prompt.3.json",3,0)
    evaluate_format("dimongen.selfb_0.3.json",3)


