import os
import sys
import numpy as np
from collections import defaultdict

from tqdm import tqdm
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from bert_score import score

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move the model to GPU

def address_predict(concept_file,predict_file):
    references = []
    concept_dict = defaultdict(list)
    previous_set = None
    with open(concept_file, 'r', encoding="utf-8") as fr1, open(predict_file, 'r', encoding="utf-8") as fr2:
        for concept, pred in zip(fr1.readlines(), fr2.readlines()):
            concepts = concept.strip().split()
            sorted_line = ' '.join(sorted(concepts))
            concept_dict[sorted_line].append(pred.strip())
    nested_list = [v for v in concept_dict.values()]

    return nested_list


def eval_self_bertscore(sentences_groups):
    hyp_list, ref_list = [], []
    for sentences in sentences_groups:
        for i in range(len(sentences)):
            hyp_list.append(sentences[i])
            ref_list.append(sentences[:i]+sentences[i+1:])
    
    _, _, F1 = score(hyp_list, ref_list, lang="en", rescale_with_baseline=True)
    bertscore = F1.mean().item()
    print("The F1 self-bertscore is ", bertscore)
    return bertscore

def eval_self_bleu(sentences_groups):
    hyp_list, ref_list = [], []
    for sentences in sentences_groups:
        for i in range(len(sentences)):
            hyp_list.append(sentences[i])
            ref_list.append('\t'.join(sentences[:i]+sentences[i+1:]))
    
    self_metrics = compute_metrics(hyp_list=hyp_list, ref_list=ref_list)
    self_metrics = {f'self_{k}': v for k, v in self_metrics.items()}
    
    return self_metrics

# Calculate the average cosine similarity between each sentence pair in a group
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


def eval_self_avgcosine(sentences_groups):
    scores = []
    for sentences in tqdm(sentences_groups):
        temp = calculate_cosine(sentences)
        scores.append(temp)
    print("The average cosine similarity is: ", np.mean(scores))
    return np.mean(scores)
    
    
def eval_entropy_distinct(predictions):
    diversity_metrics = {}
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    for pred in predictions:
        for gg in pred:
            g = gg.rstrip('2').split()
            for n in range(4):
                for idx in range(len(g)-n):
                    ngram = ' '.join(g[idx:idx+n+1])
                    counter[n][ngram] += 1
        
    for n in range(4):
        entropy_score = 0
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            entropy_score += - (v+0.0) /total * (np.log(v+0.0) - np.log(total))
        diversity_metrics[f'entropy_{n+1}'] = entropy_score

    print("The entropy_4 is ", diversity_metrics['entropy_4'])
    
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        diversity_metrics[f'distinct_{n+1}'] = (len(counter[n].values())+0.0) / total
    print("The distinct_4 is ", diversity_metrics['distinct_4'])
    return diversity_metrics

    

def compute_metrics(hyp_list, ref_list, no_overlap=False):
    refs = {idx: lines.strip().split('\t') for (idx, lines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)

    ret_scores = {}
    if not no_overlap:
        scorers = [
            (Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
            (Rouge(), "rouge_l"),
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
from collections import defaultdict


def sorted_sentence(sen_list):
    sorted_sentences = sorted(sen_list, key=lambda x: x['score'], reverse=True)
    sorted_sentences_list = [sentence['sentence'] for sentence in sorted_sentences]
    return sorted_sentences_list


def sort_predict(concept_file,predict_file):
    concept_dict = defaultdict(list)
    previous_set = None
    res = []
    temp_lst = []
    original_sentences = []
    with open(concept_file, 'r',encoding="utf-8") as fr1, open(predict_file, 'r', encoding="utf-8") as fr2:
        for concept, pred in zip(fr1.readlines(), fr2.readlines()):
            concepts = ' '.join(sorted(concept.strip().split()))
            
            original_sentences.append(pred.split('\tScore: ')[0].strip())
            if concepts != previous_set:
                if previous_set is not None:
                    sort_sens = sorted_sentence(temp_lst)
                    #print(sort_sens)
                    res.extend(sort_sens)
                temp_lst = []
                previous_set = concepts
            temp = {}
            temp['sentence'] = pred.split('\tScore: ')[0].strip()
            temp['score'] = float(pred.split('\tScore: ')[1].strip())       
            temp_lst.append(temp)
        if previous_set is not None:
            res.extend(sorted_sentence(temp_lst))
    with open(predict_file+".addressed",'w',encoding="utf-8") as fw:
        for line in res:
            fw.write(line+"\n")
    
    with open(predict_file+".original",'w',encoding="utf-8") as fw:
        for line in original_sentences:
            fw.write(line+"\n")






if __name__ == '__main__':

    sentences = address_predict(concept_file,predict_file)

    self_bleu = eval_self_bleu(sentences)
    print(self_bleu)
