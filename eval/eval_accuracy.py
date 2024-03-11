import os
import sys
import numpy as np
import argparse
import spacy
import enchant
from collections import defaultdict
import diversity
from tqdm import tqdm
from bert_score import score
from joint import calculate_harmonic_mean, calculate_FBD_for_group
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.spice.spice import Spice

nlp = spacy.load("en_core_web_sm")
d = enchant.Dict("en_US")



def tokenize(dict):
    for key in dict:
        new_sentence_list = []
        for sentence in dict[key]:
            a = ''
            for token in nlp(sentence):
                a += token.text
                a += ' '
            new_sentence_list.append(a.rstrip())
        dict[key] = new_sentence_list

    return dict




def coverage_score2(preds, concept_sets):
    covs = []
    for p, cs in tqdm(zip(preds,concept_sets), total=len(preds)):
        cs = set(cs.split())
        lemmas = set()
        for token in nlp(p):
            lemmas.add(token.lemma_)
        cov = len(lemmas&cs)/len(cs)
        covs.append(cov)
    return sum(covs)/len(covs)

def eval_coverage(key_pred_file, res_file):
    preds = []
    ref = []
    concept_sets = []
    with open(res_file) as f:
        for line in f.readlines():
            preds.append(line.strip())

    with open(key_pred_file) as f:
        for line in f.readlines():
            concept_sets.append(line.strip())
    #Coverage = coverage_score(preds, concept_sets)
    Coverage = coverage_score2(preds, concept_sets)
    print(f"System level Coverage: {Coverage*100:.2f}")
    

def bertscore(preds, refs):
    #Preds is a list of sentences, refs is a list of list of sentences
    P, R, F1 = score(preds, refs, lang="en", rescale_with_baseline=True)
    print(f"System level F1 score: {F1.mean():.3f}")
    return F1.mean()


def eval_one_acc(key_src_file, key_pred_file, res_file, gts_file):
     #res is predict, gts is true label
    gts = {}
    res = {}
    with open(key_pred_file, "r") as f_key_pred, open(res_file, "r") as f_pred:
        for key_pred_line, res_line in zip( f_key_pred.readlines(), f_pred.readlines()):
            key_pred = '#'.join(sorted(key_pred_line.strip().split(' ')))
            if key_pred not in res:
                res[key_pred] = []
                res[key_pred].append(res_line.strip())
    with open(key_src_file, "r") as f_key_src, open(gts_file, "r") as f_tgt:
        for key_src_line, gts_line in zip(f_key_src.readlines(), f_tgt.readlines()):
            key_src = '#'.join(sorted(key_src_line.strip().split(' ')))
            if key_src not in gts:
                gts[key_src] = []
                gts[key_src].append(gts_line.strip())
            else:
                gts[key_src].append(gts_line.strip())
    scorer = Scorer(gts,res)
    scorer.compute_scores()
    del scorer


def eval_avg_acc(key_src_file, key_pred_file, res_file, gts_file):
     #res is predict, gts is true label
    gts = {}
    res = {}
    with open(key_pred_file, "r") as f_key_pred, open(res_file, "r") as f_pred:
        for key_pred_line, res_line in zip( f_key_pred.readlines(), f_pred.readlines()):
            key_pred = '#'.join(sorted(key_pred_line.strip().split(' ')))
            if key_pred not in res:
                res[key_pred] = []
                res[key_pred].append(res_line.strip())
            else:
                res[key_pred].append(res_line.strip()) 
    with open(key_src_file, "r") as f_key_src, open(gts_file, "r") as f_tgt:
        for key_src_line, gts_line in zip(f_key_src.readlines(), f_tgt.readlines()):
            key_src = '#'.join(sorted(key_src_line.strip().split(' ')))
            if key_src not in gts:
                gts[key_src] = []
                gts[key_src].append(gts_line.strip())
            else:
                gts[key_src].append(gts_line.strip())
    # for each predicts in res, generate a new key and same key for gts
    res_avg = {}
    gts_avg = {}
    for key in res.keys():
        for i in range(len(res[key])):
            res_avg[key+'#'+str(i)] = [res[key][i]]
            gts_avg[key+'#'+str(i)] = gts[key]
        
    scorer = Scorer(gts_avg,res_avg)
    scorer.compute_scores()
    del scorer


#--gts_file ${TRUTH_FILE} --res_file ${PRED_FILE}
def eval_topk_acc(key_src_file, key_pred_file, res_file, gts_file, self_bleu4):
    gts = {}
    res = {}
    with open(key_pred_file, "r") as f_key_pred, open(res_file, "r") as f_pred:
        for key_pred_line, res_line in zip( f_key_pred.readlines(), f_pred.readlines()):
            key_pred = '#'.join(sorted(key_pred_line.strip().split(' ')))
            if key_pred not in res:
                res[key_pred] = []
                res[key_pred].append(res_line.strip())
            else:
                res[key_pred].append(res_line.strip()) 
    with open(key_src_file, "r") as f_key_src, open(gts_file, "r") as f_tgt:
        for key_src_line, gts_line in zip(f_key_src.readlines(), f_tgt.readlines()):
            key_src = '#'.join(sorted(key_src_line.strip().split(' ')))
            if key_src not in gts:
                gts[key_src] = []
                gts[key_src].append(gts_line.strip())
            else:
                gts[key_src].append(gts_line.strip())
        

    res_best = {}
    preds_bs = []
    preds_all = []
    refs_bs = []
    for key in res.keys():
        hyp_score_list = [compute_individual_metrics(gts[key],h)['bleu_4'] for h in res[key]]
        cand = res[key][np.argmax(hyp_score_list)]
        res_best[key] = [cand]
        preds_bs.append(cand)
        preds_all.append(res[key])
        refs_bs.append(gts[key])   
    # Add BERTScore
    bs = bertscore(preds_bs, refs_bs)
    
    
    #Other metrics
    scorer = Scorer(gts,res_best)
    bleu4 = scorer.compute_scores()
    del scorer
    
    #Joint metrics
    harmony = calculate_harmonic_mean(self_bleu4, bs)
    print(f"System level Harmonic mean: {harmony:.3f}")
    
    
    fbd = calculate_FBD_for_group(preds_all, refs_bs)
    print(f"System level FBD score: {fbd:.3f}")
    


      
        
def compute_individual_metrics(refs, hyp):
    #refs list
    #hyp str
    ret_scores = {}
    scorers = [
            (Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
        ]
    for scorer, method in scorers:
        score, scores = scorer.compute_score({0: refs}, {0: [hyp.strip()]})
        for sc, _, m in zip(score, scores, method):
            ret_scores[m] = sc
    del scorers
    return ret_scores



class Scorer():
    def __init__(self,gts,res):
        self.gts = tokenize(gts)
        self.res = tokenize(res)
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]
    
    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f"%(m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f"%(method, score))
                total_scores[method] = score
        
        print('*****DONE*****')
        for key,value in total_scores.items():
            print('{}:{}'.format(key,value))
            
        return total_scores["Bleu"][3]



if __name__ == '__main__':
    # CommonGen related files
    key_src_file="../dataset/commongen/test.concept"
    tgt_file= "../dataset/commongen/test.sentence"

    need_file = ["../generation/commongen/commongen.icd"]
    
    for name in need_file:
        print(name)
        key_pred_file = f"{name}.concept"
        generated_file = f"{name}.prediction"
        
        sentences = eval_diversity.address_predict(key_pred_file,generated_file)
        self_bleu = eval_diversity.eval_self_bleu(sentences)
        print(self_bleu)
        eval_diversity.eval_self_avgcosine(sentences)
        eval_diversity.eval_self_bertscore(sentences)
        eval_diversity.eval_entropy_distinct(sentences)
        self_bleu4 = self_bleu['self_bleu_4']
        eval_topk_acc(key_src_file, key_pred_file, generated_file, tgt_file, self_bleu4
                  )
        

    
