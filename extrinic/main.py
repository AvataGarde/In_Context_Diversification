import json
import argparse
import os
import numpy as np
import random
import torch
from transformers import set_seed
from trainer import Trainer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')

parser.add_argument('--data_path', type=str, default="data/", help="path to dataset")
parser.add_argument('--dataset_name', type=str, default="original_dataset/", help="path to dataset")
parser.add_argument('--model_path', type=str, default="checkpoints/", help="the path to saved models")
parser.add_argument('--output_dir', type=str, default="output/", help="the path to the output texts")

parser.add_argument('--model_recover_path', type=str, default=None, help="the path to load pretrained models")

parser.add_argument('--method_name', type=str, default="moere", help='list the method from: ["moe", "mokge", "moere"]')

parser.add_argument('--pretrained_model', type=str, default="facebook/bart-large", help="the name of the generation model")

parser.add_argument('--do_train', type=bool, default=False, help="")
parser.add_argument('--do_eval', type=bool, default=True, help="")
parser.add_argument('--training_epochs', type=int, default=25, help="the number of training epochs")
parser.add_argument('--eval_metric', type=str, default="top1_bleu_4", help="which metric to choose the best checkpoint (eval_loss or top1_bleu_4)")
parser.add_argument('--learning_rate', type=int, default=3e-5, help="the generated sentence number for evaluation")
parser.add_argument('--batch_size', type=int, default=64, help="training batch size")
parser.add_argument('--max_src_len', type=int, default=30, help="the maximum source length according to the dataset")
parser.add_argument('--max_tgt_len', type=int, default=30, help="the maximum target length according to the dataset")

parser.add_argument('--do_sample', type=bool, default=True, help="Sample or not")
parser.add_argument('--return_sentence_num', type=int, default=3, help="the generated sentence number for evaluation")
parser.add_argument('--beam_size', type=int, default=5, help="Beam size for searching")

#for moe methods
parser.add_argument('--expert_num', type=int, default=3, help="the number of experts (hidden variables) to use")
parser.add_argument('--prompt_len', type=int, default=5, help="the token length of hidden varibles")

# for MoKGE methods
parser.add_argument('--training_gnn_epochs', type=int, default=5, help="the number of training epochs of gnn model")

# for moe with retrieval methods
parser.add_argument('--retrieval_path', type=str, default="data/gpt_corpora/", help="the path to external retrieval sentences")
# list the method from: ["random"]
parser.add_argument('--matching_method', type=str, default="random", help="the method to matching retrieval sentence to target generation")
# source from ["default", "diversified", "icd"]
parser.add_argument('--corpora_source', type=str, default="default", help="the method to get the sources")
parser.add_argument("--num_sent", type=int, default=3)

args = parser.parse_args()

if __name__ == "__main__":
    set_seed(42)

    model = Trainer(args)

    if args.do_train:
        model.train_model()
    if args.do_eval:
        model.predict_result()