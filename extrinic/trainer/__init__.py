import transformers
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelWithLMHead
from collections import defaultdict
import torch
import logging
import numpy as np
import os
import sys
import json
import random
import datasets

from nlgeval import compute_individual_metrics
from utils.eval_acc_div import eval_accuracy_diversity, eval_top1_acc, align_files
from tqdm import tqdm
from utils.gnn import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.datasets = self.create_dataset()


        if args.method_name in ("moe","mokge"):
            self.model_path = args.model_path + args.method_name  + "/"
        elif args.method_name == "moere":
            self.model_path = args.model_path + args.method_name + "-" + args.matching_method + "-" + args.corpora_source + "/"

        if args.do_train:
            if args.model_recover_path is not None:
                logger.info("***** Recover model: %s *****success",
                        args.model_recover_path)
                model_recover = torch.load(args.model_recover_path, map_location='cpu')

                self.model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model, state_dict=model_recover)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)

            if args.method_name in ("moe","mokge", "moere"):
                self.expert_prompt = torch.randint(low=1, high=len(self.tokenizer), size=(args.expert_num, args.prompt_len))
                    
        else:
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path+ 'checkpoint-best/')
                #moe
                if args.method_name in ("moe","mokge","moere"):
                    self.expert_prompt = torch.load(self.model_path + "expert_prompt.bin")

            except:
                raise Exception("No model found in {}".format(self.model_path+ 'checkpoint-best/'))
        
        if args.method_name == "moere":
            self.args.batch_size = int(self.args.batch_size / 2)
            
        self.model.to(self.device)

        trainer_args = Seq2SeqTrainingArguments(
            output_dir=self.model_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=int(args.batch_size/args.return_sentence_num),
            save_total_limit=3,
            num_train_epochs=self.args.training_epochs,
            predict_with_generate=True,
            fp16=False,
            load_best_model_at_end=True,
            metric_for_best_model=self.args.eval_metric,
        )


        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=trainer_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["dev"],
            data_collator=self.DataCollator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        if args.method_name == "mokge":
            self.gnn_model = GNN(self.tokenizer.vocab_size)
            gnn_model_path = args.model_path + "gnn/"
            if args.do_train or not os.path.exists(gnn_model_path):
                if not os.path.exists(gnn_model_path):
                    os.makedirs(gnn_model_path) 
                cp_train = [{k: v for k,v in i["input_ids"][1].items()} for i in self.datasets["train"]]
                cp_dev = [{k: v for k,v in i["input_ids"][1].items()} for i in self.datasets["dev"]]
                self.gnn_model = train_gnn_model(self.gnn_model, gnn_model_path + "model.ckpt", cp_train, cp_dev, args.training_gnn_epochs, args.batch_size)
            else:
                state_dict = torch.load(gnn_model_path + "model.ckpt")
                own_state = self.gnn_model.state_dict()
                for name, param in state_dict.items():
                    own_state[name].copy_(param)




    def create_dataset(self):
        data_path = self.args.data_path+self.args.dataset_name
        tokenizer = self.tokenizer

        data_dic = {}
        dataset_types = ["train", ,"dev", "test"]

        if self.args.method_name in ("top_k", "top_p", "typical", "moe"):
            for dataset_type in dataset_types:
                with open(data_path+"commongen."+dataset_type+".src.txt", "r") as inputs:
                    input_lines = inputs.readlines()
                with open(data_path+"commongen."+dataset_type+".tgt.txt", "r") as labels:
                    label_lines = labels.readlines()
                
                #jsonl
                concept_dict = align_files(data_path+"commongen."+dataset_type+".src.txt", data_path+"commongen."+dataset_type+".tgt.txt")


                examples = []
                for inputs,labels in tqdm(zip(input_lines, label_lines), total=len(input_lines)):
                    #Train
                    if dataset_type == "train":
                        #input_lines like "skier ski mountain"
                        line_inputs = tokenizer.encode(inputs.strip(), max_length=self.args.max_src_len)
                        #label_lines like "skier skis on the mountain"
                        line_labels = tokenizer.encode(labels.strip(), max_length=self.args.max_tgt_len)
                    else:
                        #dev put all labels
                        line_inputs = tokenizer.encode(inputs.strip(), max_length=self.args.max_src_len)
                        line_labels = tokenizer.encode("\t".join(concept_dict[inputs.strip()]), max_length=self.args.max_tgt_len * len(concept_dict[inputs.strip()]))
                    examples.append({"input_ids":line_inputs, "labels":line_labels})

                data_dic[dataset_type] = examples
        # 
        elif self.args.method_name == "mokge":
            cp2id, id2cp, cpnet, cp_vocab = load_cpnet(self.args.data_path)
            nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
            for dataset_type in dataset_types:
                with open(data_path+"commongen."+dataset_type+".src_alpha.txt", "r") as inputs:
                    input_lines = inputs.readlines()
                with open(data_path+"commongen."+dataset_type+".tgt.txt", "r") as labels:
                    label_lines = labels.readlines()
                #address
                concept_dict = align_files(data_path+"commongen."+dataset_type+".src.txt", data_path+"commongen."+dataset_type+".tgt.txt")

                neigh_file = 'utils/'+dataset_type+'.cp_info.json'
                if os.path.exists(neigh_file):
                    with open(neigh_file, 'r') as file:
                        cp_info = json.load(file)
                        print(f"Finish load {dataset_type} cp info")
                else:
                    cp_info = dict()
                    for k in tqdm(concept_dict.keys()):
                        inp = match_concepts(k, nlp, cp_vocab, cp2id)
                        lab = match_concepts(" ".join(concept_dict[k.strip()]), nlp, cp_vocab, cp2id)
                        nodes, edges, query, cp_labels = construct_neighbor_graph(cpnet, inp, lab)
                        cp_info[k] = {"nodes": nodes, "edges": edges, "query": query, "cp_labels": cp_labels}
                    with open(neigh_file, 'w') as file:
                        json.dump(cp_info, file)
                    print(f"Finish construct {dataset_type} cp info")

                examples = []
                for inputs,labels in tqdm(zip(input_lines, label_lines), total=len(input_lines)):
                    nodes, edges, query, cp_labels = cp_info[inputs.strip()].values()
                    #nodes = [self.tokenizer.encode(id2cp[n], add_special_tokens=False)[0] for n in nodes]
                    if dataset_type == "train":
                        line_inputs = tokenizer.encode(inputs.strip(), max_length=self.args.max_src_len)
                        line_labels = tokenizer.encode(labels.strip(), max_length=self.args.max_tgt_len)
                    else:
                        #dev
                        line_inputs = tokenizer.encode(inputs.strip(), max_length=self.args.max_src_len)
                        line_labels = tokenizer.encode("\t".join(concept_dict[inputs.strip()]), max_length=self.args.max_tgt_len * len(concept_dict[inputs.strip()]))
                    
                    concepts = [tokenizer.encode(id2cp[n], add_special_tokens=False)[0] for n in nodes]
                    examples.append({"input_ids":[line_inputs, {
                                            "concepts": concepts,
                                            "edges": edges,
                                            "queries": query,
                                            "cp_labels": cp_labels
                                        }], "labels":line_labels})
                data_dic[dataset_type] = examples
                
        elif self.args.method_name == "moere":
            for TYPE in (dataset_types):
                with open(data_path+"commongen." + TYPE + ".json", "r") as f:
                    lines = [json.loads(line) for line in f.readlines()]
                with open(self.args.retrieval_path + self.args.matching_method +"/"+f"{TYPE}.{self.args.corpora_source}.jsonl", "r") as f:
                    re_lines = [json.loads(line) for line in f.readlines()]
                    #re_lines = json.load(f)
                te_case = []
                examples = []
                for line, re in tqdm(zip(lines, re_lines), total=len(lines)):
                    re_sents = ["\t".join(sent) for sent in re]
                    if TYPE == "train":
                        line_inputs = [tokenizer.encode(line["inputs"], max_length=self.args.max_src_len * self.args.num_sent) for i in line["labels"]]
                        line_labels = [tokenizer.encode(i, max_length=self.args.max_tgt_len) for i in line["labels"]]
                    else:
                        line_inputs = [tokenizer.encode(line["inputs"])]
                        line_labels = [tokenizer.encode("\t".join(line["labels"]), max_length=self.args.max_tgt_len * self.args.return_sentence_num)]
                    # the second item in "input_ids" list is the inputs for retreival sentences embedding.
                    retrieval = [tokenizer.encode(sent, add_special_tokens=False, max_length=self.args.max_tgt_len * self.args.num_sent) for sent in re_sents]
                    examples += [{
                                    "input_ids": [
                                        inp, 
                                        retrieval
                                    ], 
                                    "labels": lab,
                                } for inp, lab in zip(line_inputs, line_labels)]
                
                data_dic[TYPE] = examples

        return data_dic

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        tokenizer = self.tokenizer
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        label_str = [label.split('\t') for label in label_str]
        
        paired_inputs = []
        paired_labels = []
        for i in range(len(pred_str)):
            paired_inputs.append(pred_str[i:i + self.args.return_sentence_num])
            paired_labels.append(label_str[i])
        
        print(paired_inputs[0], paired_labels[0])
        result = eval_top1_acc(paired_inputs, paired_labels)

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in pred_ids]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: v for k, v in result.items()}


    def DataCollator(self, features, return_ts="pt"):
        if self.args.method_name == "mokge":
            cp_features = [{k: v for k,v in i["input_ids"][1].items()} for i in features]
            features = [{"input_ids": i["input_ids"][0], "labels": i["labels"]} for i in features]
        
        if self.args.method_name == "moere":
            # retrieval the extra info
            re_features = [j for i in features for j in i["input_ids"][1]]
            features = [{"input_ids": i["input_ids"][0], "labels": i["labels"]} for i in features]


        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"])
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        
        features = self.tokenizer.pad(features, padding=True, return_tensors=return_ts)
        if labels is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        
        #kg information
        if self.args.method_name == "mokge":
            concepts, node_labels, heads, tails, edge_labels, queries = kg_collate_fn(cp_features)
            node_repr = self.gnn_model.encode(concepts, heads, tails, edge_labels)
            outputs = self.gnn_model.decode(node_repr, queries)
            concept_ids = concepts.gather(dim=-1, index=outputs.topk(k=10, dim=-1).indices)
            masks_ids = torch.ones(concept_ids.shape)
            features["input_ids"] = torch.cat([features["input_ids"], concept_ids], dim=-1)
            features["attention_mask"] = torch.cat([features["attention_mask"], masks_ids], dim=-1)



        if self.args.method_name in ("moe", "mokge", "moere"):
            
            is_train = (labels is not None and '\t' not in self.tokenizer.decode(features["labels"][0]))
            #feature moe
            if self.args.method_name == "moere":
                # padding
                re_features = self.tokenizer.pad({"input_ids": re_features}, padding=True, return_tensors=return_ts, max_length=self.args.max_tgt_len * self.args.num_sent)["input_ids"]
                features = self.construct_moe_dataset(features, is_train, re_features)
            else:
                features = self.construct_moe_dataset(features, is_train)

        return features
        

    #for moe model
    def construct_moe_dataset(self, batch_inputs,train=False, re_features=None):
        """
        construct dataset with hidden variables of MOE.
        if train, the best hidden variable will be chosen to each input with hard EM algorithm
        if not train, simple concatenate all the hidden variables to the input.
        """
        batch_size, label_len = batch_inputs['labels'].shape
        #construct prompt concatenated inputs
        #repeat to match the batch size 64*3=192,5
        mixture_ids_prompt = self.expert_prompt.repeat(batch_inputs['input_ids'].shape[0], 1)
        mixture_att_prompt = torch.full(mixture_ids_prompt.shape, 1)
        mixture_inputs = {k: self.repeat(v, self.args.expert_num) for k, v in batch_inputs.items()}
        # 10+5
        mixture_inputs['input_ids'] = torch.cat([mixture_ids_prompt, mixture_inputs['input_ids']], dim=1)
        mixture_inputs['attention_mask'] = torch.cat([mixture_att_prompt, mixture_inputs['attention_mask']], dim=1)

        if re_features is not None:
            assert len(re_features) == batch_size * self.args.expert_num
            re_att = torch.full(re_features.shape, 1)

            if self.args.matching_method == "baseline":
                re_features = self.repeat(re_features[torch.arange(0, len(mixture_inputs['input_ids']), self.args.expert_num)], self.args.expert_num)
                mixture_inputs['input_ids'] = torch.cat([mixture_inputs['input_ids'], re_features], dim=1)
                mixture_inputs['attention_mask'] = torch.cat([mixture_inputs['attention_mask'], re_att], dim=1)
            else:
                # concatenate
                mixture_inputs['input_ids'] = torch.cat([mixture_inputs['input_ids'], re_features], dim=1)
                mixture_inputs['attention_mask'] = torch.cat([mixture_inputs['attention_mask'], re_att], dim=1)
        
        if train:
            #change to eval
            self.model.eval()
            _inputs = mixture_inputs.copy()
            _inputs = {k:v.to(self.device) for k,v in _inputs.items()}
            labels = _inputs.pop("labels")
            model = self.model.to(self.device)
            outputs = model(**_inputs, use_cache=False)
            #192*23* vocab size ,23 is label len
            logits = outputs[0]
            loss_function = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id ,reduction='none')
            # logits.view(-1, logits.shape[-1]) 4416 *  vocab size
            # loss 64 * 3 * 23
            loss = loss_function(logits.view(-1, logits.shape[-1]), labels.view(-1)).reshape(batch_size,self.args.expert_num,label_len)
            pad_mask = (batch_inputs['labels'] == self.tokenizer.pad_token_id).view(batch_size,1, label_len).to(self.device)
            #
            #64*3*23->64*3-> 64 -> 64*1
            mixture_ids = loss.masked_fill(pad_mask,0).sum(dim=2).argmin(dim=1).unsqueeze(1).type(torch.int64).cpu().detach()

            batch_inputs_new = batch_inputs
            #64,1->64,5-> 64 , 1 , 5
            expanded_mixture_ids = mixture_ids.expand(batch_size, self.args.prompt_len).unsqueeze(dim=1)
            # mixture_ids_prompt: 64 * 5 -> 64 * 3 *5-> 64 * 1 *5 (gather)
            inputs_ids_prompt = torch.gather(mixture_ids_prompt.view(batch_size, self.args.expert_num, -1)
                                             , dim=1, index=expanded_mixture_ids).squeeze()
            attention_prompt = torch.full(inputs_ids_prompt.shape, 1)
            batch_inputs_new['input_ids'] = torch.cat([inputs_ids_prompt,batch_inputs_new['input_ids'],], dim=1)
            batch_inputs_new['attention_mask'] = torch.cat([attention_prompt, batch_inputs_new['attention_mask']], dim=1)

            if re_features !=None:
                expanded_re_ids = mixture_ids.expand(batch_size, re_features.shape[1]).unsqueeze(dim=1)
                input_re_prompt = torch.gather(re_features.view(batch_size, self.args.expert_num, -1), dim=1, index=expanded_re_ids).squeeze()
                re_att_prompt = torch.full(input_re_prompt.shape, 1)
                batch_inputs_new['input_ids'] = torch.cat([batch_inputs_new['input_ids'], input_re_prompt], dim=1)
                batch_inputs_new['attention_mask'] = torch.cat([batch_inputs_new['attention_mask'], re_att_prompt], dim=1)
        else:
            batch_inputs_new = mixture_inputs
        
        return batch_inputs_new

    @staticmethod
    def repeat(tensor, k):
        if isinstance(tensor,torch.Tensor):
            B,*size = tensor.size()
            expand_tensor = tensor.unsqueeze(1).expand(B,k,*size).contiguous().view(B * k, *size)
            return expand_tensor
        elif isinstance(tensor, list):
            out = []
            for x in tensor:
                for _ in range(k):
                    out.append(x.copy())
            return out
    

    def train_model(self):
        self.trainer.train()
        self.trainer.save_model(self.model_path + 'checkpoint-best/')
        if self.args.method_name in ("moe", "mokge", "moere"):
            torch.save(self.expert_prompt, self.model_path + "expert_prompt.bin")
    

    def do_generation(self, batch, model):
        model = model.to(self.device)
        input_ids = torch.as_tensor(batch["input_ids"]).to(self.device)
        masks = torch.as_tensor(batch["attention_mask"]).to(self.device)
        # Here do sample is True, but in our previous code is False
        
        
        outputs = model.generate(
                input_ids=input_ids, 
                attention_mask=masks,
                max_length=self.args.max_src_len,
                do_sample=False,
                num_beams=self.args.beam_size,
                num_return_sequences=1,
                return_dict_in_generate=True, 
                output_scores=True,
            )

        generated_ids = outputs.sequences.detach().cpu()
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logits = torch.stack(outputs.scores,0).transpose(0,1).softmax(-1).detach().cpu()


        sequence_scores = outputs.sequences_scores.detach().cpu()
        #sys.exit(0)
        return predictions, logits, sequence_scores
    
    
    def write_results(self, output_dir, predictions):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        out_pred = output_dir + '/predictions.txt'

        preds = ["\t\t".join(i) for i in predictions]

        with open(out_pred, 'w') as eval_out:
            for pred in preds:
                eval_out.write(pred + '\n')


    def predict_result(self, gt_dataset=None):
        if gt_dataset is None:
            gt_dataset = self.datasets["test"]
        
        dataloader = self.trainer.get_test_dataloader(gt_dataset)
        if self.args.method_name in ("moe"):
            sources = self.tokenizer.batch_decode([j["input_ids"] for j in gt_dataset], skip_special_tokens=True)
        else:
            sources = self.tokenizer.batch_decode([j["input_ids"][0] for j in gt_dataset], skip_special_tokens=True)
        
        self.model.eval()
        preds = []
        score_list = []  # List to store all sum logs

        for batch in tqdm(dataloader):
            temp, _, seq_scores = self.do_generation(batch, self.model)
                
            preds.extend(temp)
            score_list.extend(seq_scores)
            
        scores = [score_list[i:i+self.args.return_sentence_num] for i in range(0, len(score_list), self.args.return_sentence_num)]

        predictions = [preds[i:i+self.args.return_sentence_num] for i in range(0, len(preds), self.args.return_sentence_num)]
        
        print(len(sources),len(predictions))
        assert len(sources) == len(predictions)

        #if self.args.method_name in ("top_k", "top_p", "typical"):
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                if scores and len(scores)>0:
                    predictions[i][j] = predictions[i][j] + "\tScore: " + str(scores[i][j].item())

        print(predictions[0])
        output_dir = self.args.output_dir + self.args.method_name
        if self.args.method_name == "moere":
            output_dir = output_dir + "_" + self.args.matching_method+ "_" + self.args.corpora_source
        self.write_results(output_dir, predictions)