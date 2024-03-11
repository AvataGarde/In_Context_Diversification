import spacy
from aligner import Aligner
import re
import enchant
import transformers
import torch
aligner = Aligner()

nlp = spacy.load('en_core_web_sm')
d = enchant.Dict("en_US")

# tokenizer = transformers.AutoTokenizer.from_pretrained('liujch1998/vera', use_fast=False)
# model = transformers.T5EncoderModel.from_pretrained('liujch1998/vera')
# model.D = model.shared.embedding_dim
# linear = torch.nn.Linear(model.D, 1, dtype=model.dtype)
# linear.weight = torch.nn.Parameter(model.shared.weight[32099, :].unsqueeze(0))
# linear.bias = torch.nn.Parameter(model.shared.weight[32098, 0].unsqueeze(0))
# model.eval()

# t = model.shared.weight[32097, 0].item() # temperature for calibration


def lemmatize_sentence(sentence):
    doc = nlp(sentence)
    lemmatized_sentence = set([nlp(token.text)[0].lemma_ for token in doc])
    for token in doc:
        lemmatized_sentence.add(token.lemma_)
    lemmatized_sentence = ' '.join(list(lemmatized_sentence))
    return lemmatized_sentence

def check_missing_concepts(concept_set, sentence, allow_miss=0):
    
    missing_concepts = []
    lemmatized_sentence = lemmatize_sentence(sentence.lower())
    for concept in concept_set.split():
        us_concept = concept
        if not d.check(concept):
            us_concept = d.suggest(concept)[0]  # Convert to American English
        if concept not in lemmatized_sentence and us_concept not in lemmatized_sentence:  
            missing_concepts.append(concept)
    
    if len(missing_concepts) > 0 + allow_miss:
        return " The concepts '"+", ".join(missing_concepts) +"' are missing."
    else:
        return "NONE"

def check_commonsense_score(sentence):
    input_ids = tokenizer.batch_encode_plus([sentence], return_tensors='pt', padding='longest', truncation='longest_first', max_length=128).input_ids
    with torch.no_grad():
        output = model(input_ids)
        last_hidden_state = output.last_hidden_state
        hidden = last_hidden_state[0, -1, :]
        logit = linear(hidden).squeeze(-1)
        logit_calibrated = logit / t
        score_calibrated = logit_calibrated.sigmoid()
        print(score_calibrated.item())
        if score_calibrated.item() > 0.7:
            return "NONE"
        else:
            return f" The generated sentence is conflict with commonsense with score {str(score_calibrated.item())} from [0,1]." 
        
    
def check_wrong_order(concept_set, sentence):
    true_order, _, _, _ = aligner.align(concept_set.split(), sentence, multi=False, distance=1)
    if true_order != concept_set:
        order_feedback = f" The concept ordering in your generated sentence is '{true_order}', but you need to follow the given order '{concept_set}'. "
        return order_feedback
    else:
        return "NONE"
    


def remove_numbering_from_list(lst):
    lst = [re.sub(r'^\d+[\.)]\s*', '', sentence) for sentence in lst if len(sentence.split()) >= 4]
    return lst

if __name__ == '__main__':
    statement = 'A couple goes for a walk in the park.'
    concept_set = 'couple park walk take'
    
    s = check_commonsense_score(statement)
    print(s)
    