import torch
from transformers import BertModel, BertTokenizer
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['pooler_output'].cpu().numpy()

def compute_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu_preds, sigma_preds, mu_refs, sigma_refs):
    # Check if the distributions are identical, in which case return 0
    if np.allclose(mu_preds, mu_refs) and np.allclose(sigma_preds, sigma_refs):
        return 0.0
    
    mu_diff = mu_preds - mu_refs
    sigma_sqrt = sqrtm(sigma_preds.dot(sigma_refs))
    
    # Check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(sigma_sqrt):
        sigma_sqrt = sigma_sqrt.real
        
    # Compute the actual distance
    fd = np.dot(mu_diff, mu_diff) + np.trace(sigma_preds + sigma_refs - 2 * sigma_sqrt)
    return np.sqrt(fd)

def calculate_FBD_for_pair(generated_text, refs, model, tokenizer):
    features_preds = extract_features(generated_text, model, tokenizer)
    features_refs = extract_features(refs, model, tokenizer)
    
    mu_preds, sigma_preds = compute_statistics(features_preds)
    mu_refs, sigma_refs = compute_statistics(features_refs)
    
    distance = calculate_frechet_distance(mu_preds, sigma_preds, mu_refs, sigma_refs)
    return distance

def calculate_FBD_for_group(generated_texts, reference_texts):
    assert len(generated_texts) == len(reference_texts)
    
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    fbd_scores = []
    for gen_text, refs in tqdm(zip(generated_texts, reference_texts), total=len(generated_texts), desc="Computing FBD"):
        fbd = calculate_FBD_for_pair(gen_text, refs, model, tokenizer)
        fbd_scores.append(fbd)
    
    return sum(fbd_scores) / len(fbd_scores)

def calculate_harmonic_mean(diversity, quality):
    diversity = 100 * diversity
    quality = 100 * quality
    return 2 * ((100-diversity) * quality) / ((100-diversity) + quality)

