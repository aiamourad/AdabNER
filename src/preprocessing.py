import re
import random
import numpy as np
import torch
from collections import Counter


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_diacritics(text):
    arabic_diacritics = re.compile("""
                                ّ    | # Shadda
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatweel
                             """, re.VERBOSE)
    return re.sub(arabic_diacritics, '', text)


def normalize_arabic(text):
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('ة', 'ه', text)
    text = re.sub('ى', 'ي', text)
    return text


def add_duplicate_suffixes(token_labels):
    processed = []
    for labels in token_labels:
        if labels == ['O'] or labels == 'O':
            processed.append(['O'])
            continue
        tag_counts = Counter()
        token_result = []
        for tag in labels:
            if tag == 'O':
                token_result.append('O')
                continue
            tag_counts[tag] += 1
            count = tag_counts[tag]
            if count == 1:
                token_result.append(tag)
            else:
                token_result.append(f"{tag}_{count}")
        processed.append(token_result if token_result else ['O'])
    return processed
