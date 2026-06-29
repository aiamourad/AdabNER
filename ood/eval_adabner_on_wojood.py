import sys
import os
import pandas as pd
import torch
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from src import BertForMultiLabelNER, remove_diacritics, normalize_arabic
from src.metrics import NestedNERMetrics, save_classification_report

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

class Config:
    MODEL_PATH   = 'aubmindlab/bert-base-arabertv02'
    MODEL_KEY    = 'AraBERTv2'
    STRATEGY     = 'stratified_book'
    SEED         = 42

    ADABNER_MAPPINGS = 'results/bert_experiments/label_mappings.pkl'

    WOJOOD_TEST  = 'data/wojood/test.csv'
    OUTPUT_DIR   = 'results/wojood_benchmark'
    MAX_LEN      = 512
    PREDICTION_THRESHOLD = 0.5

config = Config()
Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_wojood_test():
    print(f"  loading wojood test from {config.WOJOOD_TEST}")
    df = pd.read_csv(config.WOJOOD_TEST)
    df['token'] = df['token'].astype(str).apply(remove_diacritics).apply(normalize_arabic)
    sentences = []
    for _, group in df.groupby('global_sentence_id'):
        group = group.sort_values('word_position')
        raw_labels = []
        for tag in group['Level1_tags'].tolist():
            if pd.isna(tag) or tag == 'O':
                raw_labels.append(['O'])
            else:
                raw_labels.append(str(tag).split() or ['O'])
        sentences.append({'words': group['token'].tolist(), 'labels': raw_labels})
    print(f"    -> {len(sentences):,} sentences")
    return sentences

def benchmark():
    print("=" * 70 + "\nbenchmarking adabner on wojood (zero-shot)\n" + "=" * 70)

    with open(config.ADABNER_MAPPINGS, 'rb') as f:
        mp = pickle.load(f)
    adabner_l2i, adabner_i2l = mp['label2idx'], mp['idx2label']

    model_weights = (Path(config.OUTPUT_DIR.replace('wojood_benchmark', 'bert_experiments'))
                     / f"{config.MODEL_KEY}_{config.STRATEGY}"
                     / f"best_model_seed{config.SEED}.pt")
    if not model_weights.exists():
        raise FileNotFoundError(
            f"adabner model not found: {model_weights}\n"
            f"Run bert/train_adabner.py first, or update Config.MODEL_KEY / STRATEGY / SEED."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = BertForMultiLabelNER(config.MODEL_PATH, len(adabner_l2i)).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()
    print(f"  loaded adabner model: {model_weights}")

    wojood_test = load_wojood_test()

    woj_labels = set()
    for sent in wojood_test:
        for token_labels in sent['labels']:
            woj_labels.update(token_labels)
    woj_labels.discard('O')

    adabner_labels = set(adabner_l2i.keys()) - {'O'}
    common = adabner_labels & woj_labels
    semantic_map = {'B-WORK_OF_ART': 'B-PRODUCT', 'I-WORK_OF_ART': 'I-PRODUCT'}

    adabner_to_woj = {adabner_l2i[l]: l for l in common}
    for adabner_lbl, woj_lbl in semantic_map.items():
        if adabner_lbl in adabner_l2i and woj_lbl in woj_labels:
            adabner_to_woj[adabner_l2i[adabner_lbl]] = woj_lbl

    mapped = {l for l in adabner_labels if l in semantic_map and semantic_map[l] in woj_labels}
    unmapped = adabner_labels - common - mapped
    _cov = (len(common) + len(mapped)) / len(adabner_labels) * 100
    print(f"  label coverage: {len(common)} common + {len(mapped)} mapped / {len(adabner_labels)} adabner ({_cov:.1f}%)")

    all_true, all_pred, all_words = [], [], []
    with torch.no_grad():
        for sent in tqdm(wojood_test, desc="Evaluating on Wojood"):
            words = [str(w) for w in sent['words']]
            enc = tokenizer(
                words, is_split_into_words=True, truncation=True,
                padding='max_length', max_length=config.MAX_LEN,
                return_tensors='pt', add_special_tokens=True
            )
            logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            preds = (torch.sigmoid(logits) > config.PREDICTION_THRESHOLD).float()
            word_ids = enc.word_ids(0)

            st, sp, pw = [], [], []
            prev = None
            for ti, wi in enumerate(word_ids):
                if wi is not None and wi != prev and wi < len(words):
                    st.append(sent['labels'][wi] if wi < len(sent['labels']) else ['O'])
                    pi = torch.where(preds[0, ti] == 1)[0].cpu().numpy()
                    sp.append([adabner_to_woj[int(i)] for i in pi if int(i) in adabner_to_woj] or ['O'])
                    pw.append(words[wi])
                    prev = wi
            if st:
                all_true.append(st)
                all_pred.append(sp)
                all_words.append(pw)

    results = {}
    if all_true:
        nm = NestedNERMetrics.compute_nested_metrics(all_true, all_pred, all_words)
        results = {
            'metrics': nm,
            'micro_precision': nm['micro']['precision'],
            'micro_recall':    nm['micro']['recall'],
            'micro_f1':        nm['micro']['f1'],
            'macro_precision': nm['macro']['precision'],
            'macro_recall':    nm['macro']['recall'],
            'macro_f1':        nm['macro']['f1'],
        }
        save_classification_report(nm, Path(config.OUTPUT_DIR) / 'adabner_on_wojood_report.txt')
        print(f"\nadabner → wojood (zero-shot)")
        print(f"  micro: p={nm['micro']['precision']:.4f}  r={nm['micro']['recall']:.4f}  f1={nm['micro']['f1']:.4f}")
        print(f"  macro: p={nm['macro']['precision']:.4f}  r={nm['macro']['recall']:.4f}  f1={nm['macro']['f1']:.4f}")

    transfer = {
        'source': 'adabner',
        'target': 'Wojood',
        'model_key':  config.MODEL_KEY,
        'strategy':   config.STRATEGY,
        'seed':       config.SEED,
        'label_overlap': {
            'common_labels':      len(common),
            'semantically_mapped': len(mapped),
            'source_unmapped':    len(unmapped),
            'target_only':        len(woj_labels - adabner_labels),
            'coverage_percentage': _cov,
            'common_label_list':  sorted(common),
            'unmapped_list':      sorted(unmapped),
        },
        'results': results,
    }

    out_path = Path(config.OUTPUT_DIR) / 'adabner_on_wojood_results.json'
    with open(out_path, 'w') as f:
        json.dump(transfer, f, indent=2, default=float)

    print(f"\n results saved to: {config.OUTPUT_DIR}")
    return transfer


if __name__ == '__main__':
    benchmark()
