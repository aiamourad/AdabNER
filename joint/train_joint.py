import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (set_seed, BertForMultiLabelNER, FocalLoss, MultiLabelNERDataset,
                 remove_diacritics, normalize_arabic, add_duplicate_suffixes)
from src.metrics import NestedNERMetrics, save_classification_report, compute_averaged_metrics

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

set_seed(42)

class JointTrainingConfig:
    MODEL_PATH = 'aubmindlab/bert-base-arabertv02'
    MODEL_NAME = 'AraBERTv2'

    ADABNER_TRAIN = 'data/adabner/train_stratified_book.pkl'
    ADABNER_VAL   = 'data/adabner/val_stratified_book.pkl'
    ADABNER_TEST  = 'data/adabner/test_stratified_book.pkl'

    WOJOOD_TRAIN = 'data/wojood/train.csv'
    WOJOOD_VAL   = 'data/wojood/val.csv'
    WOJOOD_TEST  = 'data/wojood/test.csv'

    BATCH_SIZE = 48
    NUM_EPOCHS = 30
    LEARNING_RATE = 6e-5
    WARMUP_STEPS = 500
    PATIENCE = 5
    MAX_LEN = 512

    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 1.0
    PREDICTION_THRESHOLD = 0.5

    MC_SEEDS = [42, 1, 123]

    OUTPUT_DIR  = 'results/joint_training'
    FIGURES_DIR = 'results/joint_training/figures'
    RESULTS_DIR = 'results/joint_training'

config = JointTrainingConfig()
for d in [config.OUTPUT_DIR, config.FIGURES_DIR, config.RESULTS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"device: cuda | gpus: {n_gpus} ({[torch.cuda.get_device_name(i) for i in range(n_gpus)]})")
else:
    print("  device: cpu")

def remap_work_of_art_to_product(data):
    mapping = {'B-WORK_OF_ART': 'B-PRODUCT', 'I-WORK_OF_ART': 'I-PRODUCT'}
    for sent in data:
        sent['labels'] = [
            [mapping.get(l, l) for l in token_labels]
            for token_labels in sent['labels']
        ]
    return data

def load_adabner_split(pkl_path):
    print(f"loading adabner from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    data = remap_work_of_art_to_product(data)
    for sent in data:
        sent['dataset'] = 'adabner'
    print(f"{len(data):,} sentences (work_of_art -> product)")
    return data

def load_wojood_split(csv_path):
    print(f"loading wojood from {csv_path}")
    df = pd.read_csv(csv_path)
    df['token'] = df['token'].astype(str).apply(remove_diacritics).apply(normalize_arabic)
    sentences = []
    for sent_id, group in df.groupby('global_sentence_id'):
        group = group.sort_values('word_position')
        words = group['token'].tolist()
        raw_labels = []
        for tag in group['Level1_tags'].values:
            if pd.isna(tag) or tag == 'O':
                raw_labels.append(['O'])
            else:
                raw_labels.append(str(tag).split() or ['O'])
        sentences.append({
            'words': words,
            'labels': add_duplicate_suffixes(raw_labels),
            'dataset': 'wojood'
        })
    print(f"{len(sentences):,} sentences")
    return sentences

def build_unified_label_space(all_datasets):
    labels = set()
    for data in all_datasets:
        for sent in data:
            for ll in sent['labels']:
                labels.update(ll)
    all_labels = sorted(labels)
    label2idx = {l: i for i, l in enumerate(all_labels)}
    idx2label = {i: l for l, i in label2idx.items()}
    print(f"unified label space: {len(all_labels)} labels")
    return label2idx, idx2label

def get_dataset_label_set(data):
    labels = set()
    for sent in data:
        for ll in sent['labels']:
            labels.update(ll)
    labels.discard('O')
    return labels

def extract_base_types(label_set):
    types = set()
    for l in label_set:
        if '-' in l:
            rest = l.split('-', 1)[1]
            if '_' in rest:
                parts = rest.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    rest = parts[0]
            types.add(rest)
    return types


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total = 0
    for b in tqdm(loader, desc="Training", leave=False):
        ids = b['input_ids'].to(device)
        mask = b['attention_mask'].to(device)
        labels = b['labels'].to(device)
        optimizer.zero_grad()
        logits = model(ids, mask)
        valid = (labels != -100)
        loss = criterion(logits, torch.where(valid, labels, torch.zeros_like(labels)))
        loss = (loss * valid.float()).sum() / valid.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader, data, criterion, device, idx2label,
             threshold=0.5, allowed_types=None):
    model.eval()
    total, all_true, all_pred, all_words = 0, [], [], []
    with torch.no_grad():
        for bi, b in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            ids = b['input_ids'].to(device)
            mask = b['attention_mask'].to(device)
            labels = b['labels'].to(device)
            logits = model(ids, mask)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            valid = (labels != -100)
            loss = criterion(logits, torch.where(valid, labels, torch.zeros_like(labels)))
            total += ((loss * valid.float()).sum() / valid.sum()).item()

            for i in range(labels.size(0)):
                st, sp = [], []
                for j in range(labels.size(1)):
                    if (labels[i, j] != -100).any():
                        st.append([idx2label[int(x)] for x in torch.where(labels[i, j] == 1)[0].cpu().numpy()] or ['O'])
                        sp.append([idx2label[int(x)] for x in torch.where(preds[i, j] == 1)[0].cpu().numpy()] or ['O'])
                if st:
                    di = bi * loader.batch_size + i
                    if di < len(data):
                        all_true.append(st)
                        all_pred.append(sp)
                        all_words.append(data[di]['words'][:len(st)])

    res = {'loss': total / len(loader)}
    if all_true:
        nm = NestedNERMetrics.compute_nested_metrics(
            all_true, all_pred, all_words, allowed_types=allowed_types
        )
        res.update({
            'metrics': nm,
            'micro_precision': nm['micro']['precision'],
            'micro_recall': nm['micro']['recall'],
            'micro_f1': nm['micro']['f1'],
            'macro_precision': nm['macro']['precision'],
            'macro_recall': nm['macro']['recall'],
            'macro_f1': nm['macro']['f1'],
        })
    return res


def run_single_seed(seed, train_data, val_data, test_adabner, test_wojood,
                    tokenizer, label2idx, idx2label, adabner_types, wojood_types, seed_dir):
    set_seed(seed)
    print(f"\n{'─' * 60}")
    print(f"seed: {seed}")
    print(f"{'─' * 60}")

    num_labels = len(label2idx)

    train_ds = MultiLabelNERDataset(train_data, tokenizer, label2idx, config.MAX_LEN)
    val_ds = MultiLabelNERDataset(val_data, tokenizer, label2idx, config.MAX_LEN)
    test_adab_ds = MultiLabelNERDataset(test_adabner, tokenizer, label2idx, config.MAX_LEN)
    test_woj_ds = MultiLabelNERDataset(test_wojood, tokenizer, label2idx, config.MAX_LEN)

    effective_bs = config.BATCH_SIZE

    train_dl = DataLoader(train_ds, batch_size=effective_bs, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=effective_bs, num_workers=4, pin_memory=True)
    test_adab_dl = DataLoader(test_adab_ds, batch_size=effective_bs, num_workers=4, pin_memory=True)
    test_woj_dl = DataLoader(test_woj_ds, batch_size=effective_bs, num_workers=4, pin_memory=True)

    model = BertForMultiLabelNER(config.MODEL_PATH, num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, config.WARMUP_STEPS, len(train_dl) * config.NUM_EPOCHS)
    criterion = FocalLoss(config.FOCAL_ALPHA, config.FOCAL_GAMMA)

    best_model_path = seed_dir / f'best_model_seed{seed}.pt'
    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}
    best_f1, patience_ctr = 0, 0
    start = time.time()

    for ep in range(config.NUM_EPOCHS):
        tl = train_epoch(model, train_dl, optimizer, scheduler, criterion, device)
        vr = evaluate(model, val_dl, val_data, criterion, device, idx2label)

        history['train_loss'].append(tl)
        history['val_loss'].append(vr['loss'])
        vf1 = vr.get('macro_f1', 0)
        history['val_macro_f1'].append(vf1)

        print(f"Seed {seed} | Epoch {ep+1}: TrLoss={tl:.4f} VaLoss={vr['loss']:.4f} "
              f"Micro-F1={vr.get('micro_f1', 0):.4f} Macro-F1={vf1:.4f}")

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), best_model_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= config.PATIENCE:
            print(f"early stop at epoch {ep+1}")
            break

    train_time = time.time() - start

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\n  evaluating on adabner test...")
    adab_res = evaluate(model, test_adab_dl, test_adabner, criterion, device,
                        idx2label, allowed_types=adabner_types)

    print("  evaluating on wojood test...")
    woj_res = evaluate(model, test_woj_dl, test_wojood, criterion, device,
                       idx2label, allowed_types=wojood_types)

    if adab_res.get('metrics'):
        save_classification_report(adab_res['metrics'],
                                   seed_dir / f'test_adabner_seed{seed}.txt',
                                   model_name=f"Joint->adabner (seed={seed})")
        nm = adab_res['metrics']
        print(f"joint -> adabner: micro f1={nm['micro']['f1']:.4f}  macro f1={nm['macro']['f1']:.4f}")

    if woj_res.get('metrics'):
        save_classification_report(woj_res['metrics'],
                                   seed_dir / f'test_wojood_seed{seed}.txt',
                                   model_name=f"Joint->Wojood (seed={seed})")
        nm = woj_res['metrics']
        print(f"joint -> wojood:  micro f1={nm['micro']['f1']:.4f}  macro f1={nm['macro']['f1']:.4f}")

    return {
        'seed': seed,
        'epochs': ep + 1,
        'time_min': train_time / 60,
        'best_val_f1': best_f1,
        'adabner_test': adab_res,
        'wojood_test': woj_res,
        'history': history
    }


def aggregate_seed_results(all_seed_results):
    agg = {}
    for test_key in ['adabner_test', 'wojood_test']:
        agg[test_key] = {}
        for mt in ['micro', 'macro']:
            f1s = [r[test_key].get(f'{mt}_f1', 0) for r in all_seed_results]
            precs = [r[test_key].get(f'{mt}_precision', 0) for r in all_seed_results]
            recs = [r[test_key].get(f'{mt}_recall', 0) for r in all_seed_results]
            agg[test_key][mt] = {
                'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
                'precision_mean': float(np.mean(precs)), 'precision_std': float(np.std(precs)),
                'recall_mean': float(np.mean(recs)), 'recall_std': float(np.std(recs)),
            }
    return agg

def create_comparison_plot(agg, results_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    metrics_names = ['Micro F1', 'Macro F1']
    adab_vals = [agg['adabner_test'][mt]['f1_mean'] for mt in ['micro', 'macro']]
    adab_stds = [agg['adabner_test'][mt]['f1_std'] for mt in ['micro', 'macro']]
    woj_vals = [agg['wojood_test'][mt]['f1_mean'] for mt in ['micro', 'macro']]
    woj_stds = [agg['wojood_test'][mt]['f1_std'] for mt in ['micro', 'macro']]

    x = np.arange(len(metrics_names))
    w = 0.35
    ax.bar(x - w / 2, adab_vals, w, yerr=adab_stds, label='Joint -> adabner', color='steelblue', alpha=0.85, capsize=4)
    ax.bar(x + w / 2, woj_vals, w, yerr=woj_stds, label='Joint -> Wojood', color='coral', alpha=0.85, capsize=4)
    ax.set_ylabel('F1 Score')
    ax.set_title('Joint Training: Per-Dataset Test Performance (Averaged over MC Seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for i, (av, wv) in enumerate(zip(adab_vals, woj_vals)):
        ax.text(i - w / 2, av + adab_stds[i] + 0.02, f'{av:.3f}', ha='center', fontsize=9)
        ax.text(i + w / 2, wv + woj_stds[i] + 0.02, f'{wv:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(Path(results_dir) / 'joint_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 70)
    print("joint training: adabner + wojood -> test on both")
    print(f"monte carlo seeds: {config.MC_SEEDS}")
    print("=" * 70)

    print("\n loading datasets...")
    adab_train = load_adabner_split(config.ADABNER_TRAIN)
    adab_val   = load_adabner_split(config.ADABNER_VAL)
    adab_test  = load_adabner_split(config.ADABNER_TEST)
    woj_train  = load_wojood_split(config.WOJOOD_TRAIN)
    woj_val    = load_wojood_split(config.WOJOOD_VAL)
    woj_test   = load_wojood_split(config.WOJOOD_TEST)

    print("\n building unified label space...")
    label2idx, idx2label = build_unified_label_space([
        adab_train, adab_val, adab_test, woj_train, woj_val, woj_test
    ])

    adabner_labels = get_dataset_label_set(adab_train + adab_val + adab_test)
    wojood_labels = get_dataset_label_set(woj_train + woj_val + woj_test)
    adabner_types = extract_base_types(adabner_labels)
    wojood_types = extract_base_types(wojood_labels)
    common_types = adabner_types & wojood_types

    print(f"adabner: {len(adabner_types)} types  |  wojood: {len(wojood_types)} types  |  common: {len(common_types)}")

    label_info = {
        'unified_labels': len(label2idx),
        'adabner_types': sorted(adabner_types),
        'wojood_types': sorted(wojood_types),
        'common_types': sorted(common_types),
        'adabner_only': sorted(adabner_types - wojood_types),
        'wojood_only': sorted(wojood_types - adabner_types),
    }
    with open(Path(config.RESULTS_DIR) / 'label_space_info.json', 'w') as f:
        json.dump(label_info, f, indent=2)
    with open(Path(config.RESULTS_DIR) / 'unified_label_mappings.pkl', 'wb') as f:
        pickle.dump({'label2idx': label2idx, 'idx2label': idx2label}, f)

    train_combined = adab_train + woj_train
    val_combined = adab_val + woj_val

    print(f"\n combined training: {len(train_combined):,} (adabner: {len(adab_train):,}, wojood: {len(woj_train):,})")
    print(f"  combined validation: {len(val_combined):,}")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)

    print(f"\n running {len(config.MC_SEEDS)} seed(s)...")
    seed_dir = Path(config.RESULTS_DIR)
    all_seed_results = []

    for seed in config.MC_SEEDS:
        result = run_single_seed(
            seed, train_combined, val_combined, adab_test, woj_test,
            tokenizer, label2idx, idx2label, adabner_types, wojood_types, seed_dir
        )
        all_seed_results.append(result)

        with open(seed_dir / f'seed_{seed}_results.json', 'w') as f:
            save_res = {k: v for k, v in result.items() if k != 'history'}
            for tk in ['adabner_test', 'wojood_test']:
                if 'metrics' in save_res[tk]:
                    del save_res[tk]['metrics']
            json.dump(save_res, f, indent=2, default=float)

    print(f"\n aggregating across {len(config.MC_SEEDS)} seeds...")
    agg = aggregate_seed_results(all_seed_results)
    with open(Path(config.RESULTS_DIR) / 'aggregated_results.json', 'w') as f:
        json.dump(agg, f, indent=2, default=float)

    create_comparison_plot(agg, config.FIGURES_DIR)

    print("\n" + "=" * 90)
    print("joint training (adabner + wojood)")
    print("=" * 90)
    print(f"\n{'Test Set':<18} {'Metric':<12} {'F1 (mean ± std)':>22} "
          f"{'Precision':>22} {'Recall':>22}")
    print("-" * 90)
    for test_key, test_name in [('adabner_test', 'adabner'), ('wojood_test', 'Wojood')]:
        for mt in ['micro', 'macro']:
            m = agg[test_key][mt]
            print(f"{test_name:<18} {mt.capitalize():<12} "
                  f"{m['f1_mean']:.4f} ± {m['f1_std']:.4f}    "
                  f"{m['precision_mean']:.4f} ± {m['precision_std']:.4f}    "
                  f"{m['recall_mean']:.4f} ± {m['recall_std']:.4f}")
        print("-" * 90)

    print(f"\nresults saved to: {config.RESULTS_DIR}")
    print("=" * 90)

if __name__ == "__main__":
    main()
