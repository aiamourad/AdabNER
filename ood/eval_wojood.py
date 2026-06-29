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
from collections import Counter
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (set_seed, BertForMultiLabelNER, FocalLoss, MultiLabelNERDataset,
                 NestedNERMetrics, save_classification_report,
                 remove_diacritics, normalize_arabic, add_duplicate_suffixes)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

set_seed(42)

class WojoodBenchmarkConfig:
    MODEL_PATH = 'aubmindlab/bert-base-arabertv02'
    MODEL_NAME = 'AraBERTv2'

    WOJOOD_TRAIN = 'data/wojood/train.csv'
    WOJOOD_VAL = 'data/wojood/val.csv'
    WOJOOD_TEST = 'data/wojood/test.csv'

    ADABNER_TEST_PATH = 'data/adabner/test_stratified_book.pkl'

    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 6e-5
    WARMUP_STEPS = 500
    PATIENCE = 5
    MAX_LEN = 512

    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 1.0
    PREDICTION_THRESHOLD = 0.5

    OUTPUT_DIR  = 'results/wojood_benchmark'
    FIGURES_DIR = 'results/wojood_benchmark/figures'
    RESULTS_DIR = 'results/wojood_benchmark'


config = WojoodBenchmarkConfig()
for d in [config.OUTPUT_DIR, config.FIGURES_DIR, config.RESULTS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

def load_wojood_data(csv_path):
    print(f"\nloading wojood data from {csv_path}")
    df = pd.read_csv(csv_path)
    df['token'] = df['token'].astype(str).apply(remove_diacritics).apply(normalize_arabic)
    sentences, nested_count, same_type_count, max_dup = [], 0, 0, 1

    for sent_id, group in df.groupby('global_sentence_id'):
        group = group.sort_values('word_position')
        words = group['token'].tolist()
        raw_labels = []
        for tag in group['Level1_tags'].values:
            if pd.isna(tag) or tag == 'O':
                raw_labels.append(['O'])
            else:
                raw_labels.append(str(tag).split() or ['O'])

        if any(len(l) > 1 for l in raw_labels):
            nested_count += 1
        for labels in raw_labels:
            if len(labels) != len(set(labels)):
                same_type_count += 1
                max_dup = max(max_dup, max(Counter(labels).values()))
                break

        sentences.append({'sentence_id': sent_id, 'words': words, 'labels': add_duplicate_suffixes(raw_labels)})

    print(f"  {len(sentences):,} sentences | nested: {nested_count} | same-type: {same_type_count} | max dup: {max_dup}")
    return sentences

def create_label_mappings(sentences):
    labels = set()
    for s in sentences:
        for ll in s['labels']:
            labels.update(ll)
    all_labels = sorted(labels)
    label2idx = {l: i for i, l in enumerate(all_labels)}
    idx2label = {i: l for l, i in label2idx.items()}
    suffix_labels = [l for l in all_labels if any(l.endswith(f'_{i}') for i in range(2, 20))]
    print(f"  {len(all_labels)} labels ({len(suffix_labels)} suffixed)")
    return label2idx, idx2label

def load_adabner_test_data():
    print(f"\nloading adabner test data from {config.ADABNER_TEST_PATH}")
    with open(config.ADABNER_TEST_PATH, 'rb') as f:
        data = pickle.load(f)
    print(f"  loaded {len(data)} sentences")
    return data


def print_detailed_metrics(metrics, title="Evaluation Results"):
    print(f"\n{title}")
    print(f"  micro: p={metrics['micro']['precision']:.4f}  r={metrics['micro']['recall']:.4f}  f1={metrics['micro']['f1']:.4f}")
    print(f"  macro: p={metrics['macro']['precision']:.4f}  r={metrics['macro']['recall']:.4f}  f1={metrics['macro']['f1']:.4f}")

    print(f"{'=' * 70}")

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total = 0
    for b in tqdm(loader, desc="Training", leave=False):
        ids, mask, labels = b['input_ids'].to(device), b['attention_mask'].to(device), b['labels'].to(device)
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


def evaluate(model, loader, data, criterion, device, idx2label, threshold=0.5):
    model.eval()
    total, all_true, all_pred, all_words = 0, [], [], []
    with torch.no_grad():
        for bi, b in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            ids, mask, labels = b['input_ids'].to(device), b['attention_mask'].to(device), b['labels'].to(device)
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
        nm = NestedNERMetrics.compute_nested_metrics(all_true, all_pred, all_words)
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


def train_on_wojood():
    print("\n" + "=" * 70 + "\nTRAINING ON WOJOOD\n" + "=" * 70)
    mp = Path(config.RESULTS_DIR) / 'wojood_best_model.pt'
    rp = Path(config.RESULTS_DIR) / 'wojood_training_results.json'
    lp = Path(config.RESULTS_DIR) / 'wojood_label_mappings.pkl'

    if mp.exists() and rp.exists() and lp.exists():
        if input("Use existing model? (y/n): ").lower() == 'y':
            with open(lp, 'rb') as f:
                m = pickle.load(f)
            tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
            model = BertForMultiLabelNER(config.MODEL_PATH, len(m['label2idx'])).to(device)
            model.load_state_dict(torch.load(mp, map_location=device))
            with open(rp, 'r') as f:
                wr = json.load(f)
            print(f"\nloaded existing wojood model")
            print_detailed_metrics(wr['test_metrics']['metrics'], "Wojood Test Set Performance")
            return model, tokenizer, m['label2idx'], m['idx2label'], wr

    train_s = load_wojood_data(config.WOJOOD_TRAIN)
    val_s = load_wojood_data(config.WOJOOD_VAL)
    test_s = load_wojood_data(config.WOJOOD_TEST)
    label2idx, idx2label = create_label_mappings(train_s + val_s + test_s)
    with open(lp, 'wb') as f:
        pickle.dump({'label2idx': label2idx, 'idx2label': idx2label}, f)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = BertForMultiLabelNER(config.MODEL_PATH, len(label2idx)).to(device)
    train_dl = DataLoader(MultiLabelNERDataset(train_s, tokenizer, label2idx, config.MAX_LEN), config.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(MultiLabelNERDataset(val_s, tokenizer, label2idx, config.MAX_LEN), config.BATCH_SIZE)
    test_dl = DataLoader(MultiLabelNERDataset(test_s, tokenizer, label2idx, config.MAX_LEN), config.BATCH_SIZE)

    opt = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, config.WARMUP_STEPS, len(train_dl) * config.NUM_EPOCHS)
    criterion = FocalLoss(config.FOCAL_ALPHA, config.FOCAL_GAMMA)

    history = {
        'train_loss': [], 'val_loss': [],
        'val_micro_f1': [], 'val_macro_f1': [],
        'val_micro_precision': [], 'val_macro_precision': [],
        'val_micro_recall': [], 'val_macro_recall': [],
    }
    best_f1, patience = 0, 0
    start = time.time()

    for ep in range(config.NUM_EPOCHS):
        tl = train_epoch(model, train_dl, opt, sched, criterion, device)
        vr = evaluate(model, val_dl, val_s, criterion, device, idx2label)

        history['train_loss'].append(tl)
        history['val_loss'].append(vr['loss'])
        history['val_micro_f1'].append(vr.get('micro_f1', 0))
        history['val_macro_f1'].append(vr.get('macro_f1', 0))
        history['val_micro_precision'].append(vr.get('micro_precision', 0))
        history['val_macro_precision'].append(vr.get('macro_precision', 0))
        history['val_micro_recall'].append(vr.get('micro_recall', 0))
        history['val_macro_recall'].append(vr.get('macro_recall', 0))

        vf1 = vr.get('macro_f1', 0)
        print(f"Epoch {ep + 1}: TrLoss={tl:.4f} VaLoss={vr['loss']:.4f} | "
              f"Micro: P={vr.get('micro_precision', 0):.4f} R={vr.get('micro_recall', 0):.4f} F1={vr.get('micro_f1', 0):.4f} | "
              f"Macro F1={vf1:.4f}")

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), mp)
            patience = 0
        else:
            patience += 1
        if patience >= config.PATIENCE:
            print(f"early stop at {ep + 1}")
            break

    train_time = time.time() - start
    model.load_state_dict(torch.load(mp))
    test_res = evaluate(model, test_dl, test_s, criterion, device, idx2label)

    if test_res.get('metrics'):
        save_classification_report(test_res['metrics'], Path(config.RESULTS_DIR) / 'wojood_test_report.txt')
        print_detailed_metrics(test_res['metrics'], "Wojood Test Set Performance")

    results = {
        'best_val_f1': best_f1,
        'epochs_trained': ep + 1,
        'training_time_seconds': train_time,
        'test_metrics': test_res,
        'history': history
    }
    with open(rp, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    return model, tokenizer, label2idx, idx2label, results

def benchmark_on_adabner(model, tokenizer, woj_l2i, woj_i2l):
    print("\n" + "=" * 70 + "\nBENCHMARKING ON ADABNER (ZERO-SHOT)\n" + "=" * 70)
    adabner_test = load_adabner_test_data()

    adabner_mp = Path('results/bert_experiments/label_mappings.pkl')
    with open(adabner_mp, 'rb') as f:
        ym = pickle.load(f)
    adabner_l2i, adabner_i2l = ym['label2idx'], ym['idx2label']

    woj_labels, adabner_labels = set(woj_l2i.keys()), set(adabner_l2i.keys())
    common = woj_labels & adabner_labels
    semantic_map = {'B-WORK_OF_ART': 'B-PRODUCT', 'I-WORK_OF_ART': 'I-PRODUCT'}
    adabner_mapped = {l for l in (adabner_labels - common) if l in semantic_map}
    adabner_unmapped = adabner_labels - common - adabner_mapped

    _cov = (len(common) + len(adabner_mapped)) / len(adabner_labels) * 100
    print(f"  label coverage: {len(common)} common + {len(adabner_mapped)} mapped / {len(adabner_labels)} adabner ({_cov:.1f}%)")

    woj_to_adabner = {woj_l2i[l]: l for l in common}
    for yl, wl in semantic_map.items():
        if yl in adabner_labels and wl in woj_labels:
            woj_to_adabner[woj_l2i[wl]] = yl

    model.eval()
    all_true, all_pred, all_words = [], [], []

    with torch.no_grad():
        for sent in tqdm(adabner_test, desc="Evaluating on AdabNER"):
            words = [str(w) for w in sent['words']]
            enc = tokenizer(words, is_split_into_words=True, truncation=True, padding='max_length',
                            max_length=config.MAX_LEN, return_tensors='pt', add_special_tokens=True)
            logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
            preds = (torch.sigmoid(logits) > config.PREDICTION_THRESHOLD).float()
            word_ids = enc.word_ids(0)

            st, sp, pw = [], [], []
            prev = None
            for ti, wi in enumerate(word_ids):
                if wi is not None and wi != prev and wi < len(words):
                    st.append(sent['labels'][wi] if wi < len(sent['labels']) else ['O'])
                    pi = torch.where(preds[0, ti] == 1)[0].cpu().numpy()
                    sp.append([woj_to_adabner[int(i)] for i in pi if int(i) in woj_to_adabner] or ['O'])
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
            'micro_recall': nm['micro']['recall'],
            'micro_f1': nm['micro']['f1'],
            'macro_precision': nm['macro']['precision'],
            'macro_recall': nm['macro']['recall'],
            'macro_f1': nm['macro']['f1'],
        }
        save_classification_report(nm, Path(config.RESULTS_DIR) / 'transfer_test_report.txt')
        print_detailed_metrics(nm, "Zero-Shot Transfer Results (Wojood → AdabNER)")

    transfer = {
        'source': 'Wojood',
        'target': 'AdabNER',
        'label_overlap': {
            'common_labels': len(common),
            'semantically_mapped': len(adabner_mapped),
            'target_only': len(adabner_unmapped),
            'source_only': len(woj_labels - adabner_labels),
            'coverage_percentage': (len(common) + len(adabner_mapped)) / len(adabner_labels) * 100,
            'common_label_list': sorted(list(common)),
            'unmapped_list': sorted(list(adabner_unmapped))
        },
        'results': results
    }

    with open(Path(config.RESULTS_DIR) / 'transfer_results.json', 'w') as f:
        json.dump(transfer, f, indent=2, default=float)

    return transfer

def create_training_plots(history, results_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['val_micro_f1'], 'g-', label='Micro F1', linewidth=2)
    axes[0, 1].plot(epochs, history['val_macro_f1'], 'b-', label='Macro F1', linewidth=2)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Validation F1 Scores'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['val_micro_precision'], 'g-', label='Micro', linewidth=2)
    axes[1, 0].plot(epochs, history['val_macro_precision'], 'b-', label='Macro', linewidth=2)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Validation Precision'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['val_micro_recall'], 'g-', label='Micro', linewidth=2)
    axes[1, 1].plot(epochs, history['val_macro_recall'], 'b-', label='Macro', linewidth=2)
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Validation Recall'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(results_dir) / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved training curves to {results_dir}/training_curves.png")


def create_comparison_plot(woj_res, transfer_res, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    woj_metrics = woj_res['test_metrics']
    trans_metrics = transfer_res['results']

    metrics_names = ['Micro F1', 'Macro F1']
    woj_values = [woj_metrics.get('micro_f1', 0), woj_metrics.get('macro_f1', 0)]
    trans_values = [trans_metrics.get('micro_f1', 0), trans_metrics.get('macro_f1', 0)]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = axes[0].bar(x - width / 2, woj_values, width, label='Wojood (In-Domain)', color='steelblue', alpha=0.8)
    bars2 = axes[0].bar(x + width / 2, trans_values, width, label='Transfer (AdabNER)', color='coral', alpha=0.8)

    axes[0].set_ylabel('Score'); axes[0].set_title('F1 Score Comparison')
    axes[0].set_xticks(x); axes[0].set_xticklabels(metrics_names)
    axes[0].legend(); axes[0].set_ylim([0, 1]); axes[0].grid(True, alpha=0.3, axis='y')

    for bar in bars1 + bars2:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.02, f'{height:.3f}',
                     ha='center', va='bottom', fontsize=9)

    metrics_types = ['Micro', 'Macro']
    x = np.arange(len(metrics_types))
    woj_prec = [woj_metrics.get('micro_precision', 0), woj_metrics.get('macro_precision', 0)]
    woj_rec = [woj_metrics.get('micro_recall', 0), woj_metrics.get('macro_recall', 0)]
    trans_prec = [trans_metrics.get('micro_precision', 0), trans_metrics.get('macro_precision', 0)]
    trans_rec = [trans_metrics.get('micro_recall', 0), trans_metrics.get('macro_recall', 0)]

    width = 0.2
    axes[1].bar(x - 1.5 * width, woj_prec, width, label='Wojood Prec', color='steelblue', alpha=0.8)
    axes[1].bar(x - 0.5 * width, woj_rec, width, label='Wojood Rec', color='steelblue', alpha=0.5)
    axes[1].bar(x + 0.5 * width, trans_prec, width, label='Transfer Prec', color='coral', alpha=0.8)
    axes[1].bar(x + 1.5 * width, trans_rec, width, label='Transfer Rec', color='coral', alpha=0.5)

    axes[1].set_ylabel('Score'); axes[1].set_title('Precision & Recall Comparison')
    axes[1].set_xticks(x); axes[1].set_xticklabels(metrics_types)
    axes[1].legend(loc='lower right'); axes[1].set_ylim([0, 1]); axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(Path(results_dir) / 'comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved comparison plot to {results_dir}/comparison_plot.png")

def main():
    print("=" * 70 + "\nWOJOOD BENCHMARK\n" + "=" * 70)

    model, tokenizer, l2i, i2l, woj_res = train_on_wojood()

    if 'history' in woj_res and woj_res['history']['train_loss']:
        create_training_plots(woj_res['history'], config.FIGURES_DIR)

    transfer_res = benchmark_on_adabner(model, tokenizer, l2i, i2l)
    create_comparison_plot(woj_res, transfer_res, config.FIGURES_DIR)

    print("\n" + "=" * 70)
    print("final summary")
    print("=" * 70)

    woj_test = woj_res['test_metrics']
    trans = transfer_res['results']

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                      performance comparison                         │")
    print("├─────────────────┬─────────────────────┬─────────────────────────────┤")
    print("│ metric          │ wojood (in-domain)  │ transfer (adabner)        │")
    print("├─────────────────┼─────────────────────┼─────────────────────────────┤")
    print(f"│ micro precision │ {woj_test.get('micro_precision', 0):.4f}│{trans.get('micro_precision', 0):.4f}│")
    print(f"│ micro recall    │ {woj_test.get('micro_recall', 0):.4f}│{trans.get('micro_recall', 0):.4f}│")
    print(f"│ micro f1        │ {woj_test.get('micro_f1', 0):.4f}│{trans.get('micro_f1', 0):.4f}│")
    print("├─────────────────┼─────────────────────┼─────────────────────────────┤")
    print(f"│ macro precision │ {woj_test.get('macro_precision', 0):.4f}│{trans.get('macro_precision', 0):.4f}│")
    print(f"│ macro recall    │ {woj_test.get('macro_recall', 0):.4f}│{trans.get('macro_recall', 0):.4f}│")
    print(f"│ macro f1        │ {woj_test.get('macro_f1', 0):.4f}│{trans.get('macro_f1', 0):.4f}│")
    print("└─────────────────┴─────────────────────┴─────────────────────────────┘")

    print("\n" + "-" * 70)
    print("transfer gap analysis")
    print("-" * 70)
    gaps = {
        'micro f1': woj_test.get('micro_f1', 0) - trans.get('micro_f1', 0),
        'macro f1': woj_test.get('macro_f1', 0) - trans.get('macro_f1', 0),
    }
    for name, gap in gaps.items():
        print(f"  {name} gap: {gap:.4f} ({gap * 100:.1f}%)")

    avg_gap = np.mean(list(gaps.values()))
    print(f"\n  average f1 gap: {avg_gap:.4f} ({avg_gap * 100:.1f}%)")

    print("\n" + "-" * 70)
    print("label coverage")
    print("-" * 70)
    lc = transfer_res['label_overlap']
    print(f"  common labels:       {lc['common_labels']}")
    print(f"  semantically mapped: {lc['semantically_mapped']}")
    print(f"  target unmapped:     {lc['target_only']}")
    print(f"  source only:         {lc['source_only']}")
    print(f"  coverage:            {lc['coverage_percentage']:.1f}%")

    print(f"\nresults saved to: {config.RESULTS_DIR}")
    print(f"figures saved to: {config.FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
