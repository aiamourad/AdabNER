import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pickle
import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import random
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (set_seed, BertForMultiLabelNER, FocalLoss, MultiLabelNERDataset,
                 NestedNERMetrics, save_classification_report, save_averaged_report,
                 compute_averaged_metrics, remove_diacritics, normalize_arabic,
                 add_duplicate_suffixes)

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

class MultiModelConfig:
    MODELS = {
        'AraBERTv2': {
            'path': 'aubmindlab/bert-base-arabertv02',
            'name': 'AraBERTv2 (AubMind)',
        },
        'AraBERTv1': {
            'path': 'aubmindlab/bert-base-arabertv01',
            'name': 'AraBERT (AubMind)',
        },
        'CAMeLBERT': {
            'path': 'CAMeL-Lab/bert-base-arabic-camelbert-ca',
            'name': 'CAMeLBERT: CAMeL-Lab',
        },
        'ARBERT': {
            'path': 'UBC-NLP/ARBERT',
            'name': 'ARBERT (UBC-NLP)',
        },
        'ARBERTv2': {
            'path': 'UBC-NLP/ARBERTv2',
            'name': 'ARBERTv2 (UBC-NLP)',
        }
    }

    DATA_PATH = 'data/adabner/merged_up_to_date_sent_id.csv'
    SPLITS_DIR = 'data/adabner'
    STRATEGIES = ['stratified_book', 'leave_book_out']

    MC_SEEDS = [42, 1, 123]

    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 6e-5
    WARMUP_STEPS = 500
    PATIENCE = 5
    MIN_DELTA = 0.001
    MAX_LEN = 512

    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 1.0
    PREDICTION_THRESHOLD = 0.5

    OUTPUT_DIR = 'results/bert_experiments'
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.10
    TEST_RATIO = 0.10

config = MultiModelConfig()

def load_and_process_data(rank=0):

    df = pd.read_csv(config.DATA_PATH)
    if rank == 0:
        print(f"loaded {len(df):,} rows")

    df['token'] = df['token'].astype(str).apply(remove_diacritics).apply(normalize_arabic)

    sentences = []
    nested_count, same_type_count, max_dup = 0, 0, 1
    grouped = df.groupby(['book_id', 'sent_id'])

    for (book_id, sent_id), group in grouped:
        words = group['token'].tolist()
        raw_labels = []
        for tag in group['ner_tag'].values:
            if pd.isna(tag) or tag == 'O':
                raw_labels.append(['O'])
            else:
                raw_labels.append(str(tag).split())

        has_nesting = any(len(l) > 1 for l in raw_labels)
        if has_nesting:
            nested_count += 1

        for labels in raw_labels:
            if len(labels) != len(set(labels)):
                same_type_count += 1
                max_dup = max(max_dup, max(Counter(labels).values()))
                break

        processed_labels = add_duplicate_suffixes(raw_labels)
        sentences.append({'book_id': book_id, 'sentence_id': f"{book_id}_{sent_id}",
                          'words': words, 'labels': processed_labels})

    if rank == 0:
        print(f"processed {len(sentences):,} sentences")
        print(f"  > with nesting: {nested_count:,} ({100 * nested_count / len(sentences):.1f}%)")

    unique_labels = set()
    for sent in sentences:
        for label_list in sent['labels']:
            unique_labels.update(label_list)

    all_labels = sorted(list(unique_labels))
    label2idx = {label: i for i, label in enumerate(all_labels)}
    idx2label = {i: label for label, i in label2idx.items()}

    if rank == 0:
        print(f"found {len(all_labels)} unique labels")

    return sentences, label2idx, idx2label

def split_data_by_strategy(sentences_data, strategy, train_r=0.7, val_r=0.15, test_r=0.15):
    split_dir = Path(config.SPLITS_DIR)
    train_path = split_dir / f'train_{strategy}.pkl'
    val_path = split_dir / f'val_{strategy}.pkl'
    test_path = split_dir / f'test_{strategy}.pkl'

    if train_path.exists() and val_path.exists() and test_path.exists():
        try:
            with open(train_path, 'rb') as f:
                train_data = pickle.load(f)
            with open(val_path, 'rb') as f:
                val_data = pickle.load(f)
            with open(test_path, 'rb') as f:
                test_data = pickle.load(f)
            print(f"loaded existing {strategy} splits")
            return train_data, val_data, test_data
        except (UnicodeDecodeError, pickle.UnpicklingError, EOFError):
            print(f"corrupted pickle files detected, regenerating splits...")
            train_path.unlink(missing_ok=True)
            val_path.unlink(missing_ok=True)
            test_path.unlink(missing_ok=True)

    train_data, val_data, test_data = [], [], []
    book_sents = defaultdict(list)
    for s in sentences_data:
        book_sents[s['book_id']].append(s)

    if strategy == 'stratified_book':
        for book_id, sents in book_sents.items():
            n = len(sents)
            random.shuffle(sents)
            t1, t2 = int(train_r * n), int((train_r + val_r) * n)
            train_data.extend(sents[:t1])
            val_data.extend(sents[t1:t2])
            test_data.extend(sents[t2:])
    elif strategy == 'leave_book_out':
        books = list(book_sents.keys())
        random.shuffle(books)
        n = len(books)
        t1, t2 = int(train_r * n), int((train_r + val_r) * n)
        for b in books[:t1]:
            train_data.extend(book_sents[b])
        for b in books[t1:t2]:
            val_data.extend(book_sents[b])
        for b in books[t2:]:
            test_data.extend(book_sents[b])
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)

    return train_data, val_data, test_data

def train_epoch_simple(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)

        valid = (labels != -100)
        loss = criterion(logits, torch.where(valid, labels, torch.zeros_like(labels)))
        loss = (loss * valid.float()).sum() / valid.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_simple(model, loader, data, criterion, device, idx2label, threshold=0.5):
    model.eval()
    total_loss = 0
    all_true, all_pred, all_words = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(ids, mask)
            probs = torch.sigmoid(logits)

            valid = (labels != -100)
            loss = criterion(logits, torch.where(valid, labels, torch.zeros_like(labels)))
            loss = (loss * valid.float()).sum() / valid.sum()
            total_loss += loss.item()

            preds = (probs > threshold).float()

            for i in range(labels.size(0)):
                seq_true, seq_pred = [], []
                for j in range(labels.size(1)):
                    if (labels[i, j] != -100).any():
                        true_idx = torch.where(labels[i, j] == 1)[0].cpu().numpy()
                        pred_idx = torch.where(preds[i, j] == 1)[0].cpu().numpy()
                        seq_true.append([idx2label[int(x)] for x in true_idx] or ['O'])
                        seq_pred.append([idx2label[int(x)] for x in pred_idx] or ['O'])

                if seq_true:
                    data_idx = batch_idx * loader.batch_size + i
                    if data_idx < len(data):
                        all_true.append(seq_true)
                        all_pred.append(seq_pred)
                        all_words.append(data[data_idx]['words'][:len(seq_true)])

    results = {'loss': total_loss / len(loader)}
    if all_true:
        nested = NestedNERMetrics.compute_nested_metrics(all_true, all_pred, all_words)
        results['metrics'] = nested
        results['macro_f1'] = nested['macro']['f1']
        results['micro_f1'] = nested['micro']['f1']
    return results

def run_experiment_mc_simple(model_key, strategy, sentences, label2idx, idx2label, num_labels):
    print(f"\n{'=' * 70}")
    print(f"model: {config.MODELS[model_key]['name']} | strategy: {strategy.upper()}")
    print(f"monte carlo seeds: {config.MC_SEEDS}")
    print(f"{'=' * 70}")

    exp_dir = Path(config.OUTPUT_DIR) / f"{model_key}_{strategy}"
    exp_dir.mkdir(exist_ok=True)

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"using {n_gpus} gpu(s): {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
        device = torch.device("cuda")
    else:
        print("no gpu available, using cpu")
        device = torch.device("cpu")

    model_path = config.MODELS[model_key]['path']

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"tokenizer failed: {e}")
        return None

    train_data, val_data, test_data = split_data_by_strategy(sentences, strategy)

    seed_results = []

    for seed in config.MC_SEEDS:
        print(f"\n--- running with seed={seed} ---")
        set_seed(seed)

        train_ds = MultiLabelNERDataset(train_data, tokenizer, label2idx, config.MAX_LEN)
        val_ds = MultiLabelNERDataset(val_data, tokenizer, label2idx, config.MAX_LEN)
        test_ds = MultiLabelNERDataset(test_data, tokenizer, label2idx, config.MAX_LEN)

        effective_batch_size = config.BATCH_SIZE * max(1, torch.cuda.device_count()) if torch.cuda.is_available() else config.BATCH_SIZE

        train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=effective_batch_size, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=effective_batch_size, num_workers=4, pin_memory=True)

        try:
            model = BertForMultiLabelNER(model_path, num_labels)

            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                print(f"wrapping model with dataparallel on {torch.cuda.device_count()} gpus")
                model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

            model = model.to(device)
            print(f"model initialized ({num_labels} labels) - seed {seed}")
        except Exception as e:
            print(f"model failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        model_for_optim = model.module if hasattr(model, 'module') else model

        optimizer = AdamW(model_for_optim.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, config.WARMUP_STEPS, len(train_loader) * config.NUM_EPOCHS)
        criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        best_f1, patience_counter = 0, 0
        start = time.time()

        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_epoch_simple(model, train_loader, optimizer, scheduler, criterion, device)
            val_res = evaluate_simple(model, val_loader, val_data, criterion, device, idx2label)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_res['loss'])
            val_f1 = val_res.get('macro_f1', 0)
            history['val_f1'].append(val_f1)

            print(f"  seed {seed} | epoch {epoch + 1}: train loss={train_loss:.4f}, val f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), exp_dir / f'best_model_seed{seed}.pt')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print(f"  early stopping at epoch {epoch + 1}")
                    break

        train_time = time.time() - start

        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(exp_dir / f'best_model_seed{seed}.pt'))

        test_final = evaluate_simple(model, test_loader, test_data, criterion, device, idx2label)
        test_metrics = test_final.get('metrics', {})

        if test_metrics:
            save_classification_report(test_metrics, exp_dir / f'test_report_seed{seed}.txt')

        seed_result = {
            'seed': seed,
            'epochs': epoch + 1,
            'time_min': train_time / 60,
            'best_val_f1': best_f1,
            'metrics': test_metrics,
        }
        seed_results.append(seed_result)

        with open(exp_dir / f'seed_{seed}_results.json', 'w') as f:
            json.dump(seed_result, f, indent=2)

    if seed_results:
        save_averaged_report(seed_results, exp_dir / 'averaged_classification_report.txt', seeds=config.MC_SEEDS)
        avg_metrics = compute_averaged_metrics(seed_results)

        final_results = {
            'model': model_key,
            'model_name': config.MODELS[model_key]['name'],
            'strategy': strategy,
            'seeds': config.MC_SEEDS,
            'per_seed_results': seed_results,
            'averaged': avg_metrics
        }

        with open(exp_dir / 'all_seeds_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)

        print_summary(final_results)
        return final_results

    return None


def print_summary(results):
    avg = results['averaged']
    print(f"\n{'=' * 70}")
    print(f"averaged results (seeds: {results['seeds']})")
    print(f"{'=' * 70}")
    print(f"{'Metric':<20} {'F1 (mean±std)':>20} {'Precision':>20} {'Recall':>20}")
    print("-" * 70)
    for mt in ['micro', 'macro']:
        m = avg[mt]
        print(f"{mt.capitalize():<20} {m['f1_mean']:.4f}±{m['f1_std']:.4f}      "
              f"{m['precision_mean']:.4f}±{m['precision_std']:.4f}      "
              f"{m['recall_mean']:.4f}±{m['recall_std']:.4f}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("ner experiments with multiple models and strategies")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"  device: cuda | gpus: {torch.cuda.device_count()} ({[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})")
    else:
        print("  device: cpu")

    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    Path(config.SPLITS_DIR).mkdir(exist_ok=True)

    sentences, label2idx, idx2label = load_and_process_data()
    num_labels = len(label2idx)

    with open(Path(config.OUTPUT_DIR) / 'label_mappings.pkl', 'wb') as f:
        pickle.dump({'label2idx': label2idx, 'idx2label': idx2label}, f)

    all_results = {}

    for model_key in config.MODELS:
        for strategy in config.STRATEGIES:
            try:
                res = run_experiment_mc_simple(model_key, strategy, sentences, label2idx, idx2label, num_labels)
                if res:
                    all_results[f"{model_key}_{strategy}"] = res
            except Exception as e:
                print(f"{model_key}_{strategy} failed: {e}")
                import traceback
                traceback.print_exc()

    with open(Path(config.OUTPUT_DIR) / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("all experiments completed. summary of averaged results:")
    print("=" * 70)
    print(f"{'Model':<15} {'Strategy':<18} {'Micro-F1':>20} {'Macro-F1':>20}")
    print("-" * 70)

    for key in sorted(all_results.keys()):
        res = all_results[key]
        avg = res.get('averaged', {})
        micro = avg.get('micro', {})
        macro = avg.get('macro', {})

        print(f"{res['model']:<15} {res['strategy']:<18} "
              f"{micro.get('f1_mean', 0):.4f}±{micro.get('f1_std', 0):.3f}  "
              f"{macro.get('f1_mean', 0):.4f}±{macro.get('f1_std', 0):.3f}")

    print("=" * 70)
    print(f"\nresults saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
