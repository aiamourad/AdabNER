import numpy as np
from collections import defaultdict


class NestedNERMetrics:
    @staticmethod
    def extract_entities(labels_sequence, words_sequence):
        entities = defaultdict(list)
        current_entities = {}

        for pos, (label_list, word) in enumerate(zip(labels_sequence, words_sequence)):
            for label in label_list:
                if label == 'O' or '-' not in label:
                    continue
                prefix, rest = label.split('-', 1)
                base_type, tracker_key = rest, rest
                if '_' in rest:
                    parts = rest.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        base_type, tracker_key = parts[0], rest

                if prefix == 'B':
                    if tracker_key in current_entities:
                        start, tokens, clean_type = current_entities[tracker_key]
                        entities[clean_type].append((start, pos - 1, ' '.join(tokens)))
                    current_entities[tracker_key] = (pos, [word], base_type)
                elif prefix == 'I':
                    if tracker_key in current_entities:
                        current_entities[tracker_key][1].append(word)
                    else:
                        current_entities[tracker_key] = (pos, [word], base_type)

        for tracker_key, (start, tokens, clean_type) in current_entities.items():
            entities[clean_type].append((start, len(labels_sequence) - 1, ' '.join(tokens)))
        return entities

    @staticmethod
    def compute_nested_metrics(true_seqs, pred_seqs, words_seqs, allowed_types=None):
        results = {'per_type': {}}
        all_types = set()
        total_tp, total_fp, total_fn = 0, 0, 0
        
        for true_seq, pred_seq, words in zip(true_seqs, pred_seqs, words_seqs):
            true_ents = NestedNERMetrics.extract_entities(true_seq, words)
            pred_ents = NestedNERMetrics.extract_entities(pred_seq, words)
            if allowed_types is not None:
                true_ents = {k: v for k, v in true_ents.items() if k in allowed_types}
                pred_ents = {k: v for k, v in pred_ents.items() if k in allowed_types}
            all_types.update(true_ents.keys())
            all_types.update(pred_ents.keys())

        for etype in all_types:
            tp = fp = fn = 0
            for true_seq, pred_seq, words in zip(true_seqs, pred_seqs, words_seqs):
                true_set = set(NestedNERMetrics.extract_entities(true_seq, words).get(etype, []))
                pred_set = set(NestedNERMetrics.extract_entities(pred_seq, words).get(etype, []))
                tp += len(true_set & pred_set)
                fp += len(pred_set - true_set)
                fn += len(true_set - pred_set)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            results['per_type'][etype] = {'precision': p, 'recall': r, 'f1': f1,
                                          'tp': tp, 'fp': fp, 'fn': fn, 'support': tp + fn}
            total_tp += tp
            total_fp += fp
            total_fn += fn

        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
        results['micro'] = {'precision': micro_p, 'recall': micro_r, 'f1': micro_f1}

        if results['per_type']:
            macro_p = np.mean([m['precision'] for m in results['per_type'].values()])
            macro_r = np.mean([m['recall'] for m in results['per_type'].values()])
            macro_f1 = np.mean([m['f1'] for m in results['per_type'].values()])
        else:
            macro_p = macro_r = macro_f1 = 0
        results['macro'] = {'precision': macro_p, 'recall': macro_r, 'f1': macro_f1}

        return results


def save_classification_report(metrics, filepath, model_name=""):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        header = f"{model_name} - CLASSIFICATION REPORT" if model_name else "CLASSIFICATION REPORT"
        f.write(f"{header}\n")
        f.write("=" * 90 + "\n\n")

        f.write("PER-ENTITY TYPE METRICS\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Entity Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>8} {'FP':>8} {'FN':>8} {'Support':>8}\n")
        f.write("-" * 90 + "\n")

        sorted_types = sorted(metrics['per_type'].items(), key=lambda x: x[1]['f1'], reverse=True)
        for et, m in sorted_types:
            f.write(f"{et:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} "
                    f"{m['tp']:>8} {m['fp']:>8} {m['fn']:>8} {m['support']:>8}\n")

        f.write("-" * 90 + "\n\n")
        f.write("AGGREGATE METRICS\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Metric Type':<15} {'Precision':>12} {'Recall':>12} {'F1':>12}\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Micro':<15} {metrics['micro']['precision']:>12.4f} {metrics['micro']['recall']:>12.4f} {metrics['micro']['f1']:>12.4f}\n")
        f.write(f"{'Macro':<15} {metrics['macro']['precision']:>12.4f} {metrics['macro']['recall']:>12.4f} {metrics['macro']['f1']:>12.4f}\n")
        f.write("-" * 90 + "\n")


def save_averaged_report(seed_results, filepath, seeds=None, model_name=""):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        header = f"{model_name} - AVERAGED CLASSIFICATION REPORT" if model_name else "AVERAGED CLASSIFICATION REPORT"
        f.write(f"{header}\n")
        if seeds is not None:
            f.write(f"Seeds: {seeds}\n")
        f.write("=" * 100 + "\n\n")

        all_types = set()
        for sr in seed_results:
            if 'metrics' in sr and 'per_type' in sr['metrics']:
                all_types.update(sr['metrics']['per_type'].keys())

        f.write("PER-CLASS METRICS (Averaged ± Std):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Entity Type':<20} {'Prec Mean':>12} {'Prec Std':>10} {'Rec Mean':>10} {'Rec Std':>10} {'F1 Mean':>10} {'F1 Std':>10}\n")
        f.write("-" * 100 + "\n")

        for etype in sorted(all_types):
            valid = [sr for sr in seed_results
                     if 'metrics' in sr and etype in sr['metrics'].get('per_type', {})]
            if not valid:
                continue
            precs = [sr['metrics']['per_type'][etype]['precision'] for sr in valid]
            recs  = [sr['metrics']['per_type'][etype]['recall']    for sr in valid]
            f1s   = [sr['metrics']['per_type'][etype]['f1']        for sr in valid]
            f.write(f"{etype:<20} {np.mean(precs):>12.4f} {np.std(precs):>10.4f} "
                    f"{np.mean(recs):>10.4f} {np.std(recs):>10.4f} "
                    f"{np.mean(f1s):>10.4f} {np.std(f1s):>10.4f}\n")

        f.write("-" * 100 + "\n\n")
        f.write("AGGREGATE METRICS (Averaged ± Std):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Metric':<20} {'Prec Mean':>12} {'Prec Std':>10} {'Rec Mean':>10} {'Rec Std':>10} {'F1 Mean':>10} {'F1 Std':>10}\n")
        f.write("-" * 100 + "\n")

        for metric_type in ['micro', 'macro']:
            valid = [sr for sr in seed_results if 'metrics' in sr]
            if not valid:
                continue
            precs = [sr['metrics'][metric_type]['precision'] for sr in valid]
            recs  = [sr['metrics'][metric_type]['recall']    for sr in valid]
            f1s   = [sr['metrics'][metric_type]['f1']        for sr in valid]
            f.write(f"{metric_type.capitalize():<20} {np.mean(precs):>12.4f} {np.std(precs):>10.4f} "
                    f"{np.mean(recs):>10.4f} {np.std(recs):>10.4f} "
                    f"{np.mean(f1s):>10.4f} {np.std(f1s):>10.4f}\n")

        f.write("-" * 100 + "\n\n")
        f.write("PER-SEED SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Seed':<10} {'Micro-F1':>12} {'Macro-F1':>12} {'Micro-P':>12} {'Micro-R':>12}\n")
        f.write("-" * 80 + "\n")
        for sr in seed_results:
            if 'metrics' not in sr:
                continue
            m = sr['metrics']
            f.write(f"{sr['seed']:<10} {m['micro']['f1']:>12.4f} {m['macro']['f1']:>12.4f} "
                    f"{m['micro']['precision']:>12.4f} {m['micro']['recall']:>12.4f}\n")
        f.write("=" * 100 + "\n")


def compute_averaged_metrics(seed_results):
    avg = {}

    for metric_type in ['micro', 'macro']:
        valid = [sr for sr in seed_results if 'metrics' in sr]
        f1s   = [sr['metrics'][metric_type]['f1']        for sr in valid]
        precs = [sr['metrics'][metric_type]['precision'] for sr in valid]
        recs  = [sr['metrics'][metric_type]['recall']    for sr in valid]

        avg[metric_type] = {
            'f1_mean':        float(np.mean(f1s))   if f1s   else 0,
            'f1_std':         float(np.std(f1s))    if f1s   else 0,
            'precision_mean': float(np.mean(precs)) if precs else 0,
            'precision_std':  float(np.std(precs))  if precs else 0,
            'recall_mean':    float(np.mean(recs))  if recs  else 0,
            'recall_std':     float(np.std(recs))   if recs  else 0,
        }

    all_types = set()
    for sr in seed_results:
        if 'metrics' in sr:
            all_types.update(sr['metrics'].get('per_type', {}).keys())

    avg['per_type'] = {}
    for etype in all_types:
        valid = [sr for sr in seed_results if 'metrics' in sr and etype in sr['metrics'].get('per_type', {})]
        f1s   = [sr['metrics']['per_type'][etype]['f1']        for sr in valid]
        precs = [sr['metrics']['per_type'][etype]['precision'] for sr in valid]
        recs  = [sr['metrics']['per_type'][etype]['recall']    for sr in valid]
        sups  = [sr['metrics']['per_type'][etype]['support']   for sr in valid]

        avg['per_type'][etype] = {
            'f1_mean':        float(np.mean(f1s)),
            'f1_std':         float(np.std(f1s)),
            'precision_mean': float(np.mean(precs)),
            'precision_std':  float(np.std(precs)),
            'recall_mean':    float(np.mean(recs)),
            'recall_std':     float(np.std(recs)),
            'support':        int(np.mean(sups)),
        }

    return avg
