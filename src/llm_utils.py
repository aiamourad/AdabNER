import pickle
import json
import random
import re
from pathlib import Path
from collections import defaultdict
ENTITY_TYPES = [
    'PERS', 'WORK_OF_ART', 'GPE', 'FAC', 'NORP', 'EVENT', 'OCC', 'LANGUAGE',
    'DATE', 'UNIT', 'CARDINAL', 'QUANTITY', 'LOC', 'CURR', 'ORG', 'MONEY',
    'ORDINAL', 'LAW', 'TIME', 'PERCENT', 'PRODUCT'
]

ENTITY_DESCRIPTIONS = {
    'PERS': 'Person names',
    'GPE': 'Geopolitical entities (countries, cities, states)',
    'ORG': 'Organizations (companies, agencies, institutions)',
    'LOC': 'Locations (mountains, rivers, regions)',
    'DATE': 'Date expressions',
    'TIME': 'Time expressions',
    'MONEY': 'Monetary values',
    'PERCENT': 'Percentage expressions',
    'CARDINAL': 'Cardinal numbers',
    'ORDINAL': 'Ordinal numbers',
    'QUANTITY': 'Quantities with units',
    'NORP': 'Nationalities, religious, political groups',
    'FAC': 'Facilities (buildings, airports, highways)',
    'EVENT': 'Events (wars, battles, sports events)',
    'WORK_OF_ART': 'Titles of books, songs, artworks',
    'LAW': 'Laws, treaties, regulations',
    'LANGUAGE': 'Languages',
    'PRODUCT': 'Products (vehicles, weapons, foods)',
    'OCC': 'Occupations/Professions',
    'UNIT': 'Units of measurement',
    'CURR': 'Currency names'
}


def load_split_data(strategy, splits_dir):
    splits_dir = Path(splits_dir)
    with open(splits_dir / f'train_{strategy}.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(splits_dir / f'val_{strategy}.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open(splits_dir / f'test_{strategy}.pkl', 'rb') as f:
        test_data = pickle.load(f)
    print(f"loaded {strategy}: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data

def extract_entities_from_labels(words, labels):
    entities = []
    current_entities = {}

    for pos, (word, label_list) in enumerate(zip(words, labels)):
        if isinstance(label_list, str):
            label_list = [label_list]

        for label in label_list:
            if label == 'O' or '-' not in label:
                continue

            prefix, rest = label.split('-', 1)
            base_type = rest.split('_')[0] if '_' in rest and rest.rsplit('_', 1)[1].isdigit() else rest
            tracker_key = rest

            if prefix == 'B':
                if tracker_key in current_entities:
                    start, tokens, etype = current_entities[tracker_key]
                    entities.append({'type': etype, 'text': ' '.join(tokens), 'start': start, 'end': pos - 1})
                current_entities[tracker_key] = (pos, [word], base_type)
            elif prefix == 'I':
                if tracker_key in current_entities:
                    current_entities[tracker_key][1].append(word)
                else:
                    current_entities[tracker_key] = (pos, [word], base_type)

    for tracker_key, (start, tokens, etype) in current_entities.items():
        entities.append({'type': etype, 'text': ' '.join(tokens), 'start': start, 'end': len(words) - 1})

    return entities

def format_entities_for_output(entities):
    if not entities:
        return "[]"
    formatted = []
    for ent in entities:
        formatted.append(f'{{"type": "{ent["type"]}", "text": "{ent["text"]}"}}')
    return "[\n  " + ",\n  ".join(formatted) + "\n]"

def select_diverse_examples(train_data, num_examples=5):
    examples_by_type = defaultdict(list)

    for item in train_data:
        entities = extract_entities_from_labels(item['words'], item['labels'])
        entity_types = set(e['type'] for e in entities)
        if entities:
            for etype in entity_types:
                examples_by_type[etype].append((item, entities))

    selected = []
    selected_indices = set()
    type_queue = list(ENTITY_TYPES)
    random.shuffle(type_queue)

    while len(selected) < num_examples and type_queue:
        for etype in type_queue[:]:
            if len(selected) >= num_examples:
                break
            if examples_by_type[etype]:
                candidates = [ex for ex in examples_by_type[etype] if id(ex[0]) not in selected_indices]
                if candidates:
                    item, entities = random.choice(candidates)
                    selected.append((item, entities))
                    selected_indices.add(id(item))
                    type_queue.remove(etype)

    while len(selected) < num_examples:
        item = random.choice(train_data)
        if id(item) not in selected_indices:
            entities = extract_entities_from_labels(item['words'], item['labels'])
            selected.append((item, entities))
            selected_indices.add(id(item))

    return selected


def build_system_prompt():
    entity_list = "\n".join([f"- {k}: {v}" for k, v in ENTITY_DESCRIPTIONS.items()])

    return f"""You are an expert Named Entity Recognition (NER) system for Arabic text. Your task is to identify and extract all named entities from the given Arabic text.

IMPORTANT: This is a NESTED NER task. A single token can belong to multiple entities simultaneously.

Entity Types:
{entity_list}

Output Format:
- Return a JSON array of entities
- Each entity must have "type" and "text" fields
- If no entities found, return an empty array: []
- Extract ALL entities, including nested/overlapping ones

CRITICAL: Your response must contain ONLY the JSON array, nothing else. No explanations, no markdown, just the JSON."""


def build_few_shot_prompt(examples, test_sentence):
    prompt_parts = []

    for i, (item, entities) in enumerate(examples, 1):
        sentence = ' '.join(item['words'])
        entities_json = format_entities_for_output(entities)
        prompt_parts.append(f"""Example {i}:
Input: {sentence}
Output: {entities_json}""")

    test_text = ' '.join(test_sentence['words'])
    prompt_parts.append(f"""Now extract entities from this text:
Input: {test_text}
Output:""")

    return "\n\n".join(prompt_parts)

def parse_model_output(output_text):
    output_text = output_text.strip()

    output_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()

    if output_text.startswith("```"):
        output_text = re.sub(r'^```(?:json)?\n?', '', output_text)
        output_text = re.sub(r'\n?```$', '', output_text)
    output_text = output_text.strip()

    if not output_text or output_text == "[]":
        return []

    try:
        entities = json.loads(output_text)
        if isinstance(entities, list):
            valid_entities = []
            for e in entities:
                if isinstance(e, dict) and 'type' in e and 'text' in e:
                    if e['type'] in ENTITY_TYPES:
                        valid_entities.append({'type': e['type'], 'text': e['text']})
            return valid_entities
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[[\s\S]*?\]', output_text)
    if match:
        try:
            entities = json.loads(match.group())
            if isinstance(entities, list):
                valid_entities = []
                for e in entities:
                    if isinstance(e, dict) and 'type' in e and 'text' in e:
                        if e['type'] in ENTITY_TYPES:
                            valid_entities.append({'type': e['type'], 'text': e['text']})
                return valid_entities
        except json.JSONDecodeError:
            pass

    entities = []
    pattern = r'\{\s*"type"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"([^"]+)"\s*\}'
    matches = re.findall(pattern, output_text)
    for etype, text in matches:
        if etype in ENTITY_TYPES:
            entities.append({'type': etype, 'text': text})

    return entities

class LLMNERMetrics:
    @staticmethod
    def compute_metrics(true_entities_list, pred_entities_list):
        results = {'per_type': {}}
        all_types = set()

        for true_ents, pred_ents in zip(true_entities_list, pred_entities_list):
            all_types.update(e['type'] for e in true_ents)
            all_types.update(e['type'] for e in pred_ents)

        total_tp, total_fp, total_fn = 0, 0, 0

        for etype in all_types:
            tp, fp, fn = 0, 0, 0
            for true_ents, pred_ents in zip(true_entities_list, pred_entities_list):
                true_set = {e['text'] for e in true_ents if e['type'] == etype}
                pred_set = {e['text'] for e in pred_ents if e['type'] == etype}
                tp += len(true_set & pred_set)
                fp += len(pred_set - true_set)
                fn += len(true_set - pred_set)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            results['per_type'][etype] = {
                'precision': p, 'recall': r, 'f1': f1,
                'tp': tp, 'fp': fp, 'fn': fn, 'support': tp + fn
            }
            total_tp += tp
            total_fp += fp
            total_fn += fn

        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
        results['micro'] = {'precision': micro_p, 'recall': micro_r, 'f1': micro_f1}

        if results['per_type']:
            macro_p = sum(m['precision'] for m in results['per_type'].values()) / len(results['per_type'])
            macro_r = sum(m['recall'] for m in results['per_type'].values()) / len(results['per_type'])
            macro_f1 = sum(m['f1'] for m in results['per_type'].values()) / len(results['per_type'])
        else:
            macro_p = macro_r = macro_f1 = 0
        results['macro'] = {'precision': macro_p, 'recall': macro_r, 'f1': macro_f1}

        return results
