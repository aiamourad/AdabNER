import json
import random
import re
import time
import asyncio
from pathlib import Path
from tqdm import tqdm

from google import genai
from google.genai import types

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_utils import (
    load_split_data, extract_entities_from_labels, format_entities_for_output,
    select_diverse_examples, build_system_prompt,
    parse_model_output, LLMNERMetrics,
)
from src.metrics import save_classification_report, save_averaged_report, compute_averaged_metrics

class Config:
    MODEL_NAME = "gemini-3-pro-preview"
    SPLITS_DIR = 'data/adabner'
    OUTPUT_DIR = 'results/llm/gemini'
    STRATEGIES = ['stratified_book', 'leave_book_out']
    NUM_FEW_SHOT_EXAMPLES = 5
    MAX_NEW_TOKENS = 4096
    MAX_TEST_SAMPLES = None

    MC_SEEDS = [42, 1, 123]
    MAX_CONCURRENT = 5
    BATCH_SIZE = 10
    RETRY_MAX = 5
    RETRY_BACKOFF = 2


config = Config()

client = genai.Client(
    vertexai=True,
    project='',
    location=''
)
async_client = client.aio

DELIMITER = "###SENTENCE_{}###"


def build_batched_system_prompt():
    from src.llm_utils import ENTITY_DESCRIPTIONS
    elist = "\n".join(f"- {k}: {v}" for k, v in ENTITY_DESCRIPTIONS.items())
    return f"""You are an expert Named Entity Recognition (NER) system for Arabic text.

IMPORTANT: This is a NESTED NER task. A single token can belong to multiple entities simultaneously.

Entity Types:
{elist}

You will receive MULTIPLE sentences to process. For EACH sentence, output its entities using this EXACT format:

###SENTENCE_1###
[{{"type": "TYPE", "text": "entity text"}}, ...]
###SENTENCE_2###
[{{"type": "TYPE", "text": "entity text"}}, ...]

Rules:
- Use the delimiter ###SENTENCE_N### before each sentence's output (N = sentence number)
- Each sentence's output must be a valid JSON array
- If no entities found for a sentence, output: []
- Extract ALL entities, including nested/overlapping ones
- Output ONLY the delimited JSON arrays. No explanations, no markdown."""


def build_batched_few_shot_prompt(examples, batch_sentences):
    parts = []

    for i, (item, ents) in enumerate(examples, 1):
        sentence = ' '.join(item['words'])
        parts.append(f"Example {i}:\nInput:\nSentence 1: {sentence}\n\nOutput:\n###SENTENCE_1###\n{format_entities_for_output(ents)}")

    parts.append("Now extract entities from these sentences:")
    input_lines = []
    for i, item in enumerate(batch_sentences, 1):
        input_lines.append(f"Sentence {i}: {' '.join(item['words'])}")
    parts.append("Input:\n" + "\n".join(input_lines))
    parts.append("Output:")

    return "\n\n".join(parts)

def parse_batched_output(output_text, expected_count):
    output_text = output_text.strip()
    output_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()

    if output_text.startswith("```"):
        output_text = re.sub(r'^```(?:json)?\n?', '', output_text)
        output_text = re.sub(r'\n?```$', '', output_text)
    output_text = output_text.strip()

    results = [None] * expected_count

    pattern = r'###SENTENCE_(\d+)###\s*'
    segments = re.split(pattern, output_text)

    i = 1
    while i < len(segments) - 1:
        try:
            idx = int(segments[i]) - 1
            json_text = segments[i + 1].strip()
            if idx < expected_count:
                results[idx] = parse_single_json(json_text)
        except (ValueError, IndexError):
            pass
        i += 2

    for j in range(expected_count):
        if results[j] is None:
            results[j] = []

    return results


def parse_single_json(text):
    from src.llm_utils import ENTITY_TYPES
    text = text.strip()
    bracket_count = 0
    end_idx = 0
    for ci, c in enumerate(text):
        if c == '[':
            bracket_count += 1
        elif c == ']':
            bracket_count -= 1
        if bracket_count == 0 and ci > 0:
            end_idx = ci + 1
            break
    if end_idx > 0:
        text = text[:end_idx]

    if not text or text == "[]":
        return []
    try:
        entities = json.loads(text)
        if isinstance(entities, list):
            return [{'type': e['type'], 'text': e['text']} for e in entities
                    if isinstance(e, dict) and 'type' in e and 'text' in e and e['type'] in ENTITY_TYPES]
    except json.JSONDecodeError:
        pass

    ents = []
    for et, tx in re.findall(r'\{\s*"type"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"([^"]+)"\s*\}', text):
        from src.llm_utils import ENTITY_TYPES as ET
        if et in ET:
            ents.append({'type': et, 'text': tx})
    return ents

async def call_batch_async(system_prompt, user_prompt, semaphore, batch_idx, total_batches, batch_size):
    async with semaphore:
        for attempt in range(config.RETRY_MAX):
            try:
                response = await async_client.models.generate_content(
                    model=config.MODEL_NAME,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0,
                        max_output_tokens=config.MAX_NEW_TOKENS,
                        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                    ),
                )
                if response.text:
                    return batch_idx, response.text.strip()
                else:
                    await asyncio.sleep(config.RETRY_BACKOFF * (2 ** attempt))
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = config.RETRY_BACKOFF * (2 ** attempt) * 5
                    print(f"  [batch {batch_idx+1}/{total_batches}] rate limited, waiting {wait:.0f}s...")
                    await asyncio.sleep(wait)
                elif "500" in err or "503" in err:
                    await asyncio.sleep(config.RETRY_BACKOFF * (2 ** attempt))
                else:
                    print(f"  [batch {batch_idx+1}/{total_batches}] error: {e}")
                    if attempt == config.RETRY_MAX - 1:
                        return batch_idx, ""
                    await asyncio.sleep(config.RETRY_BACKOFF * (2 ** attempt))
        return batch_idx, ""


async def call_single_async(system_prompt, user_prompt, semaphore):
    async with semaphore:
        for attempt in range(config.RETRY_MAX):
            try:
                response = await async_client.models.generate_content(
                    model=config.MODEL_NAME,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt.replace("MULTIPLE sentences", "the following sentence").replace("###SENTENCE_N###", ""),
                        temperature=0,
                        max_output_tokens=config.MAX_NEW_TOKENS,
                        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                    ),
                )
                if response.text:
                    return response.text.strip()
                await asyncio.sleep(config.RETRY_BACKOFF * (2 ** attempt))
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    await asyncio.sleep(config.RETRY_BACKOFF * (2 ** attempt) * 5)
                else:
                    if attempt == config.RETRY_MAX - 1:
                        return "[]"
                    await asyncio.sleep(config.RETRY_BACKOFF * (2 ** attempt))
        return "[]"

async def run_experiment(strategy):
    print(f"\n{'=' * 70}\n{strategy.upper()} | {config.MODEL_NAME}\n{'=' * 70}")
    output_dir = Path(config.OUTPUT_DIR) / strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    train, _, test = load_split_data(strategy, config.SPLITS_DIR)
    if config.MAX_TEST_SAMPLES:
        test = test[:config.MAX_TEST_SAMPLES]

    sys_prompt = build_batched_system_prompt()
    seed_results = []

    for seed in config.MC_SEEDS:
        print(f"\n--- seed={seed} ---")
        random.seed(seed)

        ckpt = output_dir / f'checkpoint_seed{seed}.json'
        done_indices = set()
        saved_preds = {}
        if ckpt.exists():
            with open(ckpt, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            for i, p in enumerate(saved):
                if p is not None:
                    saved_preds[i] = p
                    done_indices.add(i)
            print(f"  resuming: {len(done_indices)} samples done")

        examples = select_diverse_examples(train, config.NUM_FEW_SHOT_EXAMPLES)
        with open(output_dir / f'few_shot_examples_seed{seed}.json', 'w', encoding='utf-8') as f:
            json.dump([{'sentence': ' '.join(it['words']), 'entities': en} for it, en in examples], f, ensure_ascii=False, indent=2)

        all_true = [extract_entities_from_labels(it['words'], it['labels']) for it in test]

        remaining = [i for i in range(len(test)) if i not in done_indices]
        batches = [remaining[i:i+config.BATCH_SIZE] for i in range(0, len(remaining), config.BATCH_SIZE)]

        print(f"  {len(remaining)} samples remaining in {len(batches)} batches")

        batch_prompts = []
        for batch_indices in batches:
            batch_items = [test[i] for i in batch_indices]
            prompt = build_batched_few_shot_prompt(examples, batch_items)
            batch_prompts.append(prompt)

        sem = asyncio.Semaphore(config.MAX_CONCURRENT)
        tasks = [call_batch_async(sys_prompt, batch_prompts[bi], sem, bi, len(batches), len(batches[bi]))
                 for bi in range(len(batches))]

        t0 = time.time()
        raw_batch_outputs = {}
        pbar = tqdm(total=len(tasks), desc=f"Seed {seed}")
        for coro in asyncio.as_completed(tasks):
            bi, out = await coro
            raw_batch_outputs[bi] = out
            pbar.update(1)
        pbar.close()

        failed_indices = []
        for bi, batch_indices in enumerate(batches):
            raw = raw_batch_outputs.get(bi, "")
            if not raw:
                failed_indices.extend(batch_indices)
                continue

            parsed = parse_batched_output(raw, len(batch_indices))
            for j, sample_idx in enumerate(batch_indices):
                if parsed[j] is not None and parsed[j] != []:
                    saved_preds[sample_idx] = {'pred_entities': parsed[j], 'raw_output': raw}
                elif parsed[j] == []:
                    saved_preds[sample_idx] = {'pred_entities': [], 'raw_output': raw}

        if failed_indices:
            print(f"  retrying {len(failed_indices)} failed samples individually...")
            single_sys = """You are an expert Named Entity Recognition (NER) system for Arabic text.
Extract all named entities. Return ONLY a JSON array with "type" and "text" fields. If none found, return []."""

            retry_sem = asyncio.Semaphore(config.MAX_CONCURRENT)
            retry_tasks = []
            for si in failed_indices:
                prompt = f"Input: {' '.join(test[si]['words'])}\nOutput:"
                retry_tasks.append((si, call_single_async(single_sys, prompt, retry_sem)))

            for si, task in retry_tasks:
                result = await task
                saved_preds[si] = {'pred_entities': parse_single_json(result), 'raw_output': result}

        elapsed = time.time() - t0
        print(f"  done in {elapsed/60:.2f}min")

        predictions = []
        all_pred = []
        for i in range(len(test)):
            true_ents = all_true[i]
            if i in saved_preds:
                pe = saved_preds[i].get('pred_entities', [])
                ro = saved_preds[i].get('raw_output', '')
            else:
                pe, ro = [], ''
            all_pred.append(pe)
            predictions.append({
                'sentence': ' '.join(test[i]['words']),
                'true_entities': true_ents,
                'pred_entities': pe,
                'raw_output': ro
            })

        with open(ckpt, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        metrics = LLMNERMetrics.compute_metrics(all_true, all_pred)
        save_classification_report(metrics, output_dir / f'report_seed{seed}.txt', config.MODEL_NAME)

        with open(output_dir / f'predictions_seed{seed}.json', 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        if ckpt.exists():
            ckpt.unlink()

        sr = {'seed': seed, 'model': config.MODEL_NAME, 'batch_size': config.BATCH_SIZE,
              'num_test': len(test), 'time_min': elapsed/60, 'metrics': metrics}
        seed_results.append(sr)

        with open(output_dir / f'results_seed{seed}.json', 'w', encoding='utf-8') as f:
            json.dump(sr, f, ensure_ascii=False, indent=2)

        print(f"  seed {seed}: micro f1={metrics['micro']['f1']:.4f}  macro f1={metrics['macro']['f1']:.4f}")

    avg = compute_averaged_metrics(seed_results)
    save_averaged_report(seed_results, output_dir / 'averaged_report.txt',
                         seeds=config.MC_SEEDS, model_name=config.MODEL_NAME)

    final = {'strategy': strategy, 'model': config.MODEL_NAME, 'batch_size': config.BATCH_SIZE,
             'seeds': config.MC_SEEDS, 'per_seed': seed_results, 'averaged': avg}
    with open(output_dir / 'all_seeds_results.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    return final


async def main():
    print("=" * 70)
    print(f"gemini 3 pro batched ner | seeds: {config.MC_SEEDS}")
    print("=" * 70)

    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    all_results = {}

    for strategy in config.STRATEGIES:
        try:
            all_results[strategy] = await run_experiment(strategy)
        except Exception as e:
            print(f"error in {strategy}: {e}")
            import traceback
            traceback.print_exc()

    with open(Path(config.OUTPUT_DIR) / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}\nfinal summary\n{'=' * 70}")
    print(f"{'Strategy':<20} {'Micro-F1':>25} {'Macro-F1':>25}")
    print("-" * 70)
    for s, r in all_results.items():
        a = r['averaged']
        print(f"{s:<20} {a['micro']['f1_mean']:.4f}±{a['micro']['f1_std']:.4f}        "
              f"{a['macro']['f1_mean']:.4f}±{a['macro']['f1_std']:.4f}")
    print("=" * 70)
    print(f"\nresults saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
