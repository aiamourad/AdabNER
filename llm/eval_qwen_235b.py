import json
import random
import time
import asyncio
from pathlib import Path
from tqdm import tqdm

from google import genai
from google.genai import types

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_utils import (
    load_split_data, extract_entities_from_labels,
    select_diverse_examples, build_system_prompt, build_few_shot_prompt,
    parse_model_output, LLMNERMetrics,
)
from src.metrics import save_classification_report, save_averaged_report, compute_averaged_metrics

class Config:
    MODEL_NAME = "qwen/qwen3-235b-a22b-instruct-2507-maas"

    SPLITS_DIR = 'data/adabner'
    OUTPUT_DIR = 'results/llm/qwen_235b'
    STRATEGIES = ['stratified_book', 'leave_book_out']
    NUM_FEW_SHOT_EXAMPLES = 5
    MAX_NEW_TOKENS = 2048
    MAX_TEST_SAMPLES = None

    MC_SEEDS = [42, 1, 123]

    MAX_CONCURRENT = 5
    RETRY_MAX = 5
    RETRY_BACKOFF = 2


config = Config()

client = genai.Client(
    vertexai=True,
    project='',
    location=''
)
async_client = client.aio


async def call_gemini_async(system_prompt, user_prompt, semaphore, idx, total):
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
                    return idx, response.text.strip()
                else:
                    await asyncio.sleep(config.RETRY_BACKOFF * (2 ** attempt))
                    continue

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = config.RETRY_BACKOFF * (2 ** attempt) * 5
                    print(f"  [{idx+1}/{total}] rate limited, waiting {wait:.0f}s...")
                    await asyncio.sleep(wait)
                elif "500" in err_str or "503" in err_str:
                    wait = config.RETRY_BACKOFF * (2 ** attempt)
                    await asyncio.sleep(wait)
                else:
                    print(f"  [{idx+1}/{total}] error: {e}")
                    if attempt == config.RETRY_MAX - 1:
                        return idx, "[]"
                    await asyncio.sleep(config.RETRY_BACKOFF * (2 ** attempt))

        return idx, "[]"


async def run_batch_inference(system_prompt, user_prompts, start_idx=0):
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)
    total = len(user_prompts)

    tasks = []
    for i in range(start_idx, total):
        task = call_gemini_async(system_prompt, user_prompts[i], semaphore, i, total)
        tasks.append(task)

    results = {}
    pbar = tqdm(total=len(tasks), desc="Inference", initial=0)

    for coro in asyncio.as_completed(tasks):
        idx, output = await coro
        results[idx] = output
        pbar.update(1)

    pbar.close()
    return results

async def run_experiment(strategy):
    print(f"\n{'=' * 70}")
    print(f"running experiment: {strategy.upper()}")
    print(f"seeds: {config.MC_SEEDS}")
    print(f"concurrent requests: {config.MAX_CONCURRENT}")
    print(f"{'=' * 70}")

    output_dir = Path(config.OUTPUT_DIR) / strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, test_data = load_split_data(strategy, config.SPLITS_DIR)

    if config.MAX_TEST_SAMPLES:
        test_data = test_data[:config.MAX_TEST_SAMPLES]

    system_prompt = build_system_prompt()

    seed_results = []

    for seed in config.MC_SEEDS:
        print(f"\n--- running with seed={seed} ---")
        random.seed(seed)

        checkpoint_path = output_dir / f'checkpoint_seed{seed}.json'
        existing_results = {}
        start_idx = 0

        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            existing_results = {i: s for i, s in enumerate(saved)}
            start_idx = len(saved)
            print(f"  resuming from checkpoint: {start_idx} samples already done")

        few_shot_examples = select_diverse_examples(train_data, config.NUM_FEW_SHOT_EXAMPLES)

        examples_info = [{'sentence': ' '.join(item['words']), 'entities': entities} for item, entities in few_shot_examples]
        with open(output_dir / f'few_shot_examples_seed{seed}.json', 'w', encoding='utf-8') as f:
            json.dump(examples_info, f, ensure_ascii=False, indent=2)

        all_true_entities = []
        all_user_prompts = []

        for item in test_data:
            true_entities = extract_entities_from_labels(item['words'], item['labels'])
            all_true_entities.append(true_entities)
            user_prompt = build_few_shot_prompt(few_shot_examples, item)
            all_user_prompts.append(user_prompt)

        print(f"running async inference on {len(test_data)} samples (starting from {start_idx})...")
        start_time = time.time()

        new_results = await run_batch_inference(system_prompt, all_user_prompts, start_idx)

        all_raw_outputs = {}
        for i in range(start_idx):
            if i in existing_results:
                all_raw_outputs[i] = existing_results[i].get('raw_output', '[]')
        all_raw_outputs.update(new_results)

        elapsed = time.time() - start_time
        print(f"inference completed in {elapsed / 60:.2f} minutes")
        if elapsed > 0:
            print(f"throughput: {(len(test_data) - start_idx) / elapsed:.2f} samples/second")

        predictions = []
        all_pred_entities = []

        for i in range(len(test_data)):
            item = test_data[i]
            true_ents = all_true_entities[i]

            if i in existing_results and isinstance(existing_results[i], dict):
                pred_entities = existing_results[i].get('pred_entities', [])
                raw_output = existing_results[i].get('raw_output', '[]')
            else:
                raw_output = all_raw_outputs.get(i, '[]')
                pred_entities = parse_model_output(raw_output)

            all_pred_entities.append(pred_entities)
            predictions.append({
                'sentence': ' '.join(item['words']),
                'true_entities': true_ents,
                'pred_entities': pred_entities,
                'raw_output': raw_output
            })

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        metrics = LLMNERMetrics.compute_metrics(all_true_entities, all_pred_entities)

        save_classification_report(metrics, output_dir / f'classification_report_seed{seed}_qwen.txt')

        with open(output_dir / f'predictions_seed{seed}.json', 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        seed_result = {
            'seed': seed,
            'model': config.MODEL_NAME,
            'num_test_samples': len(test_data),
            'inference_time_minutes': elapsed / 60,
            'metrics': metrics
        }
        seed_results.append(seed_result)

        with open(output_dir / f'results_seed{seed}.json', 'w', encoding='utf-8') as f:
            json.dump(seed_result, f, ensure_ascii=False, indent=2)

        print(f"  seed {seed}: micro f1={metrics['micro']['f1']:.4f}  macro f1={metrics['macro']['f1']:.4f}")

    avg_metrics = compute_averaged_metrics(seed_results)
    save_averaged_report(
        seed_results,
        output_dir / 'averaged_classification_report.txt',
        seeds=config.MC_SEEDS, model_name=config.MODEL_NAME
    )

    final_results = {
        'strategy': strategy,
        'model': config.MODEL_NAME,
        'num_few_shot': config.NUM_FEW_SHOT_EXAMPLES,
        'seeds': config.MC_SEEDS,
        'per_seed_results': seed_results,
        'averaged': avg_metrics
    }

    with open(output_dir / 'all_seeds_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"averaged results: {strategy.upper()}")
    print(f"{'=' * 70}")
    print(f"{'Metric':<20} {'Precision':>20} {'Recall':>20} {'F1':>20}")
    print("-" * 70)
    for metric_type in ['micro', 'macro']:
        m = avg_metrics[metric_type]
        print(f"{metric_type.capitalize():<20} {m['precision_mean']:.4f}±{m['precision_std']:.4f}      "
              f"{m['recall_mean']:.4f}±{m['recall_std']:.4f}      {m['f1_mean']:.4f}±{m['f1_std']:.4f}")
    print(f"{'=' * 70}")

    return final_results


async def main():
    print("=" * 70)
    print(f"qwen 235b few-shot ner (async) | {config.MODEL_NAME}")
    print(f"seeds: {config.MC_SEEDS} | concurrency: {config.MAX_CONCURRENT}")
    print("=" * 70)

    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for strategy in config.STRATEGIES:
        try:
            results = await run_experiment(strategy)
            all_results[strategy] = results
        except Exception as e:
            print(f"error in {strategy}: {e}")
            import traceback
            traceback.print_exc()

    with open(Path(config.OUTPUT_DIR) / 'all_results_qwen.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("final summary")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Micro-F1':>25} {'Macro-F1':>25}")
    print("-" * 70)

    for strategy, res in all_results.items():
        avg = res['averaged']
        micro = avg['micro']
        macro = avg['macro']
        print(f"{strategy:<20} {micro['f1_mean']:.4f}±{micro['f1_std']:.4f}        "
              f"{macro['f1_mean']:.4f}±{macro['f1_std']:.4f}")

    print("=" * 70)
    print(f"\nresults saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
