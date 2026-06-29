import json
import random
import time
from pathlib import Path
from tqdm import tqdm

import transformers
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    @property
    def _all_special_tokens_extended(self):
        return self.all_special_tokens
    PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended
    print("patched tokenizer for vllm compatibility")

from vllm import LLM, SamplingParams

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_utils import (
    load_split_data, extract_entities_from_labels,
    select_diverse_examples, build_system_prompt, build_few_shot_prompt,
    parse_model_output, LLMNERMetrics,
)
from src.metrics import save_classification_report, save_averaged_report, compute_averaged_metrics

class Config:
    MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"
    SPLITS_DIR = 'data/adabner'
    OUTPUT_DIR = 'results/llm/qwen_72b'
    STRATEGIES = ['stratified_book', 'leave_book_out']
    NUM_FEW_SHOT_EXAMPLES = 5
    MAX_NEW_TOKENS = 2048
    MAX_TEST_SAMPLES = None

    MC_SEEDS = [42, 1, 123]

    TENSOR_PARALLEL_SIZE = 8
    GPU_MEMORY_UTILIZATION = 0.90


config = Config()


def build_full_prompt(system_prompt, user_prompt):
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""


def load_model():
    print(f"loading model with vllm (tensor_parallel={config.TENSOR_PARALLEL_SIZE})...")

    llm = LLM(
        model=config.MODEL_PATH,
        tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        max_model_len=8192,
        dtype="auto",
        enforce_eager=True,
    )

    print("model loaded successfully!")
    return llm

def run_experiment(llm, strategy):
    print(f"\n{'=' * 70}")
    print(f"running experiment: {strategy.upper()}")
    print(f"seeds: {config.MC_SEEDS}")
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

        few_shot_examples = select_diverse_examples(train_data, config.NUM_FEW_SHOT_EXAMPLES)

        examples_info = [{'sentence': ' '.join(item['words']), 'entities': entities} for item, entities in few_shot_examples]
        with open(output_dir / f'few_shot_examples_seed{seed}.json', 'w', encoding='utf-8') as f:
            json.dump(examples_info, f, ensure_ascii=False, indent=2)

        all_prompts = []
        all_true_entities = []

        for item in tqdm(test_data, desc="Building prompts"):
            true_entities = extract_entities_from_labels(item['words'], item['labels'])
            all_true_entities.append(true_entities)

            user_prompt = build_few_shot_prompt(few_shot_examples, item)
            full_prompt = build_full_prompt(system_prompt, user_prompt)
            all_prompts.append(full_prompt)

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=config.MAX_NEW_TOKENS,
            stop=["<|im_end|>", "<|im_start|>"]
        )

        print(f"running vllm inference on {len(all_prompts)} samples...")
        start_time = time.time()

        outputs = llm.generate(all_prompts, sampling_params)

        elapsed = time.time() - start_time
        print(f"inference completed in {elapsed / 60:.2f} minutes")
        print(f"throughput: {len(test_data) / elapsed:.2f} samples/second")

        all_pred_entities = []
        predictions = []

        for i, (output, item, true_ents) in enumerate(zip(outputs, test_data, all_true_entities)):
            raw_output = output.outputs[0].text.strip()
            pred_entities = parse_model_output(raw_output)
            all_pred_entities.append(pred_entities)

            predictions.append({
                'sentence': ' '.join(item['words']),
                'true_entities': true_ents,
                'pred_entities': pred_entities,
                'raw_output': raw_output
            })

        metrics = LLMNERMetrics.compute_metrics(all_true_entities, all_pred_entities)

        save_classification_report(metrics, output_dir / f'classification_report_seed{seed}.txt')

        with open(output_dir / f'predictions_seed{seed}.json', 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        seed_result = {
            'seed': seed,
            'num_test_samples': len(test_data),
            'inference_time_minutes': elapsed / 60,
            'samples_per_second': len(test_data) / elapsed,
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
        seeds=config.MC_SEEDS
    )

    final_results = {
        'strategy': strategy,
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


def main():
    print("=" * 70)
    print("qwen 2.5 72b few-shot ner (vllm)")
    print(f"monte carlo seeds: {config.MC_SEEDS}")
    print("=" * 70)

    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    llm = load_model()

    all_results = {}

    for strategy in config.STRATEGIES:
        try:
            results = run_experiment(llm, strategy)
            all_results[strategy] = results
        except Exception as e:
            print(f"error in {strategy}: {e}")
            import traceback
            traceback.print_exc()

    with open(Path(config.OUTPUT_DIR) / 'all_results.json', 'w', encoding='utf-8') as f:
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
    main()
