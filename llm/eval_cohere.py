import json
import gc
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_utils import (
    ENTITY_TYPES, ENTITY_DESCRIPTIONS,
    load_split_data, extract_entities_from_labels,
    select_diverse_examples, build_system_prompt, build_few_shot_prompt,
    parse_model_output, LLMNERMetrics,
)
from src.metrics import save_classification_report, save_averaged_report, compute_averaged_metrics


def patch_cohere_tokenizer():
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
            @property
            def all_special_tokens_extended(self):
                """Returns all special tokens including additional ones."""
                return self.all_special_tokens

            PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended
    except Exception:
        pass

    try:
        from transformers.models.cohere.tokenization_cohere_fast import CohereTokenizerFast
        if not hasattr(CohereTokenizerFast, 'all_special_tokens_extended'):
            @property
            def all_special_tokens_extended(self):
                return self.all_special_tokens
            CohereTokenizerFast.all_special_tokens_extended = all_special_tokens_extended
    except ImportError:
        pass

    try:
        from transformers.models.cohere.tokenization_cohere import CohereTokenizer
        if not hasattr(CohereTokenizer, 'all_special_tokens_extended'):
            @property
            def all_special_tokens_extended(self):
                return self.all_special_tokens
            CohereTokenizer.all_special_tokens_extended = all_special_tokens_extended
    except ImportError:
        pass


def patch_cohere_config():
    try:
        from transformers.models.cohere.configuration_cohere import CohereConfig

        _orig_init = CohereConfig.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            if not hasattr(self, 'rope_theta'):
                self.rope_theta = 4000000.0

        CohereConfig.__init__ = _patched_init

    except Exception:
        pass


patch_cohere_tokenizer()
patch_cohere_config()

class Config:
    MODELS = [
        {
            "name": "CohereLabs/aya-expanse-32b",
            "short_name": "aya-expanse-32b",
            "tensor_parallel_size": 4,
            "max_model_len": 4096,
            "chat_template": "aya",
        },
    ]

    SPLITS_DIR = 'data/adabner'
    OUTPUT_DIR = 'results/llm/cohere'
    STRATEGIES = ['stratified_book', 'leave_book_out']
    NUM_FEW_SHOT_EXAMPLES = 5
    MAX_NEW_TOKENS = 512
    MAX_TEST_SAMPLES = None

    MC_SEEDS = [42, 1, 123]

    GPU_MEMORY_UTILIZATION = 0.90


config = Config()

def build_chat_prompt(system_prompt, user_prompt, chat_template):
    if chat_template == "cohere":
        return f"""<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""
    elif chat_template == "aya":
        return f"""<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""
    else:
        return f"""System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"""


def cleanup_model(llm):
    if llm is not None:
        try:
            destroy_model_parallel()
        except:
            pass
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)


def load_model_vllm(model_config):
    model_path = model_config["name"]
    tp_size = model_config["tensor_parallel_size"]
    max_len = model_config["max_model_len"]

    print(f"\nloading {model_config['short_name']}...")

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_len,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        enforce_eager=False,
        enable_prefix_caching=True,
    )

    print(f"{model_config['short_name']} loaded")
    return llm

def generate_batch_vllm(llm, prompts, max_new_tokens=512):
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        top_p=1.0,
        stop=["<|END_OF_TURN_TOKEN|>", "<|START_OF_TURN_TOKEN|>"],
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        responses.append(generated_text)

    return responses

def run_model_experiment(model_config, strategy, train_data, test_data):
    model_name = model_config["short_name"]
    chat_template = model_config["chat_template"]

    print(f"\n{'=' * 70}")
    print(f"model: {model_name} | strategy: {strategy.upper()}")
    print(f"seeds: {config.MC_SEEDS}")
    print(f"{'=' * 70}")

    output_dir = Path(config.OUTPUT_DIR) / strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = load_model_vllm(model_config)

    system_prompt = build_system_prompt()
    seed_results = []

    for seed in config.MC_SEEDS:
        print(f"\n--- running with seed={seed} ---")
        import random
        random.seed(seed)

        few_shot_examples = select_diverse_examples(train_data, config.NUM_FEW_SHOT_EXAMPLES)

        examples_info = [{'sentence': ' '.join(item['words']), 'entities': entities} for item, entities in few_shot_examples]
        with open(output_dir / f'few_shot_examples_seed{seed}.json', 'w', encoding='utf-8') as f:
            json.dump(examples_info, f, ensure_ascii=False, indent=2)

        all_prompts = []
        all_true_entities = []

        for item in test_data:
            true_entities = extract_entities_from_labels(item['words'], item['labels'])
            all_true_entities.append(true_entities)

            user_prompt = build_few_shot_prompt(few_shot_examples, item)
            full_prompt = build_chat_prompt(system_prompt, user_prompt, chat_template)
            all_prompts.append(full_prompt)

        print(f"running vllm inference on {len(all_prompts)} prompts...")
        start_time = time.time()

        all_raw_outputs = generate_batch_vllm(llm, all_prompts, config.MAX_NEW_TOKENS)

        elapsed = time.time() - start_time
        print(f"inference completed in {elapsed / 60:.2f} minutes")
        print(f"throughput: {len(test_data) / elapsed:.2f} samples/second")

        all_pred_entities = []
        predictions = []

        for i, (item, true_ents, raw_output) in enumerate(zip(test_data, all_true_entities, all_raw_outputs)):
            pred_entities = parse_model_output(raw_output)
            all_pred_entities.append(pred_entities)

            predictions.append({
                'index': i,
                'sentence': ' '.join(item['words']),
                'words': item['words'],
                'true_labels': item['labels'],
                'true_entities': true_ents,
                'pred_entities': pred_entities,
                'raw_output': raw_output,
            })

        metrics = LLMNERMetrics.compute_metrics(all_true_entities, all_pred_entities)

        save_classification_report(metrics, output_dir / f'classification_report_seed{seed}.txt', model_name)

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

    cleanup_model(llm)

    avg_metrics = compute_averaged_metrics(seed_results)
    save_averaged_report(
        seed_results,
        output_dir / 'averaged_classification_report.txt',
        seeds=config.MC_SEEDS, model_name=model_name
    )

    final_results = {
        'model': model_config['name'],
        'model_short_name': model_name,
        'strategy': strategy,
        'num_few_shot': config.NUM_FEW_SHOT_EXAMPLES,
        'seeds': config.MC_SEEDS,
        'per_seed_results': seed_results,
        'averaged': avg_metrics
    }

    with open(output_dir / 'all_seeds_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"averaged results: {model_name} | {strategy.upper()}")
    print(f"{'=' * 70}")
    print(f"{'Metric':<20} {'Precision':>20} {'Recall':>20} {'F1':>20}")
    print("-" * 70)
    for metric_type in ['micro', 'macro']:
        m = avg_metrics[metric_type]
        print(f"{metric_type.capitalize():<20} {m['precision_mean']:.4f}±{m['precision_std']:.4f}      "
              f"{m['recall_mean']:.4f}±{m['recall_std']:.4f}      {m['f1_mean']:.4f}±{m['f1_std']:.4f}")
    print(f"{'=' * 70}")

    return final_results

def generate_comparison_report(all_results):
    report_path = Path(config.OUTPUT_DIR) / 'comparison_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MULTI-MODEL COMPARISON REPORT - FEW-SHOT NESTED NER\n")
        f.write(f"Models: {[m['short_name'] for m in config.MODELS]}\n")
        f.write(f"Strategies: {config.STRATEGIES}\n")
        f.write(f"Seeds: {config.MC_SEEDS}\n")
        f.write("=" * 70 + "\n\n")

        for strategy in config.STRATEGIES:
            f.write(f"\n{'=' * 70}\n")
            f.write(f"STRATEGY: {strategy.upper()}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"{'Model':<25} {'Micro-F1':>22} {'Macro-F1':>22}\n")
            f.write("-" * 70 + "\n")

            strategy_results = all_results.get(strategy, {})

            for model_config_item in config.MODELS:
                model_name = model_config_item['short_name']
                if model_name in strategy_results:
                    avg = strategy_results[model_name]['averaged']
                    micro = avg['micro']
                    macro = avg['macro']

                    f.write(f"{model_name:<25} "
                            f"{micro['f1_mean']:.4f}±{micro['f1_std']:.4f}    "
                            f"{macro['f1_mean']:.4f}±{macro['f1_std']:.4f}\n")
                else:
                    f.write(f"{model_name:<25} {'FAILED':>22} {'FAILED':>22}\n")

            f.write("-" * 70 + "\n")

        f.write("\n\n" + "=" * 70 + "\n")
        f.write("BEST MODEL PER STRATEGY (by Micro-F1)\n")
        f.write("=" * 70 + "\n\n")

        for strategy in config.STRATEGIES:
            strategy_results = all_results.get(strategy, {})
            best_model = None
            best_f1 = -1

            for model_name, results in strategy_results.items():
                f1 = results['averaged']['micro']['f1_mean']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name

            if best_model:
                f.write(f"{strategy}: {best_model} (Micro-F1: {best_f1:.4f})\n")

        f.write("\n" + "=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("final summary: all models and strategies")
    print("=" * 70)

    for strategy in config.STRATEGIES:
        print(f"\n  {strategy.upper()}")
        print(f"  {'Model':<25} {'Micro-F1':>22} {'Macro-F1':>22}")
        print("  " + "-" * 70)

        strategy_results = all_results.get(strategy, {})

        for model_config_item in config.MODELS:
            model_name = model_config_item['short_name']
            if model_name in strategy_results:
                avg = strategy_results[model_name]['averaged']
                print(f"{model_name:<25} "
                      f"{avg['micro']['f1_mean']:.4f}±{avg['micro']['f1_std']:.4f}    "
                      f"{avg['macro']['f1_mean']:.4f}±{avg['macro']['f1_std']:.4f}")

    print("\n" + "=" * 70)

def main():
    print("=" * 70)
    print("multi-model few-shot nested ner evaluation")
    print(f"models: {[m['short_name'] for m in config.MODELS]}")
    print(f"strategies: {config.STRATEGIES}")
    print(f"seeds: {config.MC_SEEDS}")
    print("=" * 70)

    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for strategy in config.STRATEGIES:
        print(f"\n{'=' * 70}")
        print(f"strategy: {strategy.upper()}")
        print(f"{'=' * 70}")

        train_data, val_data, test_data = load_split_data(strategy, config.SPLITS_DIR)

        if config.MAX_TEST_SAMPLES:
            test_data = test_data[:config.MAX_TEST_SAMPLES]

        all_results[strategy] = {}

        for model_config_item in config.MODELS:
            try:
                results = run_model_experiment(model_config_item, strategy, train_data, test_data)
                all_results[strategy][model_config_item['short_name']] = results
            except Exception as e:
                print(f"error with {model_config_item['short_name']} on {strategy}: {e}")
                import traceback
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()

    with open(Path(config.OUTPUT_DIR) / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    generate_comparison_report(all_results)

    print(f"\nresults saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
