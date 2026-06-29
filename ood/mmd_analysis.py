import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


class Config:
    MODEL_PATH = 'aubmindlab/bert-base-arabertv02'
    
    WOJOOD_TRAIN_PATH = '/data/wojood/train.csv'
    WOJOOD_VAL_PATH = '/data/wojood/val.csv'
    WOJOOD_TEST_PATH = '/data/wojood/test.csv'
    MERGED_PATH = '/data/adabner/merged_up_to_date_sent_id.csv'
    
    OUTPUT_DIR = 'mmd_analysis_results'
    
    FOCUS_TYPES = [
        'CURR', 'GPE', 'PERS', 'ORG', 'OCC',
        'DATE', 'TIME', 'UNIT', 'QUANTITY', 'CARDINAL', 'ORDINAL',
        'EVENT', 'FAC', 'LOC', 'LAW', 'LANGUAGE', 'NORP',
        'MONEY', 'PERCENT', 'PRODUCT', 'WEBSITE'
    ]
    
    MAX_SAMPLES = 500
    BATCH_SIZE = 32

config = Config()
Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_nested_tags(tag_string, entity_types):
    result = {et: 'O' for et in entity_types}
    
    if pd.isna(tag_string) or str(tag_string).strip() == '' or tag_string == 'O':
        return result
    
    tags = str(tag_string).split()
    for tag in tags:
        if tag == 'O':
            continue
        if tag.startswith('B-') or tag.startswith('I-'):
            prefix = tag[0]  # 'B' or 'I'
            ent_type = tag[2:]  # Remove 'B-' or 'I-'
            if '_' in ent_type:
                ent_type = ent_type.split('_')[0]
            if ent_type in entity_types:
                result[ent_type] = prefix
    
    return result

def load_and_parse_data(file_paths, tag_col, token_col, sent_id_col, pos_col, source_name):
    print(f"loading {source_name}...")
    
    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  - loaded {path}: {len(df)} rows")
        except FileNotFoundError:
            print(f"  - warning: {path} not found")
    
    if not dfs:
        raise FileNotFoundError(f"No files found for {source_name}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  - total rows: {len(combined_df)}")
    
    print(f"  - parsing nested tags into separate columns...")
    parsed_tags = combined_df[tag_col].apply(lambda x: parse_nested_tags(x, config.FOCUS_TYPES))
    
    for et in config.FOCUS_TYPES:
        combined_df[f'tag_{et}'] = parsed_tags.apply(lambda x: x[et])
    
    entity_counts = {}
    print(f"\n  entity counts for {source_name}:")
    for et in config.FOCUS_TYPES:
        count = (combined_df[f'tag_{et}'] == 'B').sum()
        entity_counts[et] = count
        if count > 0:
            print(f"    {et}: {count}")
    
    sentences = []
    for sent_id, grp in combined_df.groupby(sent_id_col):
        if pos_col in grp.columns:
            grp = grp.sort_values(pos_col)
        
        sent_data = {
            'words': grp[token_col].astype(str).tolist(),
            'sent_id': sent_id
        }
        
        for et in config.FOCUS_TYPES:
            sent_data[f'tag_{et}'] = grp[f'tag_{et}'].tolist()
        
        sentences.append(sent_data)
    
    print(f"  - total sentences: {len(sentences)}")
    return sentences, entity_counts

def load_wojood_data():
    return load_and_parse_data(
        file_paths=[config.WOJOOD_TRAIN_PATH, config.WOJOOD_VAL_PATH, config.WOJOOD_TEST_PATH],
        tag_col='Level1_tags',
        token_col='token',
        sent_id_col='global_sentence_id',
        pos_col='word_position',
        source_name='Wojood'
    )

def load_merged_data():
    return load_and_parse_data(
        file_paths=[config.MERGED_PATH],
        tag_col='ner_tag',
        token_col='token',
        sent_id_col='sent_id',
        pos_col='word_pos',
        source_name='Merged'
    )

def extract_entity_spans(tags):
    """
    Given a list of 'B'/'I'/'O' tags for ONE entity type,
    return list of (start_idx, end_idx) for each entity span.
    """
    spans = []
    start = None
    
    for i, tag in enumerate(tags):
        if tag == 'B':
            if start is not None:
                spans.append((start, i)) 
            start = i
        elif tag == 'I':
            if start is None:
                start = i 
        else:  # 'O'
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(tags)))
    
    return spans


def extract_entity_embeddings(model, tokenizer, dataset, source_name):
    model.eval()
    embeddings_by_type = {et: [] for et in config.FOCUS_TYPES}

    total_entities_expected = {et: 0 for et in config.FOCUS_TYPES}
    total_entities_extracted = {et: 0 for et in config.FOCUS_TYPES}
    truncated_count = 0
    
    print(f"\nextracting embeddings for {source_name}...")
    
    with torch.no_grad():
        for item in tqdm(dataset):
            words = item['words']
            
            has_entity = False
            for et in config.FOCUS_TYPES:
                if 'B' in item[f'tag_{et}']:
                    has_entity = True
                    break
            
            if not has_entity:
                continue
            
            enc = tokenizer(words, is_split_into_words=True, return_tensors='pt',
                           truncation=True, padding=True, max_length=512)
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            

            outputs = model(input_ids, attention_mask)
            last_hidden = outputs.last_hidden_state[0]  # (seq_len, 768)
            
            word_ids = enc.word_ids(0)
            
            # Check for truncation
            max_word_id = max([w for w in word_ids if w is not None], default=0)
            if max_word_id < len(words) - 1:
                truncated_count += 1
            
            word_to_subwords = {}
            for subword_idx, word_idx in enumerate(word_ids):
                if word_idx is not None:
                    if word_idx not in word_to_subwords:
                        word_to_subwords[word_idx] = []
                    word_to_subwords[word_idx].append(subword_idx)
            
            for et in config.FOCUS_TYPES:
                tags = item[f'tag_{et}']
                spans = extract_entity_spans(tags)
                total_entities_expected[et] += len(spans)
                
                for start, end in spans:
                    if start > max_word_id:
                        continue  
                    
                    effective_end = min(end, max_word_id + 1)
                    
                    span_embeddings = []
                    for word_idx in range(start, effective_end):
                        if word_idx in word_to_subwords:
                            for subword_idx in word_to_subwords[word_idx]:
                                span_embeddings.append(last_hidden[subword_idx])
                    
                    if span_embeddings:
                        avg_vec = torch.stack(span_embeddings).mean(dim=0).cpu().numpy()
                        embeddings_by_type[et].append(avg_vec)
                        total_entities_extracted[et] += 1
    
    print(f"\n  extracted entity embeddings for {source_name}:")
    print(f"  (truncated sentences: {truncated_count})")
    for et in config.FOCUS_TYPES:
        expected = total_entities_expected[et]
        extracted = total_entities_extracted[et]
        if expected > 0:
            pct = (extracted / expected) * 100
            print(f"    {et}: {extracted}/{expected} ({pct:.1f}%)")
    
    return embeddings_by_type

## Compute MMD
def compute_mmd(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.nan
    
    xx = rbf_kernel(x, x)
    yy = rbf_kernel(y, y)
    xy = rbf_kernel(x, y)
    
    return xx.mean() + yy.mean() - 2 * xy.mean()

def visualize_domain_shift(wojood_emb, merged_emb, wojood_counts, merged_counts):
    results = []
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    for ent_type in config.FOCUS_TYPES:
        vecs_w = np.array(wojood_emb[ent_type]) if wojood_emb[ent_type] else np.array([])
        vecs_m = np.array(merged_emb[ent_type]) if merged_emb[ent_type] else np.array([])
        
        count_w_actual = wojood_counts[ent_type]
        count_m_actual = merged_counts[ent_type]
        
        count_w_emb = len(vecs_w)
        count_m_emb = len(vecs_m)
        
        if count_w_emb < 10 or count_m_emb < 10:
            print(f"skipping {ent_type}: not enough embeddings (wojood:{count_w_emb}/{count_w_actual}, merged:{count_m_emb}/{count_m_actual})")
            results.append({
                'Entity': ent_type,
                'Wojood_Count': count_w_actual,
                'Merged_Count': count_m_actual,
                'Wojood_Embeddings': count_w_emb,
                'Merged_Embeddings': count_m_emb,
                'MMD_Distance': np.nan
            })
            continue
        
        print(f"\nprocessing {ent_type} (wojood:{count_w_emb}/{count_w_actual}, merged:{count_m_emb}/{count_m_actual})...")
        
        sample_size = min(1000, count_w_emb, count_m_emb)
        idx_w = np.random.choice(count_w_emb, sample_size, replace=False)
        idx_m = np.random.choice(count_m_emb, sample_size, replace=False)
        
        mmd_score = compute_mmd(vecs_w[idx_w], vecs_m[idx_m])
        print(f"  > mmd score: {mmd_score:.4f}")
        
        results.append({
            'Entity': ent_type,
            'Wojood_Count': count_w_actual,
            'Merged_Count': count_m_actual,
            'Wojood_Embeddings': count_w_emb,
            'Merged_Embeddings': count_m_emb,
            'MMD_Distance': mmd_score
        })
        
        plot_n = min(config.MAX_SAMPLES, count_w_emb, count_m_emb)
        idx_w_plot = np.random.choice(count_w_emb, plot_n, replace=False)
        idx_m_plot = np.random.choice(count_m_emb, plot_n, replace=False)
        
        subset_w = vecs_w[idx_w_plot]
        subset_m = vecs_m[idx_m_plot]
        
        combined = np.vstack([subset_w, subset_m])
        labels = ['Wojood'] * len(subset_w) + ['Merged'] * len(subset_m)
        
        print("  > running t-sne...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(combined)
        
        df_plot = pd.DataFrame({
            'x': emb_2d[:, 0],
            'y': emb_2d[:, 1],
            'Domain': labels
        })
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x='x', y='y', hue='Domain',
                       palette={'Wojood': '#d9534f', 'Merged': '#5bc0de'},
                       alpha=0.7, s=40)
        
        plt.title(f"Domain Shift for Entity: {ent_type}\nMMD Distance: {mmd_score:.3f}", fontsize=14)
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.legend()
        
        out_file = f"{config.OUTPUT_DIR}/tsne_{ent_type}.png"
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
        print(f"  > saved plot to {out_file}")
    
    return pd.DataFrame(results)


def main():
    print("="*60)
    print("mmd analysis - wojood vs merged (full entity spans)")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    model = AutoModel.from_pretrained(config.MODEL_PATH).to(device)
    
    wojood_data, wojood_counts = load_wojood_data()
    merged_data, merged_counts = load_merged_data()
    
    emb_wojood = extract_entity_embeddings(model, tokenizer, wojood_data, "Wojood")
    emb_merged = extract_entity_embeddings(model, tokenizer, merged_data, "AdabNER Merged")
    
    df_stats = visualize_domain_shift(emb_wojood, emb_merged, wojood_counts, merged_counts)
    
    csv_path = f"{config.OUTPUT_DIR}/mmd_scores.csv"
    df_stats.to_csv(csv_path, index=False)
    
    print("\n" + "="*60)
    print("analysis complete")
    print("="*60)
    print(f"\nresults saved to {config.OUTPUT_DIR}")
    print("\nmmd scores (higher = larger domain shift):")
    print(df_stats.to_string(index=False))
    
    valid_mmd = df_stats[df_stats['MMD_Distance'].notna()]
    if not valid_mmd.empty:
        print(f"\n--- summary ---")
        print(f"entities analyzed: {len(valid_mmd)}/{len(config.FOCUS_TYPES)}")
        print(f"mean mmd: {valid_mmd['MMD_Distance'].mean():.4f}")
        print(f"max mmd: {valid_mmd['MMD_Distance'].max():.4f} ({valid_mmd.loc[valid_mmd['MMD_Distance'].idxmax(), 'Entity']})")
        print(f"min mmd: {valid_mmd['MMD_Distance'].min():.4f} ({valid_mmd.loc[valid_mmd['MMD_Distance'].idxmin(), 'Entity']})")

if __name__ == "__main__":
    main()