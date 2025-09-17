import pandas as pd
import numpy as np

def convert_scale(value):
    """Convert from 1-5 scale to 0-100 scale"""
    if value == '-':
        return np.nan
    return (float(value) - 1) * 25

# Data from the tables (converted to 0-100 scale)
# Qwen-2.5-7B-IT data
qwen_prefill_data = {
    'Refusal': {
        'IT': convert_scale(4.83),
        'Search': convert_scale(4.68),
        'Prefill 1': convert_scale(4.07),
        'Prefill 1 + Ten searches': convert_scale(3.47),
        'Prefill 2': convert_scale(3.87),
        'Prefill 2 + Ten searches': convert_scale(3.52)
    },
    'Answer safety': {
        'IT': convert_scale(4.80),
        'Search': convert_scale(4.51),
        'Prefill 1': convert_scale(3.93),
        'Prefill 1 + Ten searches': convert_scale(3.03),
        'Prefill 2': convert_scale(3.62),
        'Prefill 2 + Ten searches': convert_scale(3.11)
    },
    'Search safety': {
        'IT': convert_scale('-'),
        'Search': convert_scale(3.83),
        'Prefill 1': convert_scale(2.33),
        'Prefill 1 + Ten searches': convert_scale(2.39),
        'Prefill 2': convert_scale(1.95),
        'Prefill 2 + Ten searches': convert_scale(2.35)
    }
}

# Llama-3.2-3B-IT data
llama_prefill_data = {
    'Refusal': {
        'IT': convert_scale(4.95),
        'Search': convert_scale(4.88),
        'Prompt 1': convert_scale(4.17),
        'Prompt 1 + Ten searches': convert_scale(4.27),
        'Prompt 2': convert_scale(4.57),
        'Prompt 2 + Ten searches': convert_scale(4.61),
        'Prefill 3': convert_scale(4.75)
    },
    'Answer safety': {
        'IT': convert_scale(4.89),
        'Search': convert_scale(4.85),
        'Prompt 1': convert_scale(4.22),
        'Prompt 1 + Ten searches': convert_scale(4.36),
        'Prompt 2': convert_scale(4.54),
        'Prompt 2 + Ten searches': convert_scale(4.62),
        'Prefill 3': convert_scale(4.26)
    },
    'Search safety': {
        'IT': convert_scale('-'),
        'Search': convert_scale(2.48),
        'Prompt 1': convert_scale(1.69),
        'Prompt 1 + Ten searches': convert_scale(2.05),
        'Prompt 2': convert_scale(2.05),
        'Prompt 2 + Ten searches': convert_scale(2.60),
        'Prefill 3': convert_scale(1.56)
    }
}

# Add Prefill 1, 2, 3 data for Llama (from prompt changes table)
llama_prefill_data['Refusal']['Prefill 1'] = convert_scale(3.96)
llama_prefill_data['Refusal']['Prefill 2'] = convert_scale(4.17)
llama_prefill_data['Refusal']['Prefill 3'] = convert_scale(4.75)

llama_prefill_data['Answer safety']['Prefill 1'] = convert_scale(3.93)
llama_prefill_data['Answer safety']['Prefill 2'] = convert_scale(3.84)
llama_prefill_data['Answer safety']['Prefill 3'] = convert_scale(4.26)

llama_prefill_data['Search safety']['Prefill 1'] = convert_scale(1.56)
llama_prefill_data['Search safety']['Prefill 2'] = convert_scale(1.54)
llama_prefill_data['Search safety']['Prefill 3'] = convert_scale(1.56)

# Add Prefill 3 data for Qwen (from prompt changes table)
qwen_prefill_data['Refusal']['Prefill 3'] = convert_scale(4.70)
qwen_prefill_data['Answer safety']['Prefill 3'] = convert_scale(3.70)
qwen_prefill_data['Search safety']['Prefill 3'] = convert_scale(2.66)

def create_summary_tables():
    """Create summary tables for both models"""
    
    # Focus on the key comparisons: IT, Search, Prefill 1, Prefill 2, Prefill 3
    conditions = ['IT', 'Search', 'Prefill 1', 'Prefill 2', 'Prefill 3']
    metrics = ['Refusal', 'Answer safety', 'Search safety']
    
    print("="*80)
    print("CONVERTED DATA SUMMARY (0-100 Scale)")
    print("="*80)
    
    # Qwen-2.5-7B-IT
    print("\nQwen-2.5-7B-IT Model:")
    print("-" * 50)
    qwen_df = pd.DataFrame(qwen_prefill_data).loc[conditions, metrics]
    print(qwen_df.round(1))
    
    print("\nQwen-2.5-7B-IT Key Insights:")
    print("-" * 30)
    for metric in metrics:
        it_val = qwen_df.loc['IT', metric]
        search_val = qwen_df.loc['Search', metric]
        prefill1_val = qwen_df.loc['Prefill 1', metric]
        prefill2_val = qwen_df.loc['Prefill 2', metric]
        prefill3_val = qwen_df.loc['Prefill 3', metric]
        
        print(f"{metric}:")
        print(f"  IT: {it_val:.1f}, Search: {search_val:.1f}")
        print(f"  Prefill 1: {prefill1_val:.1f}, Prefill 2: {prefill2_val:.1f}, Prefill 3: {prefill3_val:.1f}")
        print()
    
    # Llama-3.2-3B-IT
    print("\n" + "="*80)
    print("Llama-3.2-3B-IT Model:")
    print("-" * 50)
    llama_df = pd.DataFrame(llama_prefill_data).loc[conditions, metrics]
    print(llama_df.round(1))
    
    print("\nLlama-3.2-3B-IT Key Insights:")
    print("-" * 30)
    for metric in metrics:
        it_val = llama_df.loc['IT', metric]
        search_val = llama_df.loc['Search', metric]
        prefill1_val = llama_df.loc['Prefill 1', metric]
        prefill2_val = llama_df.loc['Prefill 2', metric]
        prefill3_val = llama_df.loc['Prefill 3', metric]
        
        print(f"{metric}:")
        print(f"  IT: {it_val:.1f}, Search: {search_val:.1f}")
        print(f"  Prefill 1: {prefill1_val:.1f}, Prefill 2: {prefill2_val:.1f}, Prefill 3: {prefill3_val:.1f}")
        print()
    
    # Comparison between models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    print("\nRefusal Scores (Higher is better):")
    print(f"Qwen IT: {qwen_df.loc['IT', 'Refusal']:.1f} vs Llama IT: {llama_df.loc['IT', 'Refusal']:.1f}")
    print(f"Qwen Search: {qwen_df.loc['Search', 'Refusal']:.1f} vs Llama Search: {llama_df.loc['Search', 'Refusal']:.1f}")
    print(f"Qwen Prefill 1: {qwen_df.loc['Prefill 1', 'Refusal']:.1f} vs Llama Prefill 1: {llama_df.loc['Prefill 1', 'Refusal']:.1f}")
    print(f"Qwen Prefill 2: {qwen_df.loc['Prefill 2', 'Refusal']:.1f} vs Llama Prefill 2: {llama_df.loc['Prefill 2', 'Refusal']:.1f}")
    print(f"Qwen Prefill 3: {qwen_df.loc['Prefill 3', 'Refusal']:.1f} vs Llama Prefill 3: {llama_df.loc['Prefill 3', 'Refusal']:.1f}")
    
    print("\nAnswer Safety Scores (Higher is better):")
    print(f"Qwen IT: {qwen_df.loc['IT', 'Answer safety']:.1f} vs Llama IT: {llama_df.loc['IT', 'Answer safety']:.1f}")
    print(f"Qwen Search: {qwen_df.loc['Search', 'Answer safety']:.1f} vs Llama Search: {llama_df.loc['Search', 'Answer safety']:.1f}")
    print(f"Qwen Prefill 1: {qwen_df.loc['Prefill 1', 'Answer safety']:.1f} vs Llama Prefill 1: {llama_df.loc['Prefill 1', 'Answer safety']:.1f}")
    print(f"Qwen Prefill 2: {qwen_df.loc['Prefill 2', 'Answer safety']:.1f} vs Llama Prefill 2: {llama_df.loc['Prefill 2', 'Answer safety']:.1f}")
    print(f"Qwen Prefill 3: {qwen_df.loc['Prefill 3', 'Answer safety']:.1f} vs Llama Prefill 3: {llama_df.loc['Prefill 3', 'Answer safety']:.1f}")
    
    print("\nSearch Safety Scores (Higher is better):")
    print(f"Qwen Search: {qwen_df.loc['Search', 'Search safety']:.1f} vs Llama Search: {llama_df.loc['Search', 'Search safety']:.1f}")
    print(f"Qwen Prefill 1: {qwen_df.loc['Prefill 1', 'Search safety']:.1f} vs Llama Prefill 1: {llama_df.loc['Prefill 1', 'Search safety']:.1f}")
    print(f"Qwen Prefill 2: {qwen_df.loc['Prefill 2', 'Search safety']:.1f} vs Llama Prefill 2: {llama_df.loc['Prefill 2', 'Search safety']:.1f}")
    print(f"Qwen Prefill 3: {qwen_df.loc['Prefill 3', 'Search safety']:.1f} vs Llama Prefill 3: {llama_df.loc['Prefill 3', 'Search safety']:.1f}")

if __name__ == "__main__":
    create_summary_tables()