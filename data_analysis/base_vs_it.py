"""
Simple analysis script to compare search query usage between base and IT models.
Focuses on: number of search queries emitted per question and how many questions emit search queries.
"""

import json
from typing import Dict, List

def load_json_data(filepath: str) -> List[Dict]:
    """Load JSON data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_search_queries(item: Dict) -> List[str]:
    """Extract search queries from search_information field."""
    search_info = item.get('search_information', [])
    queries = []
    for search_item in search_info:
        if 'query' in search_item:
            queries.append(search_item['query'].strip())
    return queries

def analyze_search_usage(data: List[Dict]) -> Dict:
    """Analyze search query usage patterns in the data."""
    total_questions = len(data)
    questions_with_search = 0
    total_search_queries = 0
    search_queries_per_question = []
    
    for item in data:
        # Count questions with search queries in search_information field
        search_queries = extract_search_queries(item)
        if search_queries:
            questions_with_search += 1
            total_search_queries += len(search_queries)
            search_queries_per_question.append(len(search_queries))
        else:
            search_queries_per_question.append(0)
    
    return {
        'total_questions': total_questions,
        'questions_with_search_queries': questions_with_search,
        'percentage_with_search_queries': (questions_with_search / total_questions) * 100,
        'total_search_queries': total_search_queries,
        'avg_queries_per_question': total_search_queries / total_questions if total_questions > 0 else 0,
        'avg_queries_per_question_with_search': sum(search_queries_per_question) / questions_with_search if questions_with_search > 0 else 0,
        'search_queries_per_question': search_queries_per_question
    }

def main():
    """Main function to run the analysis."""
    base_file = "/data/kebl6672/AGENTIC-RL/refusal_responses/qwen7b_ppo_local/qwen_refusal_full_search_base.json"
    it_file = "/data/kebl6672/AGENTIC-RL/refusal_responses/qwen7b_ppo_local/qwen_refusal_full_search.json"
    
    print("Loading data...")
    base_data = load_json_data(base_file)
    it_data = load_json_data(it_file)
    
    print("Analyzing search query usage...")
    base_analysis = analyze_search_usage(base_data)
    it_analysis = analyze_search_usage(it_data)
    
    print("\n" + "="*60)
    print("SEARCH QUERY USAGE COMPARISON")
    print("="*60)
    
    print(f"\n📊 BASE MODEL:")
    print(f"  Total Questions: {base_analysis['total_questions']}")
    print(f"  Questions with Search Queries: {base_analysis['questions_with_search_queries']} ({base_analysis['percentage_with_search_queries']:.1f}%)")
    print(f"  Total Search Queries: {base_analysis['total_search_queries']}")
    print(f"  Average Queries per Question: {base_analysis['avg_queries_per_question']:.2f}")
    print(f"  Average Queries per Question (with search): {base_analysis['avg_queries_per_question_with_search']:.2f}")
    
    print(f"\n📊 IT MODEL:")
    print(f"  Total Questions: {it_analysis['total_questions']}")
    print(f"  Questions with Search Queries: {it_analysis['questions_with_search_queries']} ({it_analysis['percentage_with_search_queries']:.1f}%)")
    print(f"  Total Search Queries: {it_analysis['total_search_queries']}")
    print(f"  Average Queries per Question: {it_analysis['avg_queries_per_question']:.2f}")
    print(f"  Average Queries per Question (with search): {it_analysis['avg_queries_per_question_with_search']:.2f}")
    
    print(f"\n📈 DIFFERENCE (IT - Base):")
    print(f"  Questions with Search Queries: {it_analysis['questions_with_search_queries'] - base_analysis['questions_with_search_queries']:+d} ({it_analysis['percentage_with_search_queries'] - base_analysis['percentage_with_search_queries']:+.1f} percentage points)")
    print(f"  Total Search Queries: {it_analysis['total_search_queries'] - base_analysis['total_search_queries']:+d}")
    print(f"  Average Queries per Question: {it_analysis['avg_queries_per_question'] - base_analysis['avg_queries_per_question']:+.2f}")
    print(f"  Average Queries per Question (with search): {it_analysis['avg_queries_per_question_with_search'] - base_analysis['avg_queries_per_question_with_search']:+.2f}")

if __name__ == "__main__":
    main()
