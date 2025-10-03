import argparse
import json
import re
from unittest import result
import requests

from tqdm import tqdm

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--split", type=str, help="Dataset split (e.g., 'train', 'valid', 'test')")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (e.g., 'commongen', 'dimongen')")
    parser.add_argument("--output_path", type=str, default=None, required=False, help="Path to the output file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top passages to retrieve")
    parser.add_argument("--is_debug", action='store_true', help="Run in debug mode with fewer samples")
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = f"{args.dataset_dir}/{args.split}_augmented_top{args.top_k}.jsonl"
    
    with open(f"{args.dataset_dir}/{args.split}.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]

    if args.is_debug:
        data = data[:10]

    retrieval_augmented_data = []
    
    cur_src = ""
    pre_item = None
    for item in tqdm(data, desc="Processing data"):
        src = item['src']
        
        if src == cur_src:
            retrieval_augmented_data.append({**item, 'retrieved_passages': pre_item['retrieved_passages']})
            continue
        
        cur_src = src
        pre_item = item
        
        query = get_detailed_instruct(task, src)
        
        payload = {
            "query": query,
            "k": args.top_k
        }

        resp = requests.post("http://localhost:8000/search", json=payload)
        resp.raise_for_status()
        results = resp.json()
        
        retrieved_passages = [doc['text'] for doc in results['results']]
        item['retrieved_passages'] = retrieved_passages
        retrieval_augmented_data.append(item)

    with open(args.output_path, 'w') as f:
        for item in retrieval_augmented_data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    main()