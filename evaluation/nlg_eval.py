import nlgeval
import os
import sys
from nlgeval import NLGEval
from datasets import load_dataset
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predictions file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save evaluation results")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="Split to evaluate")
    parser.add_argument("--ref_path", type=str, default=None, help="Path to references file")
    parser.add_argument("--generation_method", type=str, default="diffusion", help="Generation method")
    
    args = parser.parse_args()
    
    n = NLGEval(no_skipthoughts=True, no_glove=True)
    
    pred_res = json.load(open(args.pred_path, 'r'))
    pred_txt = pred_res["pred_texts"]
    ref_txt = pred_res["reference_texts"]

    
    print(f"pred_texts length: {len(pred_txt)}")
    print(f"ref_texts length: {len(ref_txt)}")
    
    metrics_dict = n.compute_metrics(ref_list=[ref_txt], hyp_list=pred_txt)
    
    # Convert metrics to percentages, CIDEr to x10, and format to two decimal places
    for key, value in metrics_dict.items():
        if key == "CIDEr":
            metrics_dict[key] = round(value * 10, 2)
        else:
            metrics_dict[key] = round(value * 100, 2)
            
    print("***** Evaluation results *****")
    print(metrics_dict)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    with open(f'{args.save_path}/nlg_eval_res.json', 'w') as f:
        json.dump(metrics_dict, f)
    

if __name__ == "__main__":
    main()
