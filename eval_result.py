import argparse
import json
import os
from typing import Dict, List
import evaluate
from tqdm import tqdm
from nlgeval import NLGEval

bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')
cider = evaluate.load('sunhill/cider')
# spice = evaluate.load('sunhill/spice')

nlgeval = NLGEval()

def compute_metrics(preds:List[str], refs:List[str], listed_refs:List[List[str]]) -> Dict[str, float]:
    bleu_score = bleu.compute(predictions=preds, references=listed_refs)
    rouge_score = rouge.compute(predictions=preds, references=refs)
    meteor_score = meteor.compute(predictions=preds, references=refs)
    cider_score = cider.compute(predictions=preds, references=refs)
    # spice_score = spice.compute(predictions=preds, references=refs)
    
    print("Overall Evaluation Metrics:")
    print(f"BLEU: {bleu_score['bleu']:.4f}")
    print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
    print(f"METEOR: {meteor_score['meteor']:.4f}")
    print(f"CIDEr: {cider_score['cider_score']:.4f}")
    print("\n\n")
    # print(f"SPICE: {spice_score['spice']:.4f}")
    return {
        "BLEU": bleu_score['bleu'],
        "ROUGE-L": rouge_score['rougeL'],
        "METEOR": meteor_score['meteor'],
        "CIDEr": cider_score['cider_score'],
        # "SPICE": spice_score['spice'],
    }

def compute_aggregated_metrics(aggregated_res_dict: Dict[str, Dict[str, List[str]]]) -> Dict[str, float]:
    all_bleu_scores = []
    all_rouge_scores = []
    all_meteor_scores = []
    all_cider_scores = []
    for src, res in tqdm(aggregated_res_dict.items(), desc="Computing aggregated metrics"):
        preds = res['preds']
        refs = res['refs']
        listed_refs = [[r] for r in refs]
        bleu_scores = []
        rouge_scores = []
        meteor_scores = []
        # cider_scores = []
        # spice_scores = []
        
        for pred in preds:
            pred_list = [pred]
            bleu_score = bleu.compute(predictions=pred_list, references=[refs])
            bleu_scores.append(bleu_score['bleu'])
            rouge_score = rouge.compute(predictions=pred_list, references=[refs])
            rouge_scores.append(rouge_score['rougeL'])
            meteor_score = meteor.compute(predictions=pred_list, references=[refs])
            meteor_scores.append(meteor_score['meteor'])
            # cider_score = cider.compute(predictions=[pred], references=[refs])
            # cider_scores.append(cider_score['cider_score'])
        
        blue_score = max(bleu_scores) if bleu_scores else 0.0
        rouge_score = max(rouge_scores) if rouge_scores else 0.0
        meteor_score = max(meteor_scores) if meteor_scores else 0.0
        # cider_score = max(cider_scores) if cider_scores else 0.0
        
        all_bleu_scores.append(blue_score)
        all_rouge_scores.append(rouge_score)
        all_meteor_scores.append(meteor_score)
        # all_cider_scores.append(cider_score)
    
    avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0.0
    avg_rouge = sum(all_rouge_scores) / len(all_rouge_scores) if all_rouge_scores else 0.0
    avg_meteor = sum(all_meteor_scores) / len(all_meteor_scores) if all_meteor_scores else 0.0
    avg_cider = sum(all_cider_scores) / len(all_cider_scores) if all_cider_scores else 0.0
    
    print("Aggregated Evaluation Metrics:")
    print(f"Aggregated BLEU: {avg_bleu:.4f}")
    print(f"Aggregated ROUGE-L: {avg_rouge:.4f}")
    print(f"Aggregated METEOR: {avg_meteor:.4f}")
    print(f"Aggregated CIDEr: {avg_cider:.4f}")
    return {
        "Aggregated BLEU": avg_bleu,
        "Aggregated ROUGE-L": avg_rouge,
        "Aggregated METEOR": avg_meteor,
        "Aggregated CIDEr": avg_cider,
    }

def compute_aggregated_metrics_nlgeval(aggregated_res_dict: Dict[str, Dict[str, List[str]]], is_debug: bool = False) -> Dict[str, float]:
    all_bleu_scores = []
    all_rouge_scores = []
    all_meteor_scores = []
    all_cider_scores = []
    all_spice_scores = []
    for src, res in tqdm(aggregated_res_dict.items(), desc="Computing aggregated metrics"):
        preds = res['preds']
        refs = res['refs']
        listed_refs = [[r] for r in refs]
        bleu_scores = []
        rouge_scores = []
        meteor_scores = []
        cider_scores = []
        spice_scores = []
        
        for pred in preds:

            scores = nlgeval.compute_individual_metrics(refs, pred)
            
            if is_debug:
                print(scores)
            
            bleu_scores.append(scores['Bleu_4'])
            rouge_scores.append(scores['ROUGE_L'])
            meteor_scores.append(scores['METEOR'])
            cider_scores.append(scores['CIDEr'])
            spice_scores.append(scores['SPICE'])
        
        blue_score = max(bleu_scores) if bleu_scores else 0.0
        rouge_score = max(rouge_scores) if rouge_scores else 0.0
        meteor_score = max(meteor_scores) if meteor_scores else 0.0
        cider_score = max(cider_scores) if cider_scores else 0.0
        spice_score = max(spice_scores) if spice_scores else 0.0
        
        all_bleu_scores.append(blue_score)
        all_rouge_scores.append(rouge_score)
        all_meteor_scores.append(meteor_score)
        all_cider_scores.append(cider_score)
        all_spice_scores.append(spice_score)
    
    avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0.0
    avg_rouge = sum(all_rouge_scores) / len(all_rouge_scores) if all_rouge_scores else 0.0
    avg_meteor = sum(all_meteor_scores) / len(all_meteor_scores) if all_meteor_scores else 0.0
    avg_cider = sum(all_cider_scores) / len(all_cider_scores) if all_cider_scores else 0.0
    avg_spice = sum(all_spice_scores) / len(all_spice_scores) if all_spice_scores else 0.0
    
    print("Aggregated Evaluation Metrics:")
    print(f"Aggregated BLEU: {avg_bleu:.4f}")
    print(f"Aggregated ROUGE-L: {avg_rouge:.4f}")
    print(f"Aggregated METEOR: {avg_meteor:.4f}")
    print(f"Aggregated CIDEr: {avg_cider:.4f}")
    print(f"Aggregated SPICE: {avg_spice:.4f}")
    return {
        "Aggregated BLEU": avg_bleu,
        "Aggregated ROUGE-L": avg_rouge,
        "Aggregated METEOR": avg_meteor,
        "Aggregated CIDEr": avg_cider,
        "Aggregated SPICE": avg_spice,
    }
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--dataset_name", type=str, default='commongen', help="Name of the dataset for evaluation metrics")
    parser.add_argument("--output_dir", type=str, default='./eval_res', help="Path to the output directory")
    parser.add_argument("--is_debug", action='store_true', help="Whether to run in debug mode")
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    assert type(data) == dict, "Input JSON should be a list of dictionaries"
    assert 'pred_texts' in data, "Input JSON should contain 'pred_texts' key"
    assert 'reference_texts' in data, "Input JSON should contain 'reference_texts' key"
    assert 'source_texts' in data, "Input JSON should contain 'source_texts' key"
    assert len(data['pred_texts']) == len(data['reference_texts']) == len(data['source_texts']), "Length of 'pred_texts', 'reference_texts', and 'source_texts' should be the same"
    
    print(f"Total samples: {len(data['pred_texts'])}")
    
    if os.path.exists(f"{args.output_dir}/{args.dataset_name}_aggregated_results.json"):
        print("Aggregated results already exist. Skipping evaluation.")
        aggregated_res_dict = json.load(open(f"{args.output_dir}/{args.dataset_name}_aggregated_results.json", 'r'))
        
    else:
        print("Aggregating results by source texts...")
        aggregated_res_dict = {}
        for pred, ref, src in zip(data['pred_texts'], data['reference_texts'], data['source_texts']):
            if src in aggregated_res_dict:
                aggregated_res_dict[src]['preds'].append(pred)
                aggregated_res_dict[src]['refs'].append(ref)
            else:
                aggregated_res_dict[src] = {'preds': [pred], 'refs': [ref]}    
        
        print(f"Total unique sources: {len(aggregated_res_dict)}")

        with open(f"{args.output_dir}/{args.dataset_name}_aggregated_results.json", 'w') as f:
            json.dump(aggregated_res_dict, f, indent=4)

    if args.is_debug:
        aggregated_res_dict = {k: v for k, v in list(aggregated_res_dict.items())[:10]}
        compute_aggregated_metrics_nlgeval(aggregated_res_dict, is_debug=True)
        return
    
    else:
    
        preds = data['pred_texts']
        refs = data['reference_texts']
        listed_refs = [[r] for r in refs]
        
        metric_dict = compute_metrics(preds, refs, listed_refs)
        aggregated_metric_dict = compute_aggregated_metrics(aggregated_res_dict)
        aggregated_metric_dict_nlgeval = compute_aggregated_metrics_nlgeval(aggregated_res_dict)
        
        with open(f"{args.output_dir}/{args.dataset_name}_metrics.json", 'w') as f:
            json.dump({
                "overall_metrics": metric_dict,
                "aggregated_metrics": aggregated_metric_dict,
                "aggregated_metrics_nlgeval": aggregated_metric_dict_nlgeval
            }, f, indent=4)

if __name__ == "__main__":
    main()