import json

for split in ['train', 'dev', 'test']:
    with open(f'{split}.json', 'r', encoding='utf-8') as infile, \
         open(f'{split}.jsonl', 'w', encoding='utf-8') as outfile:
        src_data = [json.loads(line) for line in infile]
        for item in src_data:
            for l in item["labels"]:
                json.dump({"src": " ".join(item["inputs"]), "tgt": l}, outfile)
                outfile.write('\n')

