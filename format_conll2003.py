import argparse
from datasets import load_dataset
import torch
from transformers import BertTokenizer
import os
import json

def process_batch(example, tokenizer):
    target_mask = torch.tensor([0], dtype=torch.int8)
    input_ids = torch.tensor([101], dtype=int)

    for token, ner_tag in zip(example['tokens'], example['ner_tags']):
        encoded_tok = tokenizer.encode(token, return_tensors='pt')[0, 1:-1]

        input_ids = torch.cat((input_ids, encoded_tok))
        target_mask = torch.cat((target_mask, torch.full_like(encoded_tok, 1 if ner_tag != 0 else 0)))

    target_mask = torch.cat((target_mask, torch.tensor([0]))).unsqueeze(0)
    input_ids = torch.cat((input_ids, torch.tensor([102]))).unsqueeze(0)

    return {'input_ids': input_ids, 'target_mask': target_mask}

def main():
    parser = argparse.ArgumentParser(description='Prepare CycleNER model for training.')
    parser.add_argument('--output_dir', default='./data/CoNLL2003', help='Directory to output the formatted CoNLL2003 dataset to.')
    parser.add_argument('--bert_model', default='bert-base-cased', help='BERT model to harvest context vectors from.')
    args = parser.parse_args()


    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    dataset = load_dataset("conll2003")

    original_tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    # New non-BIO scheme
    new_tags = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}
    new_tags_string = {0: 'O', 1: 'PER', 2: 'ORG', 3: 'LOC', 4: 'MISC'}

    for split in ['train', 'validation', 'test']:
        dataset[split] = dataset[split].map(lambda batch: {
            **process_batch(batch, tokenizer),
            'ner_tags': [new_tags[tag] for tag in batch['ner_tags']]
            }).remove_columns(['id', 'chunk_tags'])

    dataset.save_to_disk(args.output_dir)

    with open(os.path.join(args.output_dir, 'tag_mappings.json'), 'w+') as file:
        json.dump({
            'original_tags': original_tags,
            'new_tags': new_tags,
            'new_tags_string': new_tags_string
        }, file, indent=4)

if __name__ == '__main__':
    main()
