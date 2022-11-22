import torch
from datasets import load_dataset
from transformers import BertTokenizer
from context_vector_clustering import ContextClustering

model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

original_tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
# New non-BIO scheme
new_tags = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}
new_tags_string = {0: 'O', 1: 'PER', 2: 'ORG', 3: 'LOC', 4: 'MISC'}

dataset = load_dataset("conll2003")

def process_batch(example):
    target_mask = torch.tensor([0], dtype=torch.int8)
    input_ids = torch.tensor([101], dtype=int)

    for token, ner_tag in zip(example['tokens'], example['ner_tags']):
        encoded_tok = tokenizer.encode(token, return_tensors='pt')[0, 1:-1]

        input_ids = torch.cat((input_ids, encoded_tok))
        target_mask = torch.cat((target_mask, torch.full_like(encoded_tok, 1 if ner_tag != 0 else 0)))

    target_mask = torch.cat((target_mask, torch.tensor([0]))).unsqueeze(0)
    input_ids = torch.cat((input_ids, torch.tensor([102]))).unsqueeze(0)

    return {'input_ids': input_ids, 'target_mask': target_mask}

for split in ['train', 'validation', 'test']:
    dataset[split] = dataset[split].map(lambda batch: {
        **process_batch(batch),
        'ner_tags': [new_tags[tag] for tag in batch['ner_tags']]
        }).remove_columns(['id', 'chunk_tags'])

X_fit = ContextClustering().fit(dataset['train'])
predictions = X_fit.predict(dataset['validation'])