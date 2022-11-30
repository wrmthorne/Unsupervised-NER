from context_vector_clustering import ContextClustering, CatNLayers, SumNLayers, MeanNLayers, SelectLayerN, AggregateSubtokens
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
from datasets import load_from_disk
import torch


def evaluate(cluster_ids, split, classes, tokenizer, subtoken_aggregation=False):
    cluster_ids = cluster_ids.copy()
    sample_predictions = []

    for example in tqdm(split):
        token_ids = {f'{i}-{token}': {
            'ids': [i] if subtoken_aggregation else tokenizer.encode(token)[1:-1],
            'ner_tag': ner_tag
        } for i, (token, ner_tag) in enumerate(zip(example['tokens'], example['ner_tags'])) if ner_tag != 0}

        token_predictions = {token: {
            'cluster_id': max(set((lst := [cluster_ids.pop(0) for _ in range(len(attributes['ids']))])), key=lst.count),
            'ner_tag': attributes['ner_tag']
        } for token, attributes in token_ids.items()}

        sample_predictions.append(token_predictions)

    df = pd.DataFrame(data={x: 0 for x in set([x for y in split['ner_tags'] for x in y if x != 0])}, index=classes)

    for pred in sample_predictions:
        for p in pred.values():
            df[p['ner_tag']].loc[p['cluster_id']] += 1

    eye = np.eye(len(df.columns), dtype=bool)
    total = np.sum(df.to_numpy().flatten())

    results = []
    for i in range(len(df.columns)):
        mapping = list(df[(df * np.roll(eye, i, axis=1)) != 0].stack().index)
        accuracy = np.sum((df * np.roll(eye, i, axis=1)).to_numpy().flatten()) / total * 100
        results.append({'mapping': mapping, 'accuracy': accuracy})

    best_accuracy = sorted(results, key=lambda x: x['accuracy'], reverse=True)[0]['accuracy']

    return results, best_accuracy

def collect_results(model_fit, tokenizer, val_data, agg_func, subtok_agg, l_range, seed, threshold=None, classes=[0, 1, 2, 3]):
    predictions = model_fit.predict(val_data, threshold=threshold)
    _, best_acc = evaluate(predictions.tolist(), val_data, classes, tokenizer, subtoken_aggregation=subtok_agg)

    results_dict = {
        'Aggregation': str(agg_func),
        'Layers': l_range,
        'Subtoken Aggregation?': subtok_agg,
        'Seed': seed,
        'Accuracy': best_acc,
        'Threshold': threshold
        }

    df = pd.DataFrame(results_dict)

    with open('./intermediate_dump.txt', 'a+') as file:
        file.writelines(f'{results_dict}\n')

    return df

def main():
    model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = load_from_disk('./data/CoNLL2003')
    
    seeds = [0, 42, 100]
    aggregations = [CatNLayers, MeanNLayers, SumNLayers]
    layer_ranges = [(-2, None), (-3, None), (-4, None)]
    subtoken_aggregation = [None, AggregateSubtokens(torch.sum), AggregateSubtokens(torch.mean)]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1]

    results = pd.DataFrame(columns=['Aggregation', 'Layers', 'Subtoken Aggregation?', 'Seed', 'Accuracy', 'Threshold'])

    for agg_func in aggregations:
        for subtok_agg in subtoken_aggregation:
            for l_range in layer_ranges:
                for seed in seeds:
                    for threshold in [None, thresholds]:
                        if threshold is None:
                            model = ContextClustering(n_clusters=4, random_state=seed)
                            model_fit = model.fit(dataset['train'], aggregation=agg_func(*l_range), subtoken_aggregation=subtok_agg)
                            df = collect_results(model_fit, tokenizer, dataset['validation'], agg_func, subtok_agg, l_range, seed)

                            pd.concat((results, df), ignore_index=True)
                        else:
                            model = ContextClustering(n_clusters=3, random_state=seed)
                            model_fit = model.fit(dataset['train'], aggregation=agg_func(*l_range), subtoken_aggregation=subtok_agg)

                            for t in thresholds:
                                df = collect_results(model_fit, tokenizer, dataset['validation'], agg_func, subtok_agg, l_range, seed, threshold=t, classes=[-np.inf, 0, 1, 2])

    results.to_pickle('./gridsearch_results.pkl')

if __name__ == '__main__':
    main()