# Fully Unsupervised Named Entity Classification

This algorithm is designed to take preidentified entities in text and determine which entity type they are. From a training set, the activations of BERTs embedding layers are used, with some aggregation strategy, to calculate the context of the target terms. These context vectors are then clustered into $K$ where $K$ is the number of NER tags you want.

`FOREWARNING:` This algorithm does not perform exceptionally well. There may be means to improve this idea but as it is tangential to my research, I am leaving this here for now.

![diagram of idea](./basic_idea.png)

The code is designed to run on a GPU with a decent amount of VRAM for optimal performance. CPU works but it is very slow.

# Preparation

CoNLL2003 has been used as an annotated dataset for experiments on this model. The dataset must be first prepared using the `format_conll2003.py` script. By default, `BERT-base-cased` is used. To use a different version of BERT, this must be specified to the script as the vocabularies of different BERT models varies. For the script's input options, use:

```bash
python format_conll2003.py --help
```

This script converts CoNLL2003 from a BIO scheme to PER, ORG, LOC, MISC.

# Using the Model

The model makes use of a similar API structure to SKlearn, using a `fit`, `predict` function set. Example usage:

```python
from context_vector_clustering import ContextClustering

...

model = ContextClustering(n_clusters=4, random_state=42)
model.fit(dataset['train'])

# Note cluster IDs do not match NER IDs
cluster_ids = model.predict(dataset['validation'])
```

For a full example of usage, view `development_notebook.ipynb`.

## Aggregation Methods

Different aggregation strategies have been supplied for combining BERT embedding layers.

| Strategy     | Params     | Description |
| ------------ | ---------- | ----------- |
| SelectLayerN | n          | Extracts a single layer, as specified by n. |
| SumNLayers   | start, end | Sums sequential layers between the specified bounds. |
| MeanNLayers  | start, end | Averages sequential layers between the specified bounds. |
| CatNLayers   | start, end | Concatenates sequential layers between the specified bounds. |

These methods can be imported form the same file:

```python
from context_vector_clustering import CatNLayers, SumNLayers, ...

agg = CatNLayers(-4, None)
...
```

## Subtoken Aggregation

A class for aggregating BERT's subtokens (e.g. ##a, ##tion) is also supplied. This combines all subtokens for a specific word into a single context vector instead of many separate context vectors. Two functions can be passed to this class as aggregation strategies: `torch.sum` and `torch.mean`. Can be imported the same way as everything else:

```python
from context_vector_clustering import AggregateSubtokens

agg = AggregateSubtokens(torch.sum)
...
```

## Thresholding

To experiment with the MISC class in CoNLL2003, the option to set a threshold while predicting with the model is provided. `NOTE:` You must set your `n_clusters` parameter to be $K - 1$ as in CoNLL2003, MISC is one of the tags, ergo you don't want to define a cluster for it when calling anything outside of your threshold MISC. Any values larger than the threshold will be added to the `-inf` class i.e. MISC.

```python
num_tags = 4
num_clusters = num_tags - 1

model = ContextClustering(n_clusters=num_clusters, random_state=42)
model.fit(dataset['train'])

cluster_ids = model.predict(dataset['validation'], threshold=0.3)
```

# Experiments

A grid search was performed to find the best combination of parameters. For each combination of parameters, the experiment was run 3 times with different seeds (`0`, `42`, `100`) to account for any anomalous performance. An evaluation method was written to find the best mapping of predicted cluster labels to actual tags, where the best accuracy was taken to be the accuracy for that combination of parameters. To isolate only determining which class the token belongs to, all tokens with an NER tag that isn't `O` were selected.

The following ranges were used in the grid search:

Aggregation Strategies: `CatNLayers`, `SumNLayers`, `MeanNLayers` <br>
Layer Ranges: `(-2, None)`, `(-3, None)`, `(-4, None)` i.e. last 2, 3, or 4 layers<br>
Subtoken Aggregations: `None`, `torch.sum`, `torch.mean` <br>
Thresholds: `None`, `0.1`, `0.2`, `0.25`, `0.3`, `0.35`, `0.4`, `0.45`, `0.5`, `0.55`, `0.6`

The grid search code can be found in `grid_search.py`.

Full exploration of the results can be conducted in `grid_search_inspection.ipynb`.

## Results

Maximum Accuracy: $62.303847%$ <br>

| Aggregation | Layers | Subtoken Aggregation  | Seed | Accuracy | Threshold |
| ----------- | ------ | --------------------- | ---- | -------- | --------- |
| SumNLayers  | (-4, None) | torch.mean | 100 | 62.303847 | 1.0 |
| MeanNLayers | (-4, None) | torch.mean | 100 | 62.303847 | 1.0 |
| SumNLayers  | (-4, None) | torch.mean | 100 | 62.303847 | 0.6 |
| MeanNLayers | (-4, None) | torch.mean	| 100 | 62.303847 | 0.6 |

## Discussion

There is no sugarcoating the fact that the results are not amazing. Although the best accuracy achieved was 63%, the midrange over all three runs with the same parameters was only 51%, meaning the model is very unstable. The most stable and best predicting parameters appears, overwhelmingly, to be concatenation over the last 4 layers, subtoken aggregation by summation and a threshold of 0.6+. The thresholding result is interesting, in particular, because it indicates that the model is capped at a best accuracy of 75% under these conditions. For a threshold greater than 0.6, for these parameters, the model is predicting no entities outside of the three classes meaning, even with perfect prediction of every other entity, the model would never predict the 4th, MISC class. This is also the trend among the best performing parameters. This is indicative of an interesting point: trying to cluster MISC entities will not work as they are inherently dissimilar and the context vectors for entities within the PER, LOC and ORG clusters are not similar enough to set a reasonable threshold to exclude MISC entities.

An obvious note about these results is that the grid search was not comprehensive. Because of the number of permutations, only a limited scope was resonable explored. The layer ranges were chosen to be the last as they contain the most context specific information but combinations of earlier layers may contain more relevant information for the task. Taking non sequential layers or even all layers may also be more informative. That being said, given that the majority of the contextual information is concentrated in the final few layers, it is unlikely; hence, why they were not explored.

The final criticism to make about these results is that the context vectors are token dependent. For a given token $t$, the context vector for $t$ is very different to context vector $t\prime$, even if tokens $t_{-n}, \dots, t_{-1}, t_{+1}, \dots, t_{+m}$ are the same. Therefore, it is likely that the model will only perform effectively on seen vocabulary/contexts. Initially, masking the central token was conducted but results were extremely poor i.e. almost random.