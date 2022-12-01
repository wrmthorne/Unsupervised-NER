# Fully Unsupervised Named Entity Classification

`Please cite this project if you use it in your work.`

`FOREWARNING:` This algorithm does not perform exceptionally well. There may be means to improve this idea but as it is tangential to my research, I am leaving this here for now. Full discussion of evaluation and results can be found at the bottom.

This concept was based on a [towards data science article](https://towardsdatascience.com/ssl-could-avoid-supervised-learning-fd049a27cd1b) by [Ajit Rajasekharan](https://ajitrajasekharan.medium.com/) which used BERT vectors to classify named entities in text. The issue I had with this "unsupervised" approach was the amount of labelled seed data required. This appears to be a trend in unsupervised methods for NER and so this project set to rectify the need for any labelled samples for training. Another issue I had with the approach was only using the word embedding layer of the model for clustering. As is well understood, the context of a word can completely change the meaning e.g. "The **bank** robber stood on the river **bank**". This project also aimed to incorporate context into the clustering approach.

Preidentified entities are extracted from text and the task is to classify the entity type. From a training set, the activations of BERTs embedding layers are used, with some aggregation strategy, to calculate the context of the target terms. These context vectors are then clustered into $K$ clusters where $K$ is the number of NER tags required.

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

Best parameters identified:

| Aggregation | Layers     | Subtoken Aggregation  | Threshold | **Accuracy** |
| ----------- | ---------- | --------------------- | --------- | ------------ |
| CatNLayers  | (-4, None) | torch.sum             | 0.6       | 52.02 ± 3.97 |
| CatNLayers  | (-4, None) | torch.sum             | 1.0       | 52.02 ± 3.97 |

Maximum Accuracy: $62.30%$

| Aggregation | Layers     | Subtoken Aggregation  | Seed | Accuracy | Threshold |
| ----------- | ---------- | --------------------- | ---- | -------- | --------- |
| SumNLayers  | (-4, None) | torch.mean            | 100  | 62.30    | 1.0       |
| MeanNLayers | (-4, None) | torch.mean            | 100  | 62.30    | 1.0       |
| SumNLayers  | (-4, None) | torch.mean            | 100  | 62.30    | 0.6       |
| MeanNLayers | (-4, None) | torch.mean	           | 100  | 62.30    | 0.6       |

Max accuracy is a poor measure of performance. The stability of the model was very poor as performance relied heavily on starting initialisation of cluster centres. The averaged results over the three runs for one of these top results shows this instability well:

| Aggregation | Layers | Subtoken Aggregation  | Threshold | **Accuracy**  |
| ----------- | ------ | --------------------- | --------- | ------------- |
| MeanNLayers | (-4, None) | torch.mean	       | 0.6       | 51.06 ± 11.25 |


## Discussion

There is no sugarcoating the fact that the results are not amazing. Although the best accuracy achieved was 63%, the midrange over all three runs with the same parameters was only 51%, meaning the model is very unstable. The most stable and best predicting parameters appears, overwhelmingly, to be concatenation over the last 4 layers, subtoken aggregation by summation and a threshold of 0.6+. The thresholding result is interesting, in particular, because it indicates that the model is capped at a best accuracy of 75% under these conditions. For a threshold greater than 0.6, for these parameters, the model is predicting no entities outside of the three clusters meaning, even with perfect prediction of every other entity, the model would never predict the 4th, MISC class. This is also the trend among the best performing parameters. This is indicative of an interesting point: trying to cluster MISC entities will not work as they are inherently dissimilar and the context vectors for entities within the PER, LOC and ORG clusters are not similar enough to set a reasonable threshold to exclude MISC entities.

A notable result is that, despite using 2 different aggregation strategies, the top 4 results all shared the same accuracy. The threshold discussion has been made but summation aggregation and mean aggregation should be unlikely to produce the same results, let alone the exact same result. A potential idea is that by aggregating the subtokens using mean, they are projected in such a different direction to any regular context vector and as such, are caught by some cluster centre that is away from the other two. This could be exacerbated by setting a low threshold as many elements could be filtered into the MISC class. However, when evaluating the results for which no aggregation was used and there was no threshold (i.e. 4 clusters), SumNLayers and MeanNLayers performed identically for all seeds with layers (-4, None), and for 1 seed with layers (-3, None) and (-2, None). Sum and mean are very closely related functions, the main difference being the sparsity of the vectors; all vectors are relatively in the same positions just with different magnitudes. i.e.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For layers $L_{12} = \begin{bmatrix} -1.0 & 2.0 & 3.0 & 4.0 & 0.5 \end{bmatrix}$ and $L_{11} = \begin{bmatrix} -0.5 & 1.5 & 2.5 & 3.5 & 0.25 \end{bmatrix}$, the elementwise sum is $\begin{bmatrix} -1.5 & 3.5 & 5.5 & 7.5 & 0.75 \end{bmatrix}$ and the elementwise mean is $\begin{bmatrix} -0.75 & 1.75 & 2.75 & 3.75 & 0.375 \end{bmatrix}$; half of the elementwise summation.

The final point to make about these results is that the context vectors are token dependent. For a given token $t$, the context vector for $t$ is very different to context vector $t\prime$, even if tokens $t_{-n}, \dots, t_{-1}, t_{+1}, \dots, t_{+m}$ are the same. Therefore, it is likely that the model will only perform effectively on seen vocabulary/contexts. Initially, masking the central token was conducted but results were extremely poor i.e. almost random.

## Future Work

An obvious note about these results is that the grid search was not comprehensive. Because of the number of permutations, only a limited scope was resonably explored. The layer ranges were chosen to be the last as they contain the most context specific information but combinations of earlier layers may contain more relevant information for the task. Taking non sequential layers or even all layers may also be more informative. All the top results used some aggregation over the final 4 layers, suggesting that using more layers (potentially to some limit) may yield improved performance. Futher work could expand this search space to identify the true, optimal permutation.

Other future work should seem to resolve the issue of named entity detection. In it's current state, this model simply classifies named entities which is only half the task of named entity recognition. The proposed method in the diagram above is crude and only based on proper nouns identified in POS tagging but this misses many other interesting features such as monetary values.


Exploration of spectral clustering may also yield better results as the clusters may not be centroidal. Although work was started on this concept, time became a limiting factor and memory was the main concern; over 100GB of RAM was required to load the $n \times n$ adjacency matrix into memory. With a memory efficient spectral clustering algorithm, this would be an interesting future approach.

A final idea would be to move away from the entirely unsupervised approach and direct the problem toward a few shot task which would give the model a better chance at identifying entities correctly, while still requiring minimal labelled examples. Ultimately, there is a reason why NER remains a supervised/semi-supervised task.
