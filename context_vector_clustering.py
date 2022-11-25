from transformers import BertTokenizer, BertModel
from kmeans_pytorch import kmeans, kmeans_predict
from tqdm import tqdm
import torch
import numpy as np
import random
from transformers import logging

# Surpress unnecessary warnings about BERT model weights
logging.set_verbosity_error()


class SumSubtokens:
    '''Collects subtoken embedding aggregations into single embedding aggregation using summation'''
    def __init__(self):
        pass

    def __call__(self, embedding_aggregate, target_mask):
        '''
        Finds all subtokens which compose input sentence words and sums the embedding aggregations
        for each word's subtokens.

        Args:
            embedding_aggregate: Precomputed embedding aggregates for context vectors
            target_mask: The tokens in the sequence for which the context vector was obtained,
                incremented by word affiliation

        Returns:
            tensor of shape (m, 768) where m is the number of words the subwords compose into
        '''
        non_zero_chunks = target_mask[target_mask != 0]
        binned_chunks = torch.bincount(non_zero_chunks)[1:]

        chunked_aggregate = torch.split(embedding_aggregate, list(binned_chunks))
        summed_chunks = [torch.sum(chunk, dim=0).unsqueeze(0) for chunk in chunked_aggregate]
        try:
            return torch.cat(summed_chunks, dim=0)
        except Exception:
            return embedding_aggregate

class SelectLayerN:
    '''Extracts single embedding layer to represent token'''
    def __init__(self, n: int):
        self.n = n

    def __call__(self, stacked_token_embeddings, target_indices):
        '''
        Extracts the nth layer as the chosen embedding layer

        Args:
            stacked_token_embeddings: tensor of token embeddings that have been repeated m times where m is the
                total sum of 1s in the target_mask
            target_indices: tensor of indices in stacked_token_embeddings to calculate aggreagation over

        Returns:
            tensor of shape (1, 768) of aggregated embeddings
        '''
        target_indices[:, 0] = torch.arange(0, target_indices.size(0))
        return stacked_token_embeddings[target_indices[:, 0], target_indices[:, 1], self.n]

class SumNLayers:
    '''
    Aggregation over consecutive embedding layers using summation. Start layer must be provided. If no end
    is provided, all layers from the start layer will be used.
    '''
    def __init__(self, start: int, end:int=None):
        self.start = start
        self.end = end

    def __call__(self, stacked_token_embeddings, target_indices):
        '''
        Performs the summation over a specified range of layers

        Args:
            stacked_token_embeddings: tensor of token embeddings that have been repeated m times where m is the
                total sum of 1s in the target_mask
            target_indices: tensor of indices in stacked_token_embeddings to calculate aggreagation over

        Returns:
            tensor of shape (m, 768) of aggregated embeddings
        '''
        target_indices[:, 0] = torch.arange(0, target_indices.size(0))
        target_embed_layers = stacked_token_embeddings[target_indices[:, 0], target_indices[:, 1], self.start:self.end]
        return torch.sum(target_embed_layers, dim=1)

class MeanNLayers:
    '''
    Aggregation over consecutive embedding layers using mean. Start layer must be provided. If no end
    is provided, all layers from the start layer will be used.
    '''
    def __init__(self, start: int, end:int=None):
        self.start = start
        self.end = end

    def __call__(self, stacked_token_embeddings, target_indices):
        '''
        Performs the mean over a specified range of layers

        Args:
            stacked_token_embeddings: tensor of token embeddings that have been repeated m times where m is the
                total sum of 1s in the target_mask
            target_indices: tensor of indices in stacked_token_embeddings to calculate aggreagation over

        Returns:
            tensor of shape (m, 768) of aggregated embeddings
        '''
        target_indices[:, 0] = torch.arange(0, target_indices.size(0))
        target_embed_layers = stacked_token_embeddings[target_indices[:, 0], target_indices[:, 1], self.start:self.end]
        return torch.mean(target_embed_layers, dim=1)

class CatNLayers:
    '''
    Aggregation over consecutive embedding layers using concatenation. Start layer must be provided. If no end
    is provided, all layers from the start layer will be used.
    '''
    def __init__(self, start: int, end:int=None):
        self.start = start
        self.end = end
        self.n = len(torch.arange(13)[self.start:self.end])
        
    def __call__(self, stacked_token_embeddings, target_indices):
        '''
        Performs the concatenation over a specified range of layers

        Args:
            stacked_token_embeddings: tensor of token embeddings that have been repeated m times where m is the
                total sum of 1s in the target_mask
            target_indices: tensor of indices in stacked_token_embeddings to calculate aggreagation over

        Returns:
            tensor of shape (m, n*768) of aggregated embeddings
        '''
        target_indices[:, 0] = torch.arange(0, target_indices.size(0))
        target_embed_layers = stacked_token_embeddings[target_indices[:, 0], target_indices[:, 1], self.start:self.end]
        split_target_layers = list(target_embed_layers.permute(1, 0, 2))
        return torch.cat(split_target_layers, dim=1)

class ContextClustering:
    def __init__(self, n_clusters=4, random_state=0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.cluster_ids = None
        self.cluster_centres = None

        self.model = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        self.model.eval()
        self.model.to(self.device)

    def _set_seed(self):
        '''
        Systematically sets the seeds of all random elements to the random_state passed at instantiation
        '''
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)

    def _get_context_vectors(self, input_ids, attention_mask, target_mask):
        '''
        Collects the context vectors from BERT for each term indicated in the target mask in the input sequence

        Args:
            input_ids: Encoded input string
            attention_mask: Which terms the BERT should attend to
            target_mask: The tokens in the sequence for which the context vector should be obtained

        Returns:
            tensor of shape (n, 768) where n is the number of 1s in the target mask
        '''
        # Ignore gradient to improve performance
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device), attention_mask.to(self.device))
            hidden_states = outputs.hidden_states

        # Extract model embeddings layer activations
        token_embeddings = torch.stack(hidden_states, dim=0)

        # Remove batches dimension
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Swap layer and token dimensions
        token_embeddings = token_embeddings.permute(1, 0, 2)
        
        # Identify indices within encoded text to calculate context embeddings
        target_indices = target_mask.to(self.device).nonzero()

        # Use the sum of the last 4 embedding layers as an aggregation of context for the selected indices
        stacked_token_embeddings = token_embeddings.repeat(target_indices.size(0), 1, 1, 1)
        embedding_aggregates = self.aggregation(stacked_token_embeddings, target_indices)

        if self.subtoken_aggregation is not None:
            embedding_aggregates = self.subtoken_aggregation(embedding_aggregates, target_mask)

        return embedding_aggregates

    def _collect_contexts(self, X):
        '''
        Iterates over all rows in the dataset split and obtains the context vectors of target
        tokens
        
        Args:
            X: Huggingface Dataset split with columns ['input_ids', 'target_mask']

        Returns:
            tensor of shape (n, m*768) where n is the total sum of 1s that appear in the target_mask
                column for m > 1 if concatenation of 2+ layers is used as as aggregation, else m = 1
        '''
        if isinstance(self.aggregation, CatNLayers):
            context_dim = self.aggregation.n * 768
        else:
            context_dim = 768

        context_vectors = torch.empty(0, context_dim).to(self.device)

        for example in tqdm(X, desc='Collecting Contexts'):
            input_ids = torch.tensor(example['input_ids'])
            target_mask = torch.tensor(example['target_mask'])

            if self.masked_targets:
                attention_mask = ~torch.tensor(example['target_mask'], dtype=bool)
            else:
                attention_mask = torch.ones_like(input_ids)

            embedding_aggregate = self._get_context_vectors(input_ids, attention_mask, target_mask)
            context_vectors = torch.cat((context_vectors, embedding_aggregate))

        return context_vectors

    def fit(self, X, distance='cosine', masked_targets=False, aggregation=SumNLayers(4), subtoken_aggregation=None):
        '''
        Finds all the relevant context vectors in the data and clusters them using KMeans

        Args:
            X: Huggingface Dataset split with columns ['input_ids', 'target_mask']
            distance: Distance metric to use from ['cosine', 'euclidean']
            masked_targets: Attention masking strategy. True uses the inverse of the target mask
                as the attention mask
            aggregation: callable for aggregation strategy of context vectors

        Returns:
            self: Fitted estimator
        '''
        assert distance in ['cosine', 'euclidean']

        self.distance = distance
        self.masked_targets = masked_targets
        self.aggregation = aggregation
        self.subtoken_aggregation = subtoken_aggregation

        context_vectors = self._collect_contexts(X)

        self._set_seed()
        self.cluster_ids, self.cluster_centres = kmeans(context_vectors, self.n_clusters, distance=self.distance, device=self.device)

        return self

    def predict(self, X):
        '''
        For an input set, predicts the cluster assignments

        Args:
            X: Huggingface Dataset split with columns ['input_ids', 'target_mask']

        Returns:
            flat tensor of size n where n is the total sum of 1s that appear in the target_mask
                column
        '''
        assert self.cluster_ids is not None and self.cluster_centres is not None

        context_vectors = self._collect_contexts(X)

        return kmeans_predict(context_vectors, self.cluster_centres, distance=self.distance, device=self.device)