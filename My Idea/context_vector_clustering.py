from transformers import BertTokenizer, BertModel
from kmeans_pytorch import kmeans, kmeans_predict
from tqdm import tqdm
import torch
import numpy as np
import random

def last_n_layers(n):
    '''
    Aggregates embedding layers by summing the last n layers
    
    Args:
        n: number of layers to aggregate over

    Returns:
        aggregate: callable
    '''
    def aggregate(stacked_token_embeddings, target_indices):
        '''
        Performs the aggregation on the last n layers where n is specified in the parent function

        Args:
            stacked_token_embeddings: tensor of token embeddings that have been repeated n times where n is the
                total sum of 1s in the target_mask
            target_indices: tensor of indices in stacked_token_embeddings to calculate aggreagation over

        Returns:
            tensor of shape (n, 768) of aggregated embeddings
        '''
        return torch.sum(stacked_token_embeddings[torch.arange(0, target_indices.size(0)), target_indices[:, 1], -n:], dim=1)

    return aggregate

class ContextClustering:
    def __init__(self, n_clusters=4, random_state=0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.distance = 'cosine'
        self.masked_targets = False

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

    def _get_context_vectors(self, input_ids, attention_mask, target_mask, aggregation):
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
        stacked_token_embeddings = token_embeddings.repeat(torch.sum(target_mask), 1, 1, 1)
        embedding_aggregates = aggregation(stacked_token_embeddings, target_indices)

        return embedding_aggregates

    def _collect_contexts(self, X, aggregation=last_n_layers(4)):
        '''
        Iterates over all rows in the dataset split and obtains the context vectors of target
        tokens
        
        Args:
            X: Huggingface Dataset split with columns ['input_ids', 'target_mask']

        Returns:
            tensor of shape (n, 768) where n is the total sum of 1s that appear in the target_mask
                column    
        '''
        context_vectors = torch.empty(0, 768).to(self.device)

        for example in tqdm(X, desc='Collecting Contexts'):
            input_ids = torch.tensor(example['input_ids'])
            target_mask = torch.tensor(example['target_mask'])

            if self.masked_targets:
                attention_mask = ~torch.tensor(example['target_mask'], dtype=bool)
            else:
                attention_mask = torch.ones_like(input_ids)

            embedding_aggregate = self._get_context_vectors(input_ids, attention_mask, target_mask, aggregation)
            context_vectors = torch.cat((context_vectors, embedding_aggregate))

        return context_vectors

    def fit(self, X, distance='cosine', masked_targets=False):
        '''
        Finds all the relevant context vectors in the data and clusters them using KMeans

        Args:
            X: Huggingface Dataset split with columns ['input_ids', 'target_mask']
            distance: Distance metric to use from ['cosine', 'euclidean']
            masked_targets: Attention masking strategy. True uses the inverse of the target mask
                as the attention mask

        Returns:
            self: Fitted estimator
        '''
        assert distance in ['cosine', 'euclidean']
        self.distance = distance
        self.masked_targets = masked_targets

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




