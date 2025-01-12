import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# -------------------------------------------------------
# 1. GNN_VAE Model Definition (same as in train.py)
# -------------------------------------------------------
class GNN_VAE(nn.Module):
    def __init__(self, input_dim, hidden_channels, latent_dim, decoder_channels,
                 num_tags, tag_embedding_dim=50, dropout=0.5):
        super(GNN_VAE, self).__init__()
        # Encoder
        self.conv1 = GATConv(input_dim, hidden_channels, heads=2, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = GCNConv(hidden_channels, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.dropout2 = nn.Dropout(p=dropout)

        self.mu_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, decoder_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(decoder_channels, 1)
        )

        # Residual
        self.residual = nn.Linear(latent_dim, latent_dim)

        # Tag Embedding
        self.tag_embedding = nn.Embedding(num_tags + 1, tag_embedding_dim, padding_idx=num_tags)
        self.tag_dim_reduction = nn.Linear(tag_embedding_dim, 50)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.dropout2(x)

        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, edge_index):
        row, col = edge_index
        z_row = self.residual(z[row])
        z_col = self.residual(z[col])
        edge_embeddings = torch.cat([z_row, z_col], dim=1)
        return self.decoder(edge_embeddings).squeeze()

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, edge_index)
        return recon, mu, logvar

    def incorporate_tags(self, tag_indices):
        """
        tag_indices: Tensor of shape (num_nodes, max_tags)
        Each row is the list of tag indices for a particular node (movie).
        """
        tag_embedded = self.tag_embedding(tag_indices)
        mask = (tag_indices != self.tag_embedding.padding_idx).unsqueeze(-1).float()
        tag_embedded = tag_embedded * mask
        tag_sum = tag_embedded.sum(dim=1)
        tag_count = mask.sum(dim=1)
        tag_mean = tag_sum / (tag_count + 1e-8)
        tag_reduced = self.tag_dim_reduction(tag_mean)
        return tag_reduced

# -------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------
def load_best_model(model_path, input_dim, hidden_channels, latent_dim, decoder_channels, num_tags, dropout):
    """
    Loads the GNN_VAE model from file and sets it to eval mode.
    """
    model = GNN_VAE(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        latent_dim=latent_dim,
        decoder_channels=decoder_channels,
        num_tags=num_tags,
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def run_inference(model, data, edge_subset=None):
    """
    Runs the forward pass on a specified subset of edges (or all edges if None).
    Returns the reconstructed edge weights (ratings).
    """
    if edge_subset is None:
        edge_subset = data.edge_index  # use all edges
    with torch.no_grad():
        recon, mu, logvar = model(data.x, edge_subset)
    return recon, mu, logvar

# -------------------------------------------------------
# 3. Data Preparation (Mirror from train.py)
# -------------------------------------------------------
def prepare_data_inference(dataset_path):
    """
    Rebuilds the same Data object and tag_indices tensor from train.py.
    This must match EXACTLY how you processed data in training,
    including normalizations, embeddings, PCA, etc.
    """
    # 3.1 Load CSVs
    ratings = pd.read_csv(f"{dataset_path}/ratings.csv")
    movies = pd.read_csv(f"{dataset_path}/movies.csv")
    tags = pd.read_csv(f"{dataset_path}/tags.csv")

    # 3.2 Encode genres
    genres = set()
    for g_list in movies['genres'].str.split('|'):
        genres.update(g_list)
    genres.discard('(no genres listed)')
    genre_list = sorted(list(genres))
    mlb = MultiLabelBinarizer(classes=genre_list)
    genre_features = mlb.fit_transform(movies['genres'].str.split('|'))
    genre_tensor = torch.tensor(genre_features, dtype=torch.float)

    # 3.3 Process tags
    unique_tags = tags['tag'].unique()
    tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    # In train.py, num_tags was len(tag_to_idx).
    # We'll assume that is 1589 to match your best_model parameters.

    padding_idx = 1589  # num_tags from training
    tags_grouped = tags.groupby('movieId')['tag'].apply(list).reset_index()
    movies = movies.merge(tags_grouped, on='movieId', how='left')
    movies['tag'] = movies['tag'].apply(lambda x: x if isinstance(x, list) else [])
    movies['tag_indices'] = movies['tag'].apply(lambda x: [tag_to_idx[t] for t in x if t in tag_to_idx])

    max_tags = movies['tag_indices'].apply(len).max()
    movies['tag_indices_padded'] = movies['tag_indices'].apply(
        lambda x: x + [padding_idx]*(max_tags - len(x))
    )
    tag_indices_array = np.vstack(movies['tag_indices_padded'].values)
    tag_indices_tensor = torch.tensor(tag_indices_array, dtype=torch.long)

    # 3.4 Encode users/movies
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_mapping = {id_: idx for idx, id_ in enumerate(user_ids)}
    movie_mapping = {id_: idx + len(user_mapping) for idx, id_ in enumerate(movie_ids)}

    ratings['user_idx'] = ratings['userId'].map(user_mapping)
    ratings['movie_idx'] = ratings['movieId'].map(movie_mapping)

    movies = movies[movies['movieId'].isin(movie_ids)].reset_index(drop=True)

    num_users = len(user_mapping)
    num_movies = len(movie_mapping)

    # 3.5 User features (random embeddings + zero padding)
    user_embedding_dim = 128
    user_random = torch.randn(num_users, user_embedding_dim)

    # 3.6 Movie features (random embeddings + PCA on genres)
    movie_embedding_dim = 128
    movie_random = torch.randn(num_movies, movie_embedding_dim)
    movie_content = genre_tensor[:num_movies]

    scaler = StandardScaler()
    movie_content_np = movie_content.numpy()
    movie_content_scaled = scaler.fit_transform(movie_content_np)
    movie_content_scaled = torch.tensor(movie_content_scaled, dtype=torch.float)

    pca = PCA(n_components=10)
    movie_content_reduced = torch.tensor(pca.fit_transform(movie_content_scaled.numpy()), dtype=torch.float)

    movie_features = torch.cat([movie_random, movie_content_reduced], dim=1)

    # Combine user + movie features => final x
    user_padding = torch.zeros(num_users, movie_content_reduced.size(1))
    user_features = torch.cat([user_random, user_padding], dim=1)
    x = torch.cat([user_features, movie_features], dim=0)

    # 3.7 Normalize node features
    scaler_nodes = StandardScaler()
    x_np = x.numpy()
    x_scaled = scaler_nodes.fit_transform(x_np)
    x = torch.tensor(x_scaled, dtype=torch.float)

    # 3.8 Build edge index
    edge_index = torch.tensor(
        [ratings['user_idx'].values, ratings['movie_idx'].values],
        dtype=torch.long
    )

    data = Data(x=x, edge_index=edge_index)

    # 3.9 (Optional) If you need edge_weight for some reason, replicate that code too.
    return data, tag_indices_tensor, user_mapping, movie_mapping

# -------------------------------------------------------
# 4. Main Demo
# -------------------------------------------------------
if __name__ == "__main__":
    # Example usage:
    # 4.1. Prepare Data
    dataset_path = "."  # CSVs are in the same folder
    data, tag_indices_tensor, user_mapping, movie_mapping = prepare_data_inference(dataset_path)

    # 4.2. Load best model with known hyperparams
    best_model_path = "best_gnn_vae_model1.pth"

    # The input_dim in train.py is 128 (random) + 10 (PCA) = 138
    input_dim = 138
    hidden_channels = 192
    latent_dim = 73
    decoder_channels = 202
    dropout = 0.22525254000173417
    num_tags = 1589  # forced / known from your dataset

    model = load_best_model(
        model_path=best_model_path,
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        latent_dim=latent_dim,
        decoder_channels=decoder_channels,
        num_tags=num_tags,
        dropout=dropout
    )

    # 4.3. Run inference on all edges
    reconstructed_ratings, mu, logvar = run_inference(model, data)

    # 4.4. Example: Show the first 10 reconstructed edge ratings
    print("Sample of reconstructed (scaled) ratings on the first 10 edges:")
    print(reconstructed_ratings[:10])

    # 4.5. Or pick a user index, build an edge subset, etc.
    some_user_idx = 0  # e.g. first user
    row, col = data.edge_index
    user_edge_mask = (row == some_user_idx)
    user_edge_index = data.edge_index[:, user_edge_mask]
    user_recon, _, _ = run_inference(model, data, edge_subset=user_edge_index)
    print(f"\nReconstructed ratings for user_idx={some_user_idx}:")
    print(user_recon)

    # Here you can do post-processing or re-map 'movie_idx' back to actual movie titles,
    # sort by highest rating, etc.
