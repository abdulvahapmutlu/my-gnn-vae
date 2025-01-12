# train.py

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import joblib

# ----------------------
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore")

# ----------------------
# 1. Data Preparation
# ----------------------

# You can modify this path to point to your local dataset directory.
dataset_path = "/app"

# Load data
ratings = pd.read_csv(f"{dataset_path}/ratings.csv")
movies = pd.read_csv(f"{dataset_path}/movies.csv")
tags = pd.read_csv(f"{dataset_path}/tags.csv")

# Verify columns
assert 'movieId' in movies.columns and 'genres' in movies.columns, "movies.csv missing required columns."
assert 'userId' in ratings.columns and 'movieId' in ratings.columns and 'rating' in ratings.columns, "ratings.csv missing required columns."

# 1.1 Encode Genres
genres = set()
for g_list in movies['genres'].str.split('|'):
    genres.update(g_list)
genres.discard('(no genres listed)')
genre_list = sorted(list(genres))
mlb = MultiLabelBinarizer(classes=genre_list)
genre_features = mlb.fit_transform(movies['genres'].str.split('|'))
genre_tensor = torch.tensor(genre_features, dtype=torch.float)

# 1.2 Process Tags
unique_tags = tags['tag'].unique()
tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
num_tags = len(tag_to_idx)
padding_idx = num_tags

tags_grouped = tags.groupby('movieId')['tag'].apply(list).reset_index()
movies = movies.merge(tags_grouped, on='movieId', how='left')
movies['tag'] = movies['tag'].apply(lambda x: x if isinstance(x, list) else [])
movies['tag_indices'] = movies['tag'].apply(lambda x: [tag_to_idx[t] for t in x if t in tag_to_idx])

max_tags = movies['tag_indices'].apply(len).max()
movies['tag_indices_padded'] = movies['tag_indices'].apply(lambda x: x + [padding_idx] * (max_tags - len(x)))
tag_indices_array = np.vstack(movies['tag_indices_padded'].values)
tag_indices_tensor = torch.tensor(tag_indices_array, dtype=torch.long)

# 1.3 Encode Users and Movies
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_mapping = {id_: idx for idx, id_ in enumerate(user_ids)}
movie_mapping = {id_: idx + len(user_mapping) for idx, id_ in enumerate(movie_ids)}

ratings['user_idx'] = ratings['userId'].map(user_mapping)
ratings['movie_idx'] = ratings['movieId'].map(movie_mapping)

# Filter 'movies' DataFrame to only those movieIds used in ratings
movies = movies[movies['movieId'].isin(movie_ids)].reset_index(drop=True)

num_users = len(user_mapping)
num_movies = len(movie_mapping)

# 1.4 User Features
user_embedding_dim = 128
user_random = torch.randn(num_users, user_embedding_dim)

# 1.5 Movie Features
movie_embedding_dim = 128
movie_random = torch.randn(num_movies, movie_embedding_dim)

# Combine genre features
movie_content = genre_tensor[:num_movies]
scaler = StandardScaler()
movie_content_np = movie_content.numpy()
movie_content_scaled = scaler.fit_transform(movie_content_np)
movie_content_scaled = torch.tensor(movie_content_scaled, dtype=torch.float)

pca = PCA(n_components=10)
movie_content_reduced = torch.tensor(pca.fit_transform(movie_content_scaled.numpy()), dtype=torch.float)

movie_features = torch.cat([movie_random, movie_content_reduced], dim=1)

# 1.6 Combine User and Movie Features
user_padding = torch.zeros(num_users, movie_content_reduced.size(1))
user_features = torch.cat([user_random, user_padding], dim=1)

x = torch.cat([user_features, movie_features], dim=0)

# 1.7 Normalize Node Features
scaler_nodes = StandardScaler()
x_np = x.numpy()
x_scaled = scaler_nodes.fit_transform(x_np)
x = torch.tensor(x_scaled, dtype=torch.float)

# ----------------------
# 2. Graph Construction
# ----------------------
edge_index = torch.tensor([ratings['user_idx'].values, ratings['movie_idx'].values], dtype=torch.long)
edge_weight = torch.tensor(ratings['rating'].values, dtype=torch.float)
edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())

data = Data(x=x, edge_index=edge_index)

# ----------------------
# 3. Data Split
# ----------------------
num_edges = edge_index.size(1)
all_edge_indices = torch.arange(num_edges)

train_val_idx, test_idx = train_test_split(all_edge_indices.numpy(), test_size=0.1, random_state=42)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1111, random_state=42)

train_idx = torch.tensor(train_idx, dtype=torch.long)
val_idx = torch.tensor(val_idx, dtype=torch.long)
test_idx = torch.tensor(test_idx, dtype=torch.long)


# ----------------------
# 4. Model Definition
# ----------------------
class GNN_VAE(nn.Module):
    def __init__(self, input_dim, hidden_channels, latent_dim, decoder_channels, num_tags, tag_embedding_dim=50,
                 dropout=0.5):
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
        tag_embedded = self.tag_embedding(tag_indices)
        mask = (tag_indices != self.tag_embedding.padding_idx).unsqueeze(-1).float()
        tag_embedded = tag_embedded * mask
        tag_sum = tag_embedded.sum(dim=1)
        tag_count = mask.sum(dim=1)
        tag_mean = tag_sum / (tag_count + 1e-8)
        tag_reduced = self.tag_dim_reduction(tag_mean)
        return tag_reduced


# ----------------------
# 5. Loss & Metrics
# ----------------------
def vae_loss(recon, edge_weight, mu, logvar):
    recon_loss = F.mse_loss(recon, edge_weight, reduction='mean')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001 * kl_div


def precision_at_k(pred_scores, true_labels, k=10):
    top_k = torch.topk(pred_scores, k=k, largest=True).indices
    top_k_labels = true_labels[top_k]
    return top_k_labels.sum().item() / k


def ndcg_at_k(pred_scores, true_labels, k=10):
    _, indices = torch.topk(pred_scores, k=k, largest=True)
    dcg = (true_labels[indices] / torch.log2(torch.arange(2, k + 2).float())).sum().item()
    ideal = (torch.sort(true_labels, descending=True).values[:k] / torch.log2(
        torch.arange(2, k + 2).float())).sum().item()
    return dcg / ideal if ideal > 0 else 0.0


# ----------------------
# 6. Training & Evaluation
# ----------------------
def train_model(model, data, train_idx, val_idx, edge_weight, tag_indices, optimizer, scheduler, epochs=50,
                early_stopping_patience=7):
    history = {'train_loss': [], 'val_loss': [], 'precision': [], 'ndcg': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Incorporate tag embeddings (optional usage example)
        tag_features = model.incorporate_tags(tag_indices)
        # If you want to fuse tag_features with data.x, you can do so here

        recon, mu, logvar = model(data.x, data.edge_index[:, train_idx])
        loss = vae_loss(recon, edge_weight[train_idx], mu, logvar)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            recon_val, _, _ = model(data.x, data.edge_index[:, val_idx])
            val_loss = F.mse_loss(recon_val, edge_weight[val_idx], reduction='mean')
            precision = precision_at_k(recon_val, edge_weight[val_idx], k=10)
            ndcg = ndcg_at_k(recon_val, edge_weight[val_idx], k=10)

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['precision'].append(precision)
        history['ndcg'].append(ndcg)

        scheduler.step()

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {loss.item():.4f} | "
              f"Val Loss: {val_loss.item():.4f} | "
              f"Precision@10: {precision:.4f} | "
              f"NDCG@10: {ndcg:.4f}")

        # Early Stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return history


def evaluate_model(model, data, test_idx, edge_weight, tag_indices):
    model.eval()
    with torch.no_grad():
        tag_features = model.incorporate_tags(tag_indices)
        recon_test, _, _ = model(data.x, data.edge_index[:, test_idx])
        test_loss = F.mse_loss(recon_test, edge_weight[test_idx], reduction='mean')
        precision = precision_at_k(recon_test, edge_weight[test_idx], k=10)
        ndcg = ndcg_at_k(recon_test, edge_weight[test_idx], k=10)

    print(f"Test Loss: {test_loss.item():.4f}, Precision@10: {precision:.4f}, NDCG@10: {ndcg:.4f}")
    return test_loss.item(), precision, ndcg


# ----------------------
# 7. Optuna Objective
# ----------------------
def objective(trial):
    hidden_channels = trial.suggest_int('hidden_channels', 64, 256)
    latent_dim = trial.suggest_int('latent_dim', 32, 128)
    decoder_channels = trial.suggest_int('decoder_channels', 64, 256)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)

    model = GNN_VAE(
        input_dim=x.size(1),
        hidden_channels=hidden_channels,
        latent_dim=latent_dim,
        decoder_channels=decoder_channels,
        num_tags=num_tags,
        dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    history = train_model(
        model, data, train_idx, val_idx, edge_weight, tag_indices_tensor,
        optimizer, scheduler, epochs=50, early_stopping_patience=7
    )

    final_val_loss = history['val_loss'][-1]
    return final_val_loss


# ----------------------
# 8. Run Hyperparameter Tuning
# ----------------------
if __name__ == "__main__":
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    print("Starting Hyperparameter Optimization...")
    study.optimize(objective, n_trials=50, timeout=None)

    # Best trial results
    best_trial = study.best_trial
    print(f"\nBest Trial {best_trial.number} with Val Loss: {best_trial.value:.6f}")
    print(f"Parameters: {best_trial.params}")

    best_hidden_channels = best_trial.params['hidden_channels']
    best_latent_dim = best_trial.params['latent_dim']
    best_decoder_channels = best_trial.params['decoder_channels']
    best_dropout = best_trial.params['dropout']
    best_lr = best_trial.params['lr']
    best_weight_decay = best_trial.params['weight_decay']

    # Retrain best model
    best_model = GNN_VAE(
        input_dim=x.size(1),
        hidden_channels=best_hidden_channels,
        latent_dim=best_latent_dim,
        decoder_channels=best_decoder_channels,
        num_tags=num_tags,
        dropout=best_dropout
    )
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    print("\nTraining the Best Model...")
    best_history = train_model(
        best_model, data, train_idx, val_idx, edge_weight, tag_indices_tensor,
        optimizer, scheduler, epochs=100, early_stopping_patience=10
    )

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(best_history['train_loss'], label='Train Loss')
    plt.plot(best_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nEvaluating on Test Set...")
    test_loss, test_precision, test_ndcg = evaluate_model(
        best_model, data, test_idx, edge_weight, tag_indices_tensor
    )

    torch.save(best_model.state_dict(), "best_gnn_vae_model1.pth")
    print("\nBest model saved as 'best_gnn_vae_model1.pth'.")

    with open("training_history.json", "w") as f:
        json.dump(best_history, f)
    print("Training history saved as 'training_history.json'.")

    joblib.dump(study, "optuna_study1.pkl")
    print("Optuna study saved as 'optuna_study1.pkl'.")
