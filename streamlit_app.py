import streamlit as st
import torch
import numpy as np
import pandas as pd
import os

from torch_geometric.data import Data

# If GNN_VAE is defined in train.py, you can do:
from train import GNN_VAE

##############################################################################
# 1) Data loading/prep to match training. We'll replicate minimal logic here.
##############################################################################

@st.cache_resource
def load_data_and_model():
    """
    Returns:
      - model: The trained GNN_VAE model (CPU)
      - data: PyG Data object with .x and .edge_index (same as training)
      - user_mapping: dict (userId -> user_idx)
      - movie_mapping: dict (movieId -> global movie_idx)
      - reverse_movie_mapping: dict (movie_idx -> movieId)
      - movies_df: DataFrame with 'movieId' and 'title'
    """

    # Option A: Directly use relative paths if CSVs are in the same folder
    ratings = pd.read_csv("ratings.csv")
    movies_df = pd.read_csv("movies.csv")

    # (If you need tags.csv, you can do the same: tags = pd.read_csv("tags.csv"))

    # ------------------------------------------------
    # Alternatively, compute the script's directory:
    #
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    # ratings_path = os.path.join(script_dir, "ratings.csv")
    # movies_path = os.path.join(script_dir, "movies.csv")
    # ratings = pd.read_csv(ratings_path)
    # movies_df = pd.read_csv(movies_path)
    # ------------------------------------------------

    # Minimal user/movie mapping logic:
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()

    user_mapping = {id_: idx for idx, id_ in enumerate(user_ids)}
    movie_mapping = {id_: idx + len(user_mapping) for idx, id_ in enumerate(movie_ids)}

    # Reverse mapping for movies: (movie_idx -> movieId)
    reverse_movie_mapping = {v: k for k, v in movie_mapping.items()}

    # Add columns for user_idx and movie_idx
    ratings['user_idx'] = ratings['userId'].map(user_mapping)
    ratings['movie_idx'] = ratings['movieId'].map(movie_mapping)

    # Build edge_index
    edge_index = torch.tensor(
        [ratings['user_idx'].values, ratings['movie_idx'].values],
        dtype=torch.long
    )

    # Minimal node feature approach (random features for demonstration)
    num_users = len(user_mapping)
    num_movies = len(movie_mapping)
    embedding_dim = 138  # e.g., 128 + 10 from your training

    node_features = torch.randn(num_users + num_movies, embedding_dim)

    data = Data(
        x=node_features,
        edge_index=edge_index
    )

    # Load model
    best_model_path = "best_gnn_vae_model1.pth"

    input_dim = 138
    hidden_channels = 192
    latent_dim = 73
    decoder_channels = 202
    dropout = 0.22525254000173417
    num_tags = 1589

    model = GNN_VAE(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        latent_dim=latent_dim,
        decoder_channels=decoder_channels,
        num_tags=num_tags,
        dropout=dropout
    )
    model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
    model.eval()

    return model, data, user_mapping, movie_mapping, reverse_movie_mapping, movies_df


##############################################################################
# 2) Main Streamlit App
##############################################################################

def main():
    st.title("GNN-VAE Movie Recommendation Demo")

    # Load everything
    model, data, user_mapping, movie_mapping, reverse_movie_mapping, movies_df = load_data_and_model()

    # Prompt user
    user_id_input = st.number_input("Enter User ID to get top recommendations:", min_value=1, value=1)

    if st.button("Recommend"):
        # 1) Convert user_id to user_idx
        if user_id_input not in user_mapping:
            st.warning(f"User ID {user_id_input} not found in data.")
            return
        user_idx = user_mapping[user_id_input]

        # 2) Build edge subset for this user
        row, col = data.edge_index
        user_mask = (row == user_idx)
        user_edge_index = data.edge_index[:, user_mask]  # shape [2, #edges_for_this_user]

        # 3) Run inference
        with torch.no_grad():
            recon, mu, logvar = model(data.x, user_edge_index)

        # recon is a 1D tensor of predicted ratings (scaled 0-1?), one for each edge with user_idx
        # user_edge_index[1] gives the movie_idx for each rating
        user_movie_idxs = user_edge_index[1].numpy()  # shape [num_user_edges]

        # 4) Sort top-K
        top_k = 10
        predicted_ratings = recon.numpy()
        # Sort descending
        sorted_indices = np.argsort(-predicted_ratings)
        top_indices = sorted_indices[:top_k]

        st.subheader(f"Top {top_k} Recommendations for user {user_id_input}:")
        for rank, idx in enumerate(top_indices, start=1):
            pred_score = predicted_ratings[idx]
            movie_idx = user_movie_idxs[idx]
            # Convert movie_idx -> movieId
            movie_id = reverse_movie_mapping[movie_idx]

            # Get movie title from movies_df
            filtered = movies_df[movies_df['movieId'] == movie_id]
            if len(filtered) > 0:
                title = filtered.iloc[0]['title']
            else:
                title = f"Unknown Movie ID {movie_id}"

            st.write(f"**{rank}.** {title} (Predicted Score: {pred_score:.4f})")

if __name__ == "__main__":
    main()
