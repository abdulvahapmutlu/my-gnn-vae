
# Personalized Recommendation Systems Using GNN-VAEs

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-orange)](https://my-gnn-vae-hksclsanjm5gjhwfzruavi.streamlit.app/)

This project implements a **Personalized Recommendation System** using **Graph Neural Networks with Variational Autoencoders (GNN-VAEs)**. The system models user-item interactions as a bipartite graph and predicts missing edges, which correspond to personalized recommendations. This approach leverages state-of-the-art techniques in deep learning and graph representation to create accurate and scalable recommendation systems.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Run Locally](#run-locally)
  - [Run on Docker](#run-on-docker)
- [Streamlit App Usage](#streamlit-app-usage)
- [Training and Optimization](#training-and-optimization)
- [Inference](#inference)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction

Recommender systems are an essential part of modern online platforms. From suggesting movies to recommending products, they enhance user experience by offering personalized suggestions. This project focuses on building a scalable recommendation system using **Graph Neural Networks (GNNs)** and **Variational Autoencoders (VAEs)**. By representing user-item interactions as a graph, this system learns the latent features of users and items to predict missing links (recommendations).

The live app is hosted [here](https://my-gnn-vae-hksclsanjm5gjhwfzruavi.streamlit.app/), providing an interactive interface for exploring the recommendation system.

---

## Features

- **Graph-Based Representation**: Models user-item interactions as a bipartite graph.
- **GNN-VAE Architecture**: Combines the power of GNNs and VAEs for accurate recommendations.
- **Hyperparameter Optimization**: Uses Optuna to fine-tune model parameters.
- **Streamlit Interface**: A user-friendly web app for real-time recommendations.
- **Dockerized Deployment**: Ensures portability and ease of deployment.
- **CI/CD Pipeline**: Automates testing and deployment with GitHub Actions.
- **Scalability**: Designed to handle large-scale datasets with sparse interactions.

---

## Technologies Used

- **Programming Language**: Python 3.9
- **Machine Learning**: PyTorch, PyTorch Geometric
- **Data Visualization**: Matplotlib, Seaborn
- **Hyperparameter Optimization**: Optuna
- **Web Framework**: Streamlit
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Data Processing**: Pandas, NumPy, scikit-learn
- **Version Control**: Git

---

## System Architecture

### Workflow:

1. **Data Preparation**:
   - Load and preprocess the user-item interaction data.
   - Encode genres, tags, and metadata into graph nodes.
   - Construct a bipartite graph of users and items.

2. **Model Training**:
   - Train a GNN-VAE model to learn embeddings for users and items.
   - Use an encoder-decoder structure for link prediction.

3. **Inference**:
   - Predict missing links in the graph to recommend items to users.

4. **Interactive Interface**:
   - Provide real-time recommendations using a Streamlit app.

5. **Deployment**:
   - Use Docker and Kubernetes for scalable deployment.

---

## Model Details

The **GNN-VAE** model has the following components:

1. **Encoder**:
   - Uses GAT (Graph Attention Network) and GCN (Graph Convolutional Network) layers.
   - Maps node features to a latent space.

2. **Reparameterization**:
   - Implements the VAE trick to sample latent variables from the posterior distribution.

3. **Decoder**:
   - Reconstructs edges in the graph based on latent embeddings.
   - Predicts missing edges as potential recommendations.

4. **Tag Embeddings**:
   - Incorporates tag embeddings to enhance node features.

---

## Dataset

The dataset consists of three CSV files:

1. `ratings.csv`: Contains user-item ratings.
2. `movies.csv`: Contains metadata about movies (e.g., genres).
3. `tags.csv`: Contains user-generated tags for movies.

The data is preprocessed to construct the bipartite graph and encode user-item interactions.
### Citation

The dataset used in this project is the **MovieLens Latest Small (ml-latest-small)** dataset, provided by GroupLens. 

For more information about the dataset, visit the [GroupLens website](https://grouplens.org/datasets/movielens/).

**BibTeX Citation:**

```
@misc{movielens,
  author = {F. Maxwell Harper and Joseph A. Konstan},
  title = {The MovieLens Datasets: History and Context},
  year = {2015},
  publisher = {ACM Transactions on Interactive Intelligent Systems (TIIS)},
  url = {https://grouplens.org/datasets/movielens/}
}
```
---

## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Run Locally

1. Start the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

2. Open the app in your browser at `http://localhost:8501`.

### Run on Docker

1. Build the Docker image:

   ```bash
   docker build -t my-gnn-vae .
   ```

2. Run the container:

   ```bash
   docker run -p 8501:8501 my-gnn-vae
   ```

---

## Streamlit App Usage

1. Enter a **User ID** in the input box.
2. Click on the **Recommend** button to generate recommendations.
3. View the top recommended items along with predicted scores.

---

## Training and Optimization

1. **Train the Model**:
   - Use `train.py` to train the GNN-VAE model.
   - Hyperparameters are optimized using Optuna.

   ```bash
   python train.py
   ```

2. **Track Training Progress**:
   - View training and validation losses in real-time.

3. **Save Best Model**:
   - The trained model is saved as `best_gnn_vae_model1.pth`.

---

## Inference

Use `inference.py` to generate recommendations for specific users. Example:

```bash
python inference.py
```

Modify the script to specify user IDs or edge subsets for predictions.

---

## Deployment

1. **Kubernetes Deployment**:
   - Use `deployment.yaml` to deploy the app on a Kubernetes cluster.
   - Exposes the app via NodePort at port `30008`.

2. **CI/CD Pipeline**:
   - GitHub Actions automatically test and deploy the app.
   - Pipeline configuration is in `ci-cd.txt`.

---

## Project Structure

```
.
├── .devcontainer/          # DevContainer setup
├── deployment/             # Kubernetes deployment files
├── train.py                # Model training script
├── inference.py            # Inference script
├── streamlit_app.py        # Streamlit app
├── requirements.txt        # Project dependencies
├── ratings.csv             # Ratings data
├── movies.csv              # Movies data
├── tags.csv                # Tags data
├── best_gnn_vae_model1.pth # Trained model weights
├── ci-cd.txt               # CI/CD pipeline configuration
└── README.md               # Project documentation
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:

   ```bash
   git checkout -b feature/YourFeature
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add YourFeature"
   ```

4. Push to your branch:

   ```bash
   git push origin feature/YourFeature
   ```

5. Open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).
