import numpy as np
import pandas as pd

import warnings
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

SEED = 42

def principle_component_analysis(dataframe, anomaly_column, description_column, n_component):

    dataframe_X = dataframe.drop(columns=[anomaly_column, description_column])
    dataframe_y = dataframe[[anomaly_column]]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe_X.to_numpy())

    pca = PCA(n_components=n_component, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    dataframe_pca = pd.DataFrame(X_pca, columns=[f"Principal_Component_{idx}" for idx in range(1, n_component+1)])
    dataframe_pca = dataframe_pca.assign(anomaly=dataframe.anomaly.values, description=dataframe.description.values)
    return dataframe_pca

def linear_discriminant_analysis(dataframe, anomaly_column, description_column, n_component):

    dataframe_X = dataframe.drop(columns=[anomaly_column, description_column])
    dataframe_y = dataframe[[anomaly_column]]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe_X.to_numpy())
    
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_scaled, dataframe_y.to_numpy().ravel())

    dataframe_lda = pd.DataFrame(X_lda, columns=[f"Linear_Discriminant_{idx}" for idx in range(1, X_scaled.shape[1]+1)])
    dataframe_lda = dataframe_lda.assign(anomaly=dataframe.anomaly.values, description=dataframe.description.values)
    return dataframe_lda

def non_negative_matrix_factorization(dataframe, anomaly_column, description_column, n_component):

    dataframe_X = dataframe.drop(columns=[anomaly_column, description_column])
    dataframe_y = dataframe[[anomaly_column]]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe_X.to_numpy())
    
    X_nmf_positive = X_scaled - X_scaled.min() # Simple way to make it non-negative for demo
    nmf = NMF(n_components=n_component, init='random', random_state=SEED, max_iter=500)
    X_nmf = nmf.fit_transform(X_nmf_positive)

    dataframe_nmf = pd.DataFrame(X_nmf, columns=[f"NMF_Component_{idx}" for idx in range(1, n_component+1)])
    dataframe_nmf = dataframe_nmf.assign(anomaly=dataframe.anomaly.values, description=dataframe.description.values)
    return dataframe_nmf

def t_distributed_stochastic_neighbor_embedding(dataframe, anomaly_column, description_column, n_component):

    dataframe_X = dataframe.drop(columns=[anomaly_column, description_column])
    dataframe_y = dataframe[[anomaly_column]]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe_X.to_numpy())
    
    tsne = TSNE(n_components=n_component, random_state=SEED, perplexity=50, n_jobs=5, method="exact")
    X_tsne = tsne.fit_transform(X_scaled)

    dataframe_tsne = pd.DataFrame(X_tsne, columns=[f"tSNE_Component_{idx}" for idx in range(1, n_component+1)])
    dataframe_tsne = dataframe_tsne.assign(anomaly=dataframe.anomaly.values, description=dataframe.description.values)
    return dataframe_tsne

def uniform_manifold_approximation_and_projection(dataframe, anomaly_column, description_column, n_component):

    dataframe_X = dataframe.drop(columns=[anomaly_column, description_column])
    dataframe_y = dataframe[[anomaly_column]]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe_X.to_numpy())

    reducer = umap.UMAP(n_components=n_component, random_state=SEED)
    X_umap = reducer.fit_transform(X_scaled)

    dataframe_umap = pd.DataFrame(X_umap, columns=[f"UMAP_Component_{idx}" for idx in range(1, n_component+1)])
    dataframe_umap = dataframe_umap.assign(anomaly=dataframe.anomaly.values, description=dataframe.description.values)
    return dataframe_umap