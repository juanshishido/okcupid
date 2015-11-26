import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from utils import *
import numpy as np
import pandas as pd


def run_PCA(datamatrix, filename = 'data/pca_dict_50.pkl', n_components = 50):
    '''
    Runs PCA with whitening on a data matrix of observations (rows) x features (columns)
    Saves out a pickle file of a dictionary with the fitted PCA and the reduced data
    Return the pca object and the reduced data
    '''
    
    #check that there are no NaNs in datamatrix
    if np.isnan(datamatrix).any():
        print ('data matrix has NaN entries')
        return

    #run PCA
    pca = PCA(n_components = n_components, whiten = True)
    pca.fit(datamatrix)

    #reduced data
    X_reduced = pca.transform(datamatrix)

    #save file
    pca_dict = {'pca':pca, 'X_reduced':X_reduced}
    with open(filename, 'wb') as f:
        pickle.dump(data_matrix_dense, f)
        

    print ('variance explained with {0} components: {1}'.format(n_components, pca.explained_variance_ratio_.sum()))

    return pca, X_reduced

def run_kmeans(X_reduced, range_n_clusters = [3,5,7]):
    '''
    Run kmeans using k-means++ initialization for different cluster sizes
    Print the silhouette score
    Takes in the PCA-reduced data and (optional) a range of clusters.
    Return a dict with an entry per each cluster number
    '''

    km_dict = dict()
    for n_clusters in range_n_clusters:

        # Initialize the clusterer with n_clusters value 
        kmeans = KMeans(n_clusters=n_clusters, init = 'k-means++')
        km = kmeans.fit(X_reduced)

        # Silhouette_score gives the average value for all samples.
        # Gives a perspective into the density and separation of the clusters
        silhouette_avg = metrics.silhouette_score(X_reduced, km.labels_, sample_size = 1000)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        
        km_dict[n_clusters] = km

    return km_dict
