import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
#from utils import *
from utils.hash import *
import numpy as np
import pandas as pd


def run_PCA(datamatrix, p, ppath, n_components = 50, force_update = False):
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
    with open(ppath, 'wb') as f:
        pickle.dump(pca_dict, f)

    #add or update hash
    hash_update(p, hash_get(p), force_update=force_update)

    print ('variance explained with {0} components: {1}'.format(n_components, pca.explained_variance_ratio_.sum()))

    return pca, X_reduced

def run_kmeans(X_reduced, filename = 'data/km_dict.pkl', range_n_clusters = [3,4, 5, 6, 7]):
    '''
    Run kmeans using k-means++ initialization for different cluster sizes
    Runs each kmean 50 times (using n_init argument) - takes clustering with lowest inertia
    Calculates the silhouette score 10 times (with sampling) to get a stable estimate
    Takes in the PCA-reduced data and (optional) a range of clusters.
    Save and return a dict with an entry per each cluster number
    '''

    km_dict = dict()
    for n_clusters in range_n_clusters:

        # Initialize the clusterer with n_clusters value 
        kmeans = KMeans(n_clusters=n_clusters, init = 'k-means++', n_init = 50) 
        km = kmeans.fit(X_reduced) #will take one with lowest inertia (within-cluster sum-of-squares)

        #calculates score 10 times (to get more stable estimate)
        silhouette_avg = []
        i = 0
        while i < 10:
            try:
                silhouette_avg.append(metrics.silhouette_score(X_reduced, km.labels_, sample_size = 1000))
                i+=1
            except:
                continue

        km_dict[n_clusters] = (km, np.mean(silhouette_avg), np.std(silhouette_avg))

    #save file
    
    with open(filename, 'wb') as f:
        pickle.dump(km_dict, f)
   
    
    return km_dict
