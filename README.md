# OkCupid
An unsupervised approach for detecting differences in self-presentation on
OkCupid.

A final project for Applied Natural Language Processing, Fall 2015.

## Proposed Approach

We will use the following methods to detect differences in self-presentation.

* Pointwise mutual information for generating features
    * unigrams, bigrams, and trigrams used by at least 1% of users
* Dimensionality reduction with Principle Component Analysis
* Clustering with k-means
* Keyphrase extraction for cluster "descriptions"
* Look for differences in demographic distributions by cluster

## Essay Prompts

0: My self summary   
1: What I'm doing with my life   
2: I'm really good at   
3: The first thing people notice about me   
4: Favorite books, movies, tv, food   
5: The six things I could never do without   
6: I spend a lot of time thinking about   
7: On a typical Friday night I am   
8: The most private thing I am willing to admit   
9: You should message me if   
