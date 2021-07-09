import pandas as pd
import nltk
#import os
#import sys
#import regex as re
#import numpy as np
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
#from IPython.display import display

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score
from sklearn.neighbors import NearestNeighbors


def wordcloudgenerator(text):
    stopwords_model = stopwords.words('english') 
    wordcloud = WordCloud(stopwords=stopwords_model).generate(text)
    return wordcloud


def findwordfreq(data):
    wordfreq = {}
    for title in data:
        words = title.split(' ')
        for word in words:
            if word in wordfreq:
                wordfreq[word] +=1
            else:
                wordfreq[word] = 1
    return wordfreq

def textcounter_countvectorizer(data):
    vec = CountVectorizer(binary = False)
    vec.fit(data.Headline.tolist())
    output = pd.DataFrame(vec.transform(data.Headline.tolist()).toarray(), columns=sorted(vec.vocabulary_.keys()))
    return output

def textcounter_tfidf(data):
    vec = TfidfVectorizer()
    vec.fit(data.Headline.tolist())
    output = pd.DataFrame(vec.transform(data.Headline.tolist()).toarray(), columns=sorted(vec.vocabulary_.keys()))
    features = vec.transform(data.Headline.tolist())
    return (output,features)

def intensityanalyzer(data):
    results = []

    for headline in data['Headline']:
        pol_score = SIA().polarity_scores(headline) # run analysis
        pol_score['headline'] = headline # add headlines for viewing
        results.append(pol_score)

    return results

def cluster(features,data):
    class_clusters = MiniBatchKMeans(n_clusters = 5)
    class_clusters.fit(features)
    pca=PCA(n_components = 2)
    reduced_features = pca.fit_transform(features.toarray())
    reduced_cluster_centers = pca.transform(class_clusters.cluster_centers_)
    
    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=class_clusters.predict(features))
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
    plt.show()
    
    score_homogeneity = homogeneity_score(data, class_clusters.predict(features))
    return(score_homogeneity)

def near_neighbour(features,data):
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(features)
    distance, neighbour = knn.kneighbors(features, n_neighbors=2, return_distance=True) 
    for i in data:
        for input_text, distances, neighbors in zip(i, distance, neighbour):
            print("Input text = ", input_text[:200], "\n")
            for dist, neighbor_idx in zip(distances, neighbors):
                print("Distance = ", dist, "Neighbor idx = ", neighbor_idx)
                print("-"*20)
        print("="*20)
        print()
