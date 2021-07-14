import pandas as pd
#import nltk
#import os
#import sys
#import numpy as np
import matplotlib.pyplot as plt


import analysis


if __name__ == '__main__':
	data = pd.read_csv("datasets/AMD.csv")
	text = "".join(title for title in data.Headline)
	
	#stop words generator and word cloud

	wordcloud=analysis.wordcloudgenerator(text)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()

	#find word frequency
	wordfreq=analysis.findwordfreq(data.Headline)
	print(wordfreq)
	
	#Counting of text	
	out_vectorizer=analysis.textcounter_countvectorizer(data)
	print(out_vectorizer)

	#TFIDF
	out_tfidf,feature_tfidf=analysis.textcounter_tfidf(data)
	print(feature_tfidf)
	print(out_tfidf)
	
	#K-MEANS CLUSTERING
	score_homogenity=analysis.cluster(feature_tfidf,data.Headline)
	#print(score_homogenity)

	#Nearest Neighbour search
	analysis.near_neighbour(feature_tfidf,data.Headline.tolist())

	#intensity analyzer
	results = analysis.intensityanalyzer(data)
	print(results)

