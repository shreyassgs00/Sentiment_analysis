import nltk
import pandas as pd
import numpy as np
import regex as re

nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from IPython.display import display

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score

data = pd.read_csv('datasets/AMD.csv')
stopwords_model = stopwords.words('english')
text = "".join(title for title in data.Headline)
wordcloud = WordCloud(stopwords=stopwords_model).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print(data.Headline)

wordfreq = {}
for title in data.Headline:
    words = title.split(' ')
    for word in words:
        if word in wordfreq:
            wordfreq[word] +=1
        else:
            wordfreq[word] = 1

#vocabulary = sorted(set(word for sentence in data.Headline for word in sentence.split()))

#Counting of text
vec = CountVectorizer(binary = False)
vec.fit(data.Headline.tolist())
output = pd.DataFrame(vec.transform(data.Headline.tolist()).toarray(), columns=sorted(vec.vocabulary_.keys()))
display(output)

#TFIDF counting
vec = TfidfVectorizer()
vec.fit(data.Headline.tolist())
output = pd.DataFrame(vec.transform(data.Headline.tolist()).toarray(), columns=sorted(vec.vocabulary_.keys()))
features = vec.transform(data.Headline.tolist())
display(output)
print(features)

class_clusters = MiniBatchKMeans(n_clusters = 5)
class_clusters.fit(features)

pca = PCA(n_components = 2)
reduced_features = pca.fit_transform(features.toarray())
reduced_cluster_centers = pca.transform(class_clusters.cluster_centers_)

plt.scatter(reduced_features[:,0], reduced_features[:,1], c=class_clusters.predict(features))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
plt.show()

score_homogeneity = homogeneity_score(data.Headline, class_clusters.predict(features))
print(score_homogeneity)


#Nearest neighbour search
