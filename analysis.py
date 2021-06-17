import nltk
import pandas as pd
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from IPython.display import display

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('datasets/AMD.csv')
stopwords_model = stopwords.words('english')
text = "".join(title for title in data.Headline)
wordcloud = WordCloud(stopwords=stopwords_model).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

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

vec = TfidfVectorizer()
vec.fit(data.Headline.tolist())

output = pd.DataFrame(vec.transform(data.Headline.tolist()).toarray(), columns=sorted(vec.vocabulary_.keys()))
display(output)