import regex as re
import os

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

current_path = os.path.dirname(__file__)
path_decrease = os.path.relpath('/media/shreyas/DATA/linux_files/Capstone/Sentiment_analysis/polarity_analyser/datasets/decrease',current_path)
path_increase = os.path.relpath('/media/shreyas/DATA/linux_files/Capstone/Sentiment_analysis/polarity_analyser/datasets/increase',current_path)
path_invert = os.path.relpath('/media/shreyas/DATA/linux_files/Capstone/Sentiment_analysis/polarity_analyser/datasets/invert',current_path)
path_positive = os.path.relpath('/media/shreyas/DATA/linux_files/Capstone/Sentiment_analysis/polarity_analyser/datasets/positive',current_path)
path_negative = os.path.relpath('/media/shreyas/DATA/linux_files/Capstone/Sentiment_analysis/polarity_analyser/datasets/negative',current_path)

def clean(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

def create_tokens(text):
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    new_text = (" ").join(element for element in tokens if element.lower() not in stopwords.words('english'))
    return new_text

def parts_of_speech_tagging(tokens):
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    pos_tag_list = []
    tags = pos_tag(tokens)
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            pos_tag_list.append(tuple([word, pos_dict.get(tag[0])]))
    return pos_tag_list

def lemmatize(pos_tag_list):
    wordnet_lemmatiser = WordNetLemmatizer()
    lemma_rew = " "
    for word, pos in pos_tag_list:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " "+lemma
        else:
            lemma = wordnet_lemmatiser.lemmatize(word, pos = pos)
            lemma_rew = lemma_rew + " "+lemma
    return lemma_rew
        
def create_list(file_name):
    word_list = []
    for word in file_name:
        word_list.append(word.strip('\n'))
    return word_list

def replace_in_text():
    antonyms = set()
    for synonym in wordnet.synsets(word, pos=pos):
        for lemma in synonym.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
                if len(antonyms) == 1:
                    return antonyms.pop()
                else:
                    return None

def create_replacement_patterns(text):
    replacement_patterns = [
        (r'won\'t', 'will not'),
        (r'can\'t', 'cannot'),
        (r'i\'m', 'i am'),
        (r'ain\'t', 'is not'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        (r'(\w+)\'s', '\g<1> is'),
        (r'(\w+)\'re', '\g<1> are'),
        (r'(\w+)\'d', '\g<1> would')
    ]

    class RegexpReplacer(object):
        def __init__(self, patterns=replacement_patterns):
            self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

        def replace(self, text):
            s = text
            for (pattern, repl) in self.patterns:
                s = re.sub(pattern, repl, s)
            return s

    replacer = RegexpReplacer()
    list_of_words = []
    for words in text:
        list_of_words.append(replacer.replace(words))
    return list_of_words

def find_polarity(text):
    polarity = 0
    length = len(text)
    with open(path_increase, 'r') as increase, open(path_decrease, 'r') as decrease, open(path_invert, 'r') as invert, open(path_positive, 'r') as positive, open(path_negative, 'r') as negative:
        inc_word_list = create_list(increase)
        dec_word_list = create_list(decrease)
        inv_word_list = create_list(invert)
        pos_word_list = create_list(positive)
        neg_word_list = create_list(negative)
        for index, word in enumerate(text):
            if word in inc_word_list:
                const = 1
                while const <= 3:
                    if index + const < length:
                        if text[index + const] in pos_word_list:
                            polarity += 1
                        if text[index + const] in neg_word_list:
                            polarity -= 1
                    const += 1
            if word in dec_word_list:
                const = 1
                while const <= 3:
                    if index + const < length:
                        if text[index + const] in pos_word_list:
                            polarity -= 1
                        if text[index + const] in neg_word_list:
                            polarity += 1
                    const += 1

            if word in inv_word_list:
                const = 1
                while const <= 3:
                    if index + const < length:
                        ant = replace_in_text(text[index + const])
                        if ant:
                            if ant in pos_word_list:
                                polarity += 2
                            if ant in neg_word_list:
                                polarity -= 2
                    const += 1
            if word in pos_word_list:
                polarity += 1
            if word in neg_word_list:
                polarity -= 1
        return polarity