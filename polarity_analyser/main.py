import polarity_analyser
import pandas as pd

def main():
    dataframe = pd.read_csv('datasets/AMD.csv')
    data = dataframe.Headline.tolist()
    
    cleaned_data = []
    for i in range(0, len(data)):
        cleaned_data.append(polarity_analyser.clean(data[i]))
        tokens = polarity_analyser.create_tokens(cleaned_data[i])
        
        stopwords_removed_text = polarity_analyser.remove_stopwords(tokens)
        stopwords_removed_text_tokens = polarity_analyser.create_tokens(stopwords_removed_text)
        
        parts_of_speech_tags = polarity_analyser.parts_of_speech_tagging(stopwords_removed_text_tokens)
        
        lemmatized_sentence = polarity_analyser.lemmatize(parts_of_speech_tags)
        lemmatized_tokens = polarity_analyser.create_tokens(lemmatized_sentence)
        lemmatized_replacement_patterns = polarity_analyser.create_replacement_patterns(lemmatized_tokens)
        
        polarity = polarity_analyser.find_polarity(lemmatized_replacement_patterns)
        print(lemmatized_sentence,"       Polarity:", polarity)

if __name__ == '__main__':
    main()