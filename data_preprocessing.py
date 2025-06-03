

import gensim.downloader as api
from gensim.models import KeyedVectors


# Load and save pre-trained Word2Vec model
# word2vec_model = api.load("word2vec-google-news-300")
# word2vec_model.save("word2vec_model.bin")


# Load and save the pre-trained GloVe model
# glove_model = api.load("glove-wiki-gigaword-300")
# glove_model.save("glove_model.bin")


# # Load and save the pre-trained FastText model
# fasttext_model = api.load("fasttext-wiki-news-subwords-300")
# fasttext_model.save("fasttext_model.bin")


# Read data
df_pos = open("Train.pos", "r", encoding="latin-1").read()
df_neg = open("Train.neg", "r", encoding="latin-1").read()

# Create lists for positive and negative sentences
df_pos_list = [i for i in df_pos.split("\n") if len(i) >= 2]
df_neg_list = [i for i in df_neg.split("\n") if len(i) >= 2]

# Import required libraries
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd

# Load pre-trained Word2Vec, GloVe, and FastText models
loaded_word2vec_model = KeyedVectors.load("word2vec_model.bin")
loaded_glove_model = KeyedVectors.load("glove_model.bin")
loaded_fasttext_model = KeyedVectors.load("fasttext_model.bin")

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define a function to expand common English contractions
contractions = {
    "it’s": "it is", "it's": "it is", "don't": "do not", "i'm": "i am", "you're": "you are",
    "he's": "he is", "she's": "she is", "we're": "we are", "they're": "they are", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not", "mustn't": "must not"
}

def expand_contractions(text):
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def preprocess_text(text):
    text = text.lower()
    text = expand_contractions(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [re.sub(r'(.)\1{2,}', r'\1', word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Preprocess the text data
df_pos_preprocessed = [preprocess_text(sentence) for sentence in df_pos_list]
df_neg_preprocessed = [preprocess_text(sentence) for sentence in df_neg_list]


# Create a DataFrame for positive and negative data
positive_df = pd.DataFrame({
    'original_text': df_pos_list,
    'processed_text': df_pos_preprocessed,
    'level': 'positive'
})

negative_df = pd.DataFrame({
    'original_text': df_neg_list,
    'processed_text': df_neg_preprocessed,
    'level': 'negative'
})

# Concatenate both DataFrames
final_df = pd.concat([positive_df, negative_df], ignore_index=True)

# Save the DataFrame to a CSV file
final_df.to_csv("processed_data.csv", index=False)




# Generate Word2Vec, GloVe, and FastText embeddings
def get_sentence_embedding(sentence, model):
    words = sentence.split()
    word_embeddings = []
    
    for word in words:
        if word in model:
            word_embeddings.append(model[word])
        else:
            word_embeddings.append(np.zeros(model.vector_size))

    # Calculate the mean of the embeddings; if no embeddings, return a zero vector
    if len(word_embeddings) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(word_embeddings, axis=0)


# Generate embeddings for positive and negative sentences
# w2v_pos_embeddings = [get_sentence_embedding(sentence, loaded_word2vec_model) for sentence in df_pos_preprocessed]
# w2v_neg_embeddings = [get_sentence_embedding(sentence, loaded_word2vec_model) for sentence in df_neg_preprocessed]

# glove_pos_embeddings = [get_sentence_embedding(sentence, loaded_glove_model) for sentence in df_pos_preprocessed]
# glove_neg_embeddings = [get_sentence_embedding(sentence, loaded_glove_model) for sentence in df_neg_preprocessed]

# fasttext_pos_embeddings = [get_sentence_embedding(sentence, loaded_fasttext_model) for sentence in df_pos_preprocessed]
# fasttext_neg_embeddings = [get_sentence_embedding(sentence, loaded_fasttext_model) for sentence in df_neg_preprocessed]


# # Generate embeddings for all sentences in final_df
final_df['w2v_embedding'] = final_df['processed_text'].apply(lambda x: get_sentence_embedding(x, loaded_word2vec_model))
final_df['glove_embedding'] = final_df['processed_text'].apply(lambda x: get_sentence_embedding(x, loaded_glove_model))

final_df['fasttest_embedding'] = final_df['processed_text'].apply(lambda x: get_sentence_embedding(x, loaded_fasttext_model))


# # Save the DataFrame with embeddings to a CSV file
final_df.to_csv("text_embedding.csv", index=False)


# print(get_sentence_embedding("I am data scientist", loaded_word2vec_model))