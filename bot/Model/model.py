import pandas as pd
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import datetime
from . import falcon_model

nltk.download('popular', quiet=True) # for downloading packages
import warnings

# Ignore all warnings
warnings.simplefilter("ignore")


# Setting stop words from nltk library to english
stopwords = stopwords.words('english')

# Reading the dataset
df = pd.read_csv('openfabric-ai-software-engineer/bot/Dataset/train.csv')


# Tokenizing the words
def tokenizer(text):
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

# Cleaning the input (removing punctuations, digits, and converting to lower case)
# may not be used
def clean_input(input):
    input = input.lower()
    input = input.translate(str.maketrans('', '', string.digits))
    input = input.translate(str.maketrans('', '', string.punctuation))
    input = input.strip()
    return input

# Solving the question using cosine similarity and tfidf vectorizer 
vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords, lowercase=True)
matrix = vectorizer.fit_transform(tuple(df['question']))
def solve(question):
    question_vector = vectorizer.transform([question])
    cos_results = cosine_similarity(question_vector, matrix)
    index = np.argmax(cos_results, axis = None)
    return cos_results, index

# Getting the response 
def get_response(question):
    # question = clean_input(question)
    question = question.lower()

    # Getting the cosine similarity results and index of the most similar question
    similarity_results, index_res = solve(question)
    response = ''
    if similarity_results[0, index_res] < 0.6:
        response = falcon_model.generate_response(question)
    else:
        response =  df['answer'][index_res]

     # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")    
    # Write the question and response to a file
    with open('openfabric-ai-software-engineer/bot/Logs/queries.txt', 'a') as f:
        f.write(current_time + ',' + str(similarity_results[0, index_res]) + ',' )
        f.write( question +','+ response  +'\n')
    return response