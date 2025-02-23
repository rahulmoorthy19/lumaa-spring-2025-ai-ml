import pandas as pd
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

## Need it for stopwords
nltk.download('punkt')
nltk.download('stopwords')

class MovieRecommender:
    '''
    This is a movie recommender class which given the query recommends
    the top 5 movies relevant to the query

    Example Instantation
    recommender = MovieRecommender("imdb_top_1000.csv")
    '''
    def __init__(self, filename):
        ## Dataset preprocessing stopward declaration
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.load_dataset(filename)
        self.preprocess_dataset()

    def display_dataset(self):
        '''Display top 5 rows of dataset'''
        print(self.movies_summary.head())
    
    def load_dataset(self, filename):
         '''Loading the dataset and extracting the relevant column'''
         self.movies_summary = pd.read_csv(filename, index_col=False)[["Series_Title", "Overview"]]
    
    def preprocess_dataset(self):
        '''Preprocessing dataset and converting to TFIDF features'''
        self.movies_summary["Overview"] = self.movies_summary["Overview"].apply(self.preprocess_text)
        self.movies_feature_vectors = self.vectorizer.fit_transform(self.movies_summary["Overview"])

    def preprocess_text(self, text):
        '''Preprocess text by punctuation, whitespace removal, lowercase conversion
        and stop word removal'''
        tokens = word_tokenize(text.lower())  
        tokens = [word for word in tokens if word not in string.punctuation] 
        tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

    def get_TFIDF_feature(self, text):
        '''Get features of query'''
        return self.vectorizer.transform([text])

    def compute_similarity(self, query_vector):
        '''Computing similarity of query with the dataset'''
        return cosine_similarity(query_vector, self.movies_feature_vectors)
    
    def get_top_N(self, similarity_measure, N):
        '''Getting top N entries based on similarity'''
        top_n_movies = np.argsort(similarity_measure)[0][-N:][::-1]
        return self.movies_summary.iloc[top_n_movies][["Series_Title"]], similarity_measure[0][top_n_movies]
    
    def recommend(self, description, N = 5):
        '''Recommendation pipeline which returns the top 5 recommendation
           Input: description
           Output: Top 5 recommendation and similarity score'''
        preprocessed_query = self.preprocess_text(description)
        query_feature = self.get_TFIDF_feature(preprocessed_query)
        movies_similarity = self.compute_similarity(query_feature)
        recommended_movies, recommended_movies_similarity = self.get_top_N(movies_similarity, N)
        self.display_recommendation(np.array(recommended_movies.values)[:,0], recommended_movies_similarity)
    
    def display_recommendation(self, recommendation, similarity):
        '''Displaying the table consiting of top 5 recommendation and similarity'''
        print("Your top 5 recommended movies are:")
        recommended_movies_table = pd.DataFrame({
            'Movie Title': recommendation,
            'Similarity Score': similarity 
            })
        print(recommended_movies_table)

if __name__ == '__main__':
    recommender = MovieRecommender("imdb_top_1000.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("query_description", type=str, help="Enter the movie description to query")
    args = parser.parse_args()
    recommender.recommend(args.query_description)