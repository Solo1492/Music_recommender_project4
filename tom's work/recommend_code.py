import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('database.csv')
df

#capture
song_vectorizer = CountVectorizer()
song_vectorizer.fit(df['name'])

df = df.sort_values(by=['popularity'], ascending=False).head(10000)

def get_similarities(song_name, data):
   
  # Getting vector for the input song.
  text_array1 = song_vectorizer.transform(data[data['name']==song_name][['genres', 'artists']].apply(lambda x: ' '.join(x), axis=1)).toarray()

  # Selecting numerical columns for input song.
  num_cols = ['duration_ms', 'explicit', 'danceability', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'energy', 'liveness', 'valence', 'tempo']
  num_array1 = data[data['name']==song_name][num_cols].to_numpy()

  # Initialize sim list
  sim = []

  # Loop over rows in data and calculate similarity
  for idx, row in data.iterrows():
    name = row['name']

    # Getting vector for current song.
    text_array2 = song_vectorizer.transform(data[data['name']==name][['genres', 'artists']].apply(lambda x: ' '.join(x), axis=1)).toarray()

    # Selecting numerical columns forcurrent song.
    num_array2 = data[data['name']==name][num_cols].to_numpy()

    # Calculating similarities for text as well as numeric features
    text_sim = cosine_similarity(text_array1, text_array2)[0][0]
    num_sim = cosine_similarity(num_array1, num_array2)[0][0]

    # Combine text and numeric similarities using weights
    text_weight = 0.3
    num_weight = 0.7
    total_sim = (text_weight * text_sim) + (num_weight * num_sim)

    sim.append(total_sim)

  return sim

def recommend_songs(song_name, data=df):
    try:
        # Base case
        if df[df['name'] == song_name].shape[0] == 0:
            message = 'This song is either not so popular or you have entered an invalid name. Some songs you may like:'
            suggestions = data.sort_values(by=['popularity'], ascending=False).head(5)[['name', 'artists']].apply(tuple, axis=1).tolist()
            return message, suggestions

        data['similarity_factor'] = get_similarities(song_name, data)

        # Filter out songs from the same artist
        input_artist = data[data['name'] == song_name]['artists'].iloc[0]
        data = data[data['artists'] != input_artist]

        data.sort_values(by=['similarity_factor', 'popularity'],
                        ascending=[False, False],
                        inplace=True)

        # First song will be the input song itself as the similarity will be highest.
        recommendations = data[['name', 'artists']][1:6].apply(tuple, axis=1).tolist()
        return None, recommendations
    except ValueError:
        print('An error occurred while trying to recommend songs.')
