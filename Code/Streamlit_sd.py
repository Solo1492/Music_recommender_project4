import streamlit as st
import pandas as pd
import pickle
from fuzzywuzzy import process
from sklearn.neighbors import KNeighborsRegressor

# load the tracks dataframe
tracks = pd.read_csv('tracks.csv')

# load the trained KNN model from pickle file
with open('../Code/sd_model.pkl', 'rb') as f:
    knn = pickle.load(f)

# define the Streamlit app
def app():
    st.title('Song Recommender')
    st.write('Enter the name of a song to get recommendations.')
    track_name = st.text_input('Song Name')
    
    if st.button('Recommend'):
        # get recommendations using the KNN model
        recommendations = get_recommendations(track_name)
        
        # display the recommendations in the Streamlit app
        st.write(recommendations)

# define the function to get recommendations
def get_recommendations(track_name, k=10):
    # use fuzzywuzzy to find the closest match for the track name in the dataframe
    matches = process.extract(track_name, tracks['name'], limit=1)
    closest_match = matches[0][0]
    
    # find the index of the closest match in the dataframe
    track_index = tracks[tracks['name'] == closest_match].index[0]
    
    # create a new dataframe with only the feature columns
    features_df = tracks.drop(['name', 'artists', 'genres'], axis=1)
    
    # get the feature vector for the track
    track_features = features_df.iloc[track_index].values.reshape(1, -1)
    
    # fit a KNN model on the feature dataframe
    knn = KNeighborsRegressor(n_neighbors=k, metric='cosine')
    knn.fit(features_df, tracks['danceability'])
    
    # find the indices of the k nearest neighbors
    distances, indices = knn.kneighbors(track_features)
    
    # create a list of recommended tracks
    recommendations = []
    for i in indices[0][1:]:
        recommendations.append(tracks.iloc[i]['name'] + " by " + tracks.iloc[i]['artists'])
    
    # return the recommended tracks
    return recommendations

if __name__ == '__main__':
    app()