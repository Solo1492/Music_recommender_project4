# Import dependencies
from flask import Flask, request, render_template
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle

# Load the saved KNN model
with open('knn_model_sd.pkl', 'rb') as f:
    knn = pickle.load(f)

# Load the tracks dataset
tracks = pd.read_csv('tracks.csv')

# Define the Flask app
app_sd_flask = Flask(__name__)

# Define a function to preprocess user input
def preprocess_input(input_dict):
    input_list = [float(input_dict[feature]) for feature in features]
    input_array = np.array(input_list).reshape(1, -1)
    return input_array

# Define the Flask route
@app_sd_flask.route('/')
def index():
    return render_template('index.html')

@app_sd_flask.route('/predict', methods=['POST'])
def predict():
    # Get the user input
    user_input = request.form.to_dict()
    # Preprocess the user input
    input_array = preprocess_input(user_input)
    # Use the KNN model to get the nearest neighbors
    distances, indices = knn.kneighbors(input_array)
    # Get the names of the nearest songs
    nearest_songs = tracks.iloc[indices[0]][['name', 'artist']]
    # Return the nearest songs
    return render_template('results.html', nearest_songs=nearest_songs.to_html())

if __name__ == '__main__':
    app_sd_flask.run(debug=True)