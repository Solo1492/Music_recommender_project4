from flask import Flask, render_template, request
from recommend_code import recommend_songs

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return get_recommendations()
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    song_name = request.form['song_name']
    message, recommendations = recommend_songs(song_name)
    return render_template('index.html', message=message, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
