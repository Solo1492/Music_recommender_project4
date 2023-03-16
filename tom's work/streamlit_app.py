import streamlit as st
from recommend_code import recommend_songs

st.set_page_config(page_title="Song Recommender", page_icon=None, layout='centered', initial_sidebar_state='auto')

st.title("Song Recommender")

song_name = st.text_input("Enter a song name:")

if song_name:
    message, recommendations = recommend_songs(song_name)

    if message:
        st.warning(message)

    if recommendations:
        st.subheader("Recommended Songs:")

        for song, artist in recommendations:
            st.markdown(f"**Song Name:** {song}<br>**Artist:** {artist}", unsafe_allow_html=True)
