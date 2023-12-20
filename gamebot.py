import streamlit as st
import pandas as pd
from transformers import pipeline

# Mock data for game reviews
game_reviews_data = {
    'User_ID': [1, 2, 1, 3, 2, 3, 4, 5, 4],
    'Game_ID': [1, 1, 2, 2, 3, 3, 4, 4, 5],
    'Rating': [5, 4, 3, 4, 5, 2, 4, 3, 5],
}

# Mock data for video game catalog
game_catalog_data = {
    'Game_ID': [1, 2, 3, 4, 5],
    'Game Title': ['Game A', 'Game B', 'Game C', 'Game D', 'Game E'],
    'Platform': ['PS4', 'Nintendo Switch', 'Xbox One', 'PC', 'PS4'],
    'Genre': ['Action', 'Adventure', 'RPG', 'Simulation', 'Action'],
}

# Function to get game details based on user input
def get_game_details(game_id):
    game_info = game_catalog_data[game_catalog_data['Game_ID'] == game_id].iloc[0]
    return f"**Platform:** {game_info['Platform']}\n**Genre:** {game_info['Genre']}"

# Create a Pandas DataFrame for game reviews
reviews_df = pd.DataFrame(game_reviews_data)

# Sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Streamlit app
st.title("Game Recommendation System")

# User input for feedback
user_feedback = st.text_input("Provide your feedback on the game:")

# Perform sentiment analysis on user feedback
if user_feedback:
    sentiment_result = sentiment_analyzer(user_feedback)[0]
    st.write(f"Sentiment Analysis: {sentiment_result['label']} ({sentiment_result['score']:.2f})")

# User input for game title
selected_game_id = st.selectbox("Select a game:", game_catalog_data['Game_ID'])
selected_game_title = game_catalog_data[game_catalog_data['Game_ID'] == selected_game_id]['Game Title'].iloc[0]

# Display game details
st.subheader("Game Details:")
game_details = get_game_details(selected_game_id)
st.markdown(game_details)

# Placeholder logic for recommending games similar to the selected one
# You can replace this logic with a more sophisticated content-based recommendation approach
similar_games = [game_id for game_id in game_catalog_data['Game_ID'] if game_id != selected_game_id]
recommended_games = game_catalog_data[game_catalog_data['Game_ID'].isin(similar_games)]['Game Title'].tolist()

# Display recommendations
st.subheader("Recommendations:")
st.write("Games you might also like:")
st.write(recommended_games)

# Display game reviews
st.subheader("Game Reviews:")
st.dataframe(reviews_df)

# Placeholder links for external sources
st.subheader("External Links:")
metacritic_link = f"[Metacritic - {selected_game_title}](https://www.metacritic.com/search/game/{selected_game_title})"
st.markdown(metacritic_link)

youtube_link = f"[YouTube - {selected_game_title}](https://www.youtube.com/results?search_query={selected_game_title}+gameplay)"
st.markdown(youtube_link)

