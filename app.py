import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Page configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

@st.cache_resource
def load_and_prepare_data():
    """Loads dataset, cleans text, computes TF-IDF and similarity matrix."""
    # For demonstration, you can load a sample or your full CSV file here.
    # Ensure your CSV has 'title' and 'overview' (or 'tags' / 'genres') columns.
    try:
        df = pd.read_csv('tmdb_5000_movies.csv')
    except FileNotFoundError:
        # Fallback sample DataFrame if file isn't uploaded yet
        data = {
            'title': ['The Dark Knight', 'Inception', 'Interstellar', 'The Matrix', 'Avatar'],
            'overview': [
                'Batman raises the stakes in his war on crime with the help of Lt. Jim Gordon and District Attorney Harvey Dent.',
                'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task.',
                'Interstellar chronicles the adventures of a group of explorers who make use of a newly discovered wormhole.',
                'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
                'A paraplegic marine dispatched to the moon Pandora on a unique mission becomes torn between following orders and protecting an alien world.'
            ]
        }
        df = pd.DataFrame(data)

    # Task 1 & 2: Handle missing values and text preprocessing
    df['overview'] = df['overview'].fillna('')
    
    def clean_text(text):
        text = text.lower() # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation and special characters
        return text

    df['clean_text'] = df['overview'].apply(clean_text)

    # Task 3: TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])

    # Task 4: Cosine Similarity Computation
    # Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space,
    # capturing how similar the items are regardless of their magnitude (document length).
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, similarity_matrix

# Load data and similarity matrix
df, similarity = load_and_prepare_data()

# Task 5: Recommendation Function
def recommend(item_name, top_n=5):
    if item_name not in df['title'].values:
        return []
    
    idx = df[df['title'] == item_name].index[0]
    distances = similarity[idx]
    
    # Sort items based on similarity scores in descending order
    sorted_indices = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)
    
    # Get top N items (excluding the item itself at index 0)
    recommended_indices = [i[0] for i in sorted_indices[1:top_n+1]]
    
    return df['title'].iloc[recommended_indices].tolist()

# Task 6: Streamlit UI Interface
st.title("🎬 Content-Based Movie Recommendation System")
st.write("Select a movie from the dropdown below to get top recommendations based on content similarity.")

selected_movie = st.selectbox(
    "Choose a movie:",
    options=df['title'].values
)

if st.button("Generate Recommendations", type="primary"):
    if selected_movie:
        recommendations = recommend(selected_movie, top_n=5)
        
        st.subheader(f"Top Recommendations for: *{selected_movie}*")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
    else:
        st.warning("Please select a valid movie.")