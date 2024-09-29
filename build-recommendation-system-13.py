# Build Recommendation System
# Implement a recommendation system using collaborative filtering, content-based filtering, or a hybrid approach to suggest products or content based on user behavior.
# pip install scikit-surprise
# Extending to Content-Based Filtering
# Hybrid Recommendation System
# To combine collaborative filtering and content-based filtering, you can average the predicted ratings from both approaches or implement a weighted average system.
# Hybrid Approach: Combine the two by averaging the ratings and similarities, and recommend the highest-rated items.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

# Load a dataset with movies and their descriptions
# Assume the file has columns 'movieId', 'title', 'description'
movies = pd.read_csv('movies.csv')

# Fit a TF-IDF vectorizer on movie descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['description'])

# Calculate similarity between movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get content-based recommendations


def get_content_based_recommendations(movie_title, movies, cosine_sim, n_recommendations=5):
    # Get the index of the movie that matches the title
    idx = movies.index[movies['title'] == movie_title][0]

    # Get pairwise similarity scores for all movies with this movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top n most similar movies
    sim_scores = sim_scores[1:n_recommendations+1]

    # Get the movie titles of the top n similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]


# Example: Get top 5 content-based recommendations for a movie
recommendations = get_content_based_recommendations(
    'Toy Story', movies, cosine_sim)
print(recommendations)
