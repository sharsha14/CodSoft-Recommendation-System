import numpy as np

user_item_matrix = np.array([
    [5, 4, 0, 0, 0, 1],
    [0, 5, 1, 0, 0, 0],
    [0, 0, 0, 5, 4, 0],
    [0, 0, 0, 0, 0, 5],
    [4, 0, 0, 0, 0, 0]
])

def recommend_movies(user_item_matrix, user_id, num_recommendations=3):
    user_similarity = np.dot(user_item_matrix, user_item_matrix[user_id]) / (
        np.linalg.norm(user_item_matrix, axis=1) * np.linalg.norm(user_item_matrix[user_id])
    )

    # Sort users by similarity in descending order
    similar_users = np.argsort(user_similarity)[::-1]

    # Get the top N similar users (excluding the user itself)
    top_users = similar_users[1:num_recommendations+1]

    # Find movies that the user hasn't rated
    unrated_movies = np.where(user_item_matrix[user_id] == 0)[0]

    # Predict movie ratings based on similar users' ratings
    predicted_ratings = np.sum(user_item_matrix[top_users][:, unrated_movies], axis=0)

    # Sort unrated movies by predicted ratings in descending order
    recommended_movies = unrated_movies[np.argsort(predicted_ratings)[::-1]]

    return recommended_movies

# Specify the user for whom you want to make recommendations (e.g., user 0)
user_id = 0

# Get movie recommendations for the specified user
recommendations = recommend_movies(user_item_matrix, user_id, num_recommendations=3)

# Print the recommended movie indices
print("Recommended movies for user", user_id, ":")
for movie_id in recommendations:
    print("Movie", movie_id)
