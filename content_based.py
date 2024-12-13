from sklearn.metrics.pairwise import cosine_similarity


def content_based_recommendation(user_interactions, all_posts_df):
    user_likes = user_interactions['video_id'].tolist()

    # Filter posts based on user interactions
    relevant_posts = all_posts_df[all_posts_df['video_id'].isin(user_likes)]

    # Create feature matrix from metadata
    content_matrix = all_posts_df[['tags']]  # Use more features as needed

    # Cosine similarity between user preferences and posts
    similarity_matrix = cosine_similarity(content_matrix)

    recommendations = similarity_matrix.argsort()[:, -10:]
    recommended_posts = all_posts_df.iloc[recommendations.flatten()]

    return recommended_posts
