########################### 1

import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# API headers for authorization
headers = {
    "Flic-Token": "your_flic_token_here"
}


# Function to fetch paginated data from the API
def fetch_paginated_data(url):
    page = 1
    all_data = []

    while True:
        response = requests.get(url, headers=headers, params={"page": page, "page_size": 1000})
        data = response.json()
        all_data.extend(data['data'])
        if len(data['data']) < 1000:
            break
        page += 1

    return all_data


# Fetch data and handle missing values
def preprocess_data():
    viewed_posts = fetch_paginated_data("https://api.socialverseapp.com/posts/view")
    liked_posts = fetch_paginated_data("https://api.socialverseapp.com/posts/like")
    inspired_posts = fetch_paginated_data("https://api.socialverseapp.com/posts/inspire")
    rated_posts = fetch_paginated_data("https://api.socialverseapp.com/posts/rating")
    all_posts = fetch_paginated_data("https://api.socialverseapp.com/posts/summary/get")
    users = fetch_paginated_data("https://api.socialverseapp.com/users/get_all")

    # Convert to DataFrame
    def posts_to_dataframe(posts_data):
        return pd.DataFrame(posts_data)

    # Prepare DataFrames
    viewed_df = posts_to_dataframe(viewed_posts)
    liked_df = posts_to_dataframe(liked_posts)
    inspired_df = posts_to_dataframe(inspired_posts)
    rated_df = posts_to_dataframe(rated_posts)
    all_posts_df = posts_to_dataframe(all_posts)
    users_df = pd.DataFrame(users)

    # Handle missing values
    viewed_df.fillna(viewed_df.median(), inplace=True)
    liked_df.fillna(liked_df.median(), inplace=True)

    return viewed_df, liked_df, inspired_df, rated_df, all_posts_df, users_df

########################### 2

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

########################### 3

from sklearn.decomposition import TruncatedSVD

def collaborative_filtering(user_id, svd_matrix, user_video_matrix):
    user_idx = user_video_matrix.index.get_loc(user_id)

    # Predict ratings based on similar users
    predicted_ratings = svd_matrix[user_idx, :].dot(svd_matrix.T)
    recommended_video_idx = predicted_ratings.argsort()[-10:]

    recommended_videos = user_video_matrix.columns[recommended_video_idx]

    return recommended_videos

########################### 4

from content_based import content_based_recommendation
from collaborative_filtering import collaborative_filtering


def hybrid_recommendation(user_id, user_interactions, all_posts_df, user_video_matrix, svd_matrix):
    content_recs = content_based_recommendation(user_interactions, all_posts_df)
    collaborative_recs = collaborative_filtering(user_id, svd_matrix, user_video_matrix)

    hybrid_recs = pd.concat([content_recs, collaborative_recs])
    return hybrid_recs.drop_duplicates().head(10)

########################### 5

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse

########################### 6

from fastapi import FastAPI
from typing import Optional
from hybrid_model import hybrid_recommendation
from data_preprocessing import preprocess_data

app = FastAPI()


@app.get("/feed")
def recommend_videos(username: str, category_id: Optional[str] = None, mood: Optional[str] = None):
    viewed_df, liked_df, inspired_df, rated_df, all_posts_df, users_df = preprocess_data()
    user_interactions = viewed_df[viewed_df['user_id'] == username]
    user_video_matrix = pd.pivot_table(viewed_df, index='user_id', columns='video_id', values='engagement_score',
                                       fill_value=0)
    svd_matrix = TruncatedSVD(n_components=10).fit_transform(user_video_matrix)

    recommendations = hybrid_recommendation(username, user_interactions, all_posts_df, user_video_matrix, svd_matrix)

    if category_id:
        recommendations = recommendations[recommendations['category_id'] == category_id]
    if mood:
        recommendations = recommendations[recommendations['mood'] == mood]

    return {"username": username, "recommendations": recommendations.head(10).to_dict(orient="records")}
