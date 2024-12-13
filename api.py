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
