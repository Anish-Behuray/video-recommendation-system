from sklearn.decomposition import TruncatedSVD


def collaborative_filtering(user_id, svd_matrix, user_video_matrix):
    user_idx = user_video_matrix.index.get_loc(user_id)

    # Predict ratings based on similar users
    predicted_ratings = svd_matrix[user_idx, :].dot(svd_matrix.T)
    recommended_video_idx = predicted_ratings.argsort()[-10:]

    recommended_videos = user_video_matrix.columns[recommended_video_idx]

    return recommended_videos
