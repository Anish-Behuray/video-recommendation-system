from content_based import content_based_recommendation
from collaborative_filtering import collaborative_filtering


def hybrid_recommendation(user_id, user_interactions, all_posts_df, user_video_matrix, svd_matrix):
    content_recs = content_based_recommendation(user_interactions, all_posts_df)
    collaborative_recs = collaborative_filtering(user_id, svd_matrix, user_video_matrix)

    hybrid_recs = pd.concat([content_recs, collaborative_recs])
    return hybrid_recs.drop_duplicates().head(10)
