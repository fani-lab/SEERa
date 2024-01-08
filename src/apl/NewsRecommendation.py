import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import params
from cmn import Common as cmn

def main(user_clusters, user_final_interests, news_table):
    merged_user_df = pd.merge(user_clusters, user_final_interests, on='UserId')
    weighted_avg_interests = merged_user_df.groupby('Community')['FinalInterests'].apply(lambda x: np.mean(np.vstack(x), axis=0).tolist()).reset_index()
    columns = [f'Interest_{i + 1}' for i in range(len(weighted_avg_interests['FinalInterests'].iloc[0]))]
    weighted_avg_interests_df = pd.DataFrame(weighted_avg_interests['FinalInterests'].tolist(), columns=columns)
    community_interests_df = pd.concat([weighted_avg_interests['Community'], weighted_avg_interests_df], axis=1)
    community_interests_matrix = community_interests_df.iloc[:, 1:].values
    try:
        from ast import literal_eval
        news_table['TopicInterests'] = news_table['TopicInterests'].apply(literal_eval)
    except:
        news_table['TopicInterests'] = news_table['TopicInterests']
    news_interests_matrix = news_table['TopicInterests'].tolist()
    similarities = cosine_similarity(community_interests_matrix, news_interests_matrix)
    community_recommendations = pd.DataFrame(similarities, columns=news_table['NewsId'].tolist()).join(community_interests_df['Community'])
    community_recommendations_main = pd.DataFrame([community_recommendations.columns[row.argsort()[-params.apl['topK']:][::-1]] for row in similarities],columns=[f'TopNews_{i + 1}' for i in range(params.apl['topK'])]).join(community_recommendations[['Community']])
    assert np.all(np.max(similarities, axis=1) == community_recommendations[community_recommendations_main['TopNews_1']].values.diagonal()), "Assertion failed: Maximum values don't match"
    community_recommendations_main.to_csv(f"../output/{params.apl['path2save']}/community_recommendations.csv")
    user_community_recommendations = pd.merge(merged_user_df, community_recommendations_main, on='Community')
    user_community_recommendations.to_csv(f"../output/{params.apl['path2save']}/user_community_recommendations.csv")
    user_interests_matrix = np.vstack(merged_user_df['FinalInterests'].apply(np.array))
    user_news_similarities = cosine_similarity(user_interests_matrix, news_interests_matrix)
    user_recommendations = pd.DataFrame(user_news_similarities, columns=news_table['NewsId'].tolist()).join(merged_user_df['UserId'])
    user_recommendations_main = pd.DataFrame([user_recommendations.columns[row.argsort()[-params.apl['topK']:][::-1]] for row in user_news_similarities],columns=[f'TopNews_{i + 1}' for i in range(params.apl['topK'])]).join(user_recommendations[['UserId']])
    assert np.all(np.max(user_news_similarities, axis=1) == user_recommendations[user_recommendations_main['TopNews_1']].values.diagonal()), "Assertion failed: Maximum values don't match"
    user_recommendations_main.to_csv(f"../output/{params.apl['path2save']}/user_recommendations.csv")
    return community_recommendations_main, user_community_recommendations, user_recommendations_main


