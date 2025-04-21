import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load news data
def load_news_data(path='news.tsv'):
    news_df = pd.read_csv(path, sep='\t', header=None)
    news_df.columns = ['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    news_df['content'] = news_df['title'].fillna('') + ' ' + news_df['abstract'].fillna('')
    return news_df[['id', 'content', 'title']]

# Load behaviors data
def load_behaviors_data(path='behaviors.tsv'):
    behaviors_df = pd.read_csv(path, sep='\t', header=None)
    behaviors_df.columns = ['impression_id', 'user_id', 'time', 'clicked_news', 'impressions']
    return behaviors_df[['user_id', 'clicked_news']]

# Content-Based Recommender
class ContentBasedRecommender:
    def __init__(self, news_df):
        self.news_df = news_df
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.news_df['content'])
        self.news_index = {news_id: i for i, news_id in enumerate(self.news_df['id'])}

    def recommend(self, clicked_news_ids, top_n=10):
        indices = [self.news_index[nid] for nid in clicked_news_ids if nid in self.news_index]
        if not indices:
            return []

        user_profile = np.asarray(self.tfidf_matrix[indices].mean(axis=0))
        similarity_scores = cosine_similarity(user_profile, self.tfidf_matrix).flatten()
        recommended_indices = similarity_scores.argsort()[::-1]

        # Remove already clicked
        recommended_indices = [idx for idx in recommended_indices if self.news_df['id'].iloc[idx] not in clicked_news_ids]
        top_indices = recommended_indices[:top_n]
        return self.news_df.iloc[top_indices][['title', 'id']]

# Main Streamlit App
def main():
    st.title("📰 News Recommender System (MIND Dataset)")

    news_df = load_news_data()
    behaviors_df = load_behaviors_data()
    model = ContentBasedRecommender(news_df)

    user_ids = behaviors_df['user_id'].unique()
    user_id = st.selectbox("Select a User ID", user_ids)

    clicked_row = behaviors_df[behaviors_df['user_id'] == user_id].iloc[0]
    clicked_ids = clicked_row['clicked_news'].split()
    clicked_titles = news_df[news_df['id'].isin(clicked_ids)]['title'].tolist()

    st.markdown("### 📰 Clicked Articles")
    for title in clicked_titles:
        st.write("•", title)

    recommendations = model.recommend(clicked_ids)

    st.markdown("### ✨ Recommended Articles")
    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        for _, row in recommendations.iterrows():
            st.write("•", row['title'])

    # ======== Visualizations ========
    st.subheader("📊 Visualizations")

    # Boxplot: Article Lengths
    news_df['length'] = news_df['content'].apply(lambda x: len(str(x).split()))
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=news_df, x='length', ax=ax1)
    ax1.set_title("Distribution of Article Lengths")
    st.pyplot(fig1)

    # KMeans Clustering
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(model.tfidf_matrix)
    news_df['cluster'] = clusters

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(model.tfidf_matrix.toarray())
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1],
                    hue=news_df['cluster'], palette="tab10", ax=ax2, legend='full')
    ax2.set_title("KMeans Clusters of News Articles (PCA Projection)")
    st.pyplot(fig2)

    # Heatmap of article similarities (sample of 50)
    sample_indices = np.random.choice(news_df.index, size=50, replace=False)
    sample_sim = cosine_similarity(model.tfidf_matrix[sample_indices])
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(sample_sim, xticklabels=False, yticklabels=False, ax=ax3)
    ax3.set_title("Heatmap of Article Similarities (Sample of 50)")
    st.pyplot(fig3)

if __name__ == "__main__":
    main()
