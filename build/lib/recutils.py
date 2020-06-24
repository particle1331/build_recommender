from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import json
import joblib


class PopularityRecommender:
    '''Recommend the most popular products items regardless of user purchase history.'''
    
    def __init__(self):
        self.MODEL_NAME = 'Popularity'
        
    def fit(self, train_interactions):
        train_interactions.qty = np.log2(1 + train_interactions.qty) # smoothen interactions
        self.global_pop = pd.DataFrame(train_interactions.groupby('productid').qty.sum()).sort_values('qty', ascending=False).reset_index()
        self.user_interactions = train_interactions.groupby('loyalty').productid.agg(set)

    def get_model_name(self):
        return self.MODEL_NAME
    
    def recommend_items(self, user):
        return self.global_pop['productid'].tolist()
    
    def _get_interacted_items(self, user):
        return self.user_interactions.get(user, {})


class CFRecommender_KNN:
    '''Item-item nearest neighbors collaborative filtering.'''
    
    def __init__(self, k_neighbors=5):
        self.MODEL_NAME = 'KNN collaborative filtering'
        self.k = k_neighbors
        
    def fit(self, train_interactions):
        self.pop_model = PopularityRecommender()
        self.pop_model.fit(train_interactions)
        train_interactions.qty = np.log2(1 + train_interactions.qty)
        self.user_interactions = train_interactions.groupby('loyalty').productid.agg(set)
        self.utility_matrix = train_interactions.groupby(['loyalty','productid']).qty.sum().reset_index()\
                                                .pivot(index='productid', 
                                                       columns='loyalty', 
                                                       values='qty').fillna(0)
        self.similarity_matrix = pd.DataFrame(
                                    cosine_similarity(MinMaxScaler().fit_transform(self.utility_matrix.values)),
                                    index=self.utility_matrix.index, 
                                    columns=self.utility_matrix.index)
        
        # memoizer - move all computation to train time; at the expense of memory.
        self.recommendations = {}
        
    def recommend_items(self, user):
        if user not in self.utility_matrix.columns:
            return self.pop_model.recommend_items(user)
        
        if user in self.recommendations:
            return self.recommendations[user].sort_values(ascending=False).index.tolist()
        
        self.recommendations[user] = self.utility_matrix[user]
        for item in self.recommendations[user].index:
            sim = self.similarity_matrix[item].iloc[np.argpartition(-self.similarity_matrix[item].values, self.k)[:self.k]]
            if sum(sim.values == 0):
                self.recommendations[user].loc[item] = 0
            else:
                # smoothen by adding a constant
                self.recommendations[user].loc[item] = np.dot(0.01 + self.utility_matrix[user].loc[list(sim.index)].values, sim.values) / sum(sim.values)
        return self.recommend_items(user)

    def _get_interacted_items(self, user):
        return self.user_interactions.get(user, {})
    
    def get_model_name(self):
        return self.MODEL_NAME


