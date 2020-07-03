from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import json


class PopularityRecommender:
    '''Recommend the most popular products items regardless of user purchase history.'''
    
    def __init__(self, franchise_inverse, store_inverses):
        self.MODEL_NAME = 'Popularity'
        self.franchise_inverse = franchise_inverse
        self.store_inverses = store_inverses
        
    def fit(self, train_interactions):
        train_interactions.qty = np.log2(1 + train_interactions.qty) # smoothen interactions
        self.pop = pd.DataFrame(train_interactions.groupby('productid').qty.sum()).sort_values('qty', ascending=False).reset_index()
        self.user_interactions = train_interactions.groupby('loyalty').productid.agg(set)

    def get_model_name(self):
        return self.MODEL_NAME
    
    def recommend_items(self, user, waiterproid):
        user = str(user)
        store_inverse = self.store_inverses[waiterproid]
        franchise_inverse = self.franchise_inverse
        recommendations = []
        for fid in self.pop['productid'].tolist():
            name = franchise_inverse[fid]
            if name in store_inverse.keys():
                recommendations.append(store_inverse[name])
        return recommendations
    
    def _get_interacted_items(self, user):
        return self.user_interactions.get(user, {})


class CFRecommender_KNN:
    '''Item-item nearest neighbors collaborative filtering.'''
    
    def __init__(self, franchise_inverse, store_inverses, k_neighbors=5):
        self.MODEL_NAME = 'KNN collaborative filtering'
        self.k = k_neighbors
        self.franchise_inverse = franchise_inverse
        self.store_inverses = store_inverses
        self.pop_model = PopularityRecommender(franchise_inverse, store_inverses)
        
    def fit(self, train_interactions):
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
        
        
    def recommend_items(self, user, waiterproid):
        user = str(user)
        if user not in self.utility_matrix.columns:
            return self.pop_model.recommend_items(user, waiterproid)
                
        ranking = self.utility_matrix[user]
        for item in ranking.index:
            sim = self.similarity_matrix[item].iloc[np.argpartition(-self.similarity_matrix[item].values, self.k)[:self.k]]
            if sum(sim.values) == 0:
                ranking.loc[item] = 0
            else:
                # smoothen by adding a constant
                ranking.loc[item] = np.dot(0.01 + self.utility_matrix[user].loc[list(sim.index)].values, sim.values) / sum(sim.values)
        
        store_inverse = self.store_inverses[waiterproid]
        franchise_inverse = self.franchise_inverse
        recommendations = []
        for fid in ranking.sort_values(ascending=False).index.tolist():
            name = franchise_inverse[fid]
            if name in store_inverse.keys():
                recommendations.append(store_inverse[name])
        return recommendations
        

    def _get_interacted_items(self, user):
        return self.user_interactions.get(user, {})
    
    def get_model_name(self):
        return self.MODEL_NAME