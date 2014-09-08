import numpy as np
from scipy.stats.stats import pearsonr
from .base_similarity import BaseSimilarity

class WeightedPearsonUserSimilarity(BaseSimilarity):

    def __init__(self, model, trust_level=10.0):

        self._model = model
        self._trust = float(trust_level)
        self._memory = {}

    def get_similarity(self, user1, user2):

        common, vector1,vector2  = self.get_common_items(user1, user2)

        trustworthiness = 1
        if len(common) < self._trust:
            trustworthiness = float(len(common)) / self._trust

        corr = 0

        if len(common) == 1:
            corr = 0
        elif len(common) > 1:
            corr = pearsonr(vector1,vector2)[0]
        
        if np.isnan(corr):
            corr = 0

        sim = trustworthiness * corr
        
        return sim


    def get_common_items(self, user1, user2):

        if(user1 == user2):
            return np.asarray([]), np.asarray([]), np.asarray([])

        
        prefs1 = self._model.keys_from_user(user1)
        prefs2 = self._model.keys_from_user(user2)

        # idx = np.intersect1d( np.where( prefs1[:] > 0)[0], np.where(prefs2[:] > 0)[0])
        idx = np.intersect1d( prefs1, prefs2)

        if( len(idx) > 0):
            comon_prefs = idx
            vector1 = [self._model.preference_value(user1,i) for i in comon_prefs]
            vector2 = [self._model.preference_value(user2,i) for i in comon_prefs]

            return np.asarray(comon_prefs), np.asarray(vector1), np.asarray(vector2)
        else:
            return np.asarray([]), np.asarray([]), np.asarray([])
