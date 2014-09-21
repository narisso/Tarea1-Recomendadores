import numpy as np
from scipy.stats.stats import pearsonr
from .base_similarity import BaseSimilarity
from itertools import imap

class WeightedPearsonUserSimilarity(BaseSimilarity):

    def __init__(self, model, trust_level=10.0):

        self._model = model
        self._trust = float(trust_level)

    def get_similarity(self, user1, user2):

        common, vector1, vector2  = self.get_common_items(user1, user2)
        trustworthiness = 1

        if len(common) < self._trust:
            trustworthiness = float(len(common)) / self._trust

        corr = 0

        if len(common) == 1:
            corr = 0
        elif len(common) > 1:
            corr = self.pearson_corr(vector1,vector2)
        
        if np.isnan(corr):
            corr = 0

        sim = trustworthiness * corr
        
        return sim



    def pearson_corr(self, x, y):
        # Assume len(x) == len(y)
        n = len(x)
        sum_x = float(sum(x))
        sum_y = float(sum(y))
        sum_x_sq = sum(map(lambda x: pow(x, 2), x))
        sum_y_sq = sum(map(lambda x: pow(x, 2), y))
        psum = sum(imap(lambda x, y: x * y, x, y))
        num = psum - (sum_x * sum_y/n)
        den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
        if den == 0: return 0
        return num / den

    def get_common_items(self, user1, user2):

        if(user1 == user2):
            return np.asarray([]), np.asarray([]), np.asarray([])

        prefs1 = self._model.preference_values_from_user(user1).keys()
        prefs2 = self._model.preference_values_from_user(user2).keys()

        # idx = np.intersect1d( np.where( prefs1[:] > 0)[0], np.where(prefs2[:] > 0)[0])
        comon_prefs = np.intersect1d(prefs1, prefs2)

        if( len(comon_prefs) > 0):
            user1_idx = self._model.user_id_to_index(user1)
            user2_idx = self._model.user_id_to_index(user2)
            vector1 = [self._model.preference_value_from_index(user1_idx,i) for i in comon_prefs]
            vector2 = [self._model.preference_value_from_index(user2_idx,i) for i in comon_prefs]

            return np.asarray(comon_prefs), np.asarray(vector1), np.asarray(vector2)
        else:
            return np.asarray([]), np.asarray([]), np.asarray([])
