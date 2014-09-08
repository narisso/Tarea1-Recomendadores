import numpy as np
from .base_similarity import BaseSimilarity

class CosineUserSimilarity(BaseSimilarity):

    def __init__(self, model):

        self._model = model
        self._memory = {}

    def get_similarity(self, user1, user2):

        x,y,z = self.get_common_items(user1,user2)

        prefs1, prefs2 = self.get_centered_prefs(user1,user2)
        sim = self.cosine_sim(prefs1, prefs2)

        return sim


    def cosine_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))


    def get_centered_prefs(self, user1, user2):

        prefs1 = self._model.preference_values_from_user(user1).toarray()
        prefs2 = self._model.preference_values_from_user(user2).toarray()

        avg1 = self._model.get_user_id_avg(user1)
        avg2 = self._model.get_user_id_avg(user2)

        # Center Cosine
        prefs1 = prefs1 - np.int8((prefs1>0)*avg1)
        prefs2 = prefs2 - np.int8((prefs2>0)*avg2)

        return prefs1[0], prefs2[0]


    def get_common_items(self, user1, user2):

        if(user1 == user2):
            return np.asarray([]), np.asarray([]), np.asarray([])
        
        prefs1 = self._model.nonzero_indexes_from_user(user1)
        prefs2 = self._model.nonzero_indexes_from_user(user2)

        # idx = np.intersect1d( np.where( prefs1[:] > 0)[0], np.where(prefs2[:] > 0)[0])
        idx = np.intersect1d( prefs1, prefs2)

        if( len(idx) > 0):
            comon_prefs = [self._model.index_to_item_id(i) for i in idx]
            vector1 = [self._model.preference_value(user1,i) for i in comon_prefs]
            vector2 = [self._model.preference_value(user2,i) for i in comon_prefs]

            return np.asarray(comon_prefs), np.asarray(vector1), np.asarray(vector2)
        else:
            return np.asarray([]), np.asarray([]), np.asarray([])