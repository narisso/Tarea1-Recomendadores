import numpy as np
from .base_similarity import BaseSimilarity

class AdjCosineItemSimilarity(BaseSimilarity):

    def __init__(self, model):

        self._model = model

    def get_similarity(self, item1, item2):

        common, prefs1, prefs2 = self.get_common_users(item1,item2)

        if(len(common) == 0):
            return 0

        avgs = np.empty(len(common))
        for i in range(0,len(common)):
            u = self._model.index_to_user_id(common[i])
            avgs[i] = self._model.get_user_id_avg(u)


        sim = self.adj_cosine_sim(prefs1, prefs2, avgs)

        if(np.isnan(sim)):
            sim = 0

        return sim

    def adj_cosine_sim(self, v1, v2, avgs):
        try:
            v1_adj = v1 - avgs
            v2_adj = v2 - avgs
            return np.dot(v1_adj, v2_adj) / (np.sqrt(np.dot(v1_adj, v1_adj)) * np.sqrt(np.dot(v2_adj, v2_adj)))
        except:
            return 0

    def get_common_users(self, item1, item2):

        if(item1 == item2):
            return np.asarray([]), np.asarray([]), np.asarray([])

        prefs1 = self._model.preference_values_for_item(item1).keys()
        prefs2 = self._model.preference_values_for_item(item2).keys()

        # idx = np.intersect1d( np.where( prefs1[:] > 0)[0], np.where(prefs2[:] > 0)[0])
        comon_prefs = np.intersect1d(prefs1, prefs2)

        if( len(comon_prefs) > 0):
            item1_idx = self._model.item_id_to_index(item1)
            item2_idx = self._model.item_id_to_index(item2)
            vector1 = [self._model.preference_value_from_index(i,item1_idx) for i in comon_prefs]
            vector2 = [self._model.preference_value_from_index(i,item2_idx) for i in comon_prefs]

            return np.asarray(comon_prefs), np.asarray(vector1), np.asarray(vector2)
        else:
            return np.asarray([]), np.asarray([]), np.asarray([])
