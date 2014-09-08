import numpy as np
from .base_similarity import BaseSimilarity
import pickle
import os

class CosineItemSimilarity(BaseSimilarity):

    def __init__(self, model):

        self._model = model
        self._memory = {}

        if os.path.isfile('tmp/cosine_item_sim.pkl'):
            pkl_file = open('tmp/cosine_item_sim.pkl', 'rb')
            self.cosine_sim = pickle.load(pkl_file)
            pkl_file.close()

            pkl_file2 = open('tmp/item_to_index_dict.pkl', 'rb')
            self.cosine_dict = pickle.load(pkl_file2)
            pkl_file2.close()

            self.loaded_dict = True
        else:
            self.loaded_dict = False

    def get_similarity(self, item1, item2):

        if(self.loaded_dict):
            try:
                item1_idx = self.cosine_dict[item1]
                item2_idx = self.cosine_dict[item2]    
                return self.cosine_sim[item1_idx][item2_idx]
            except:
                return 0

        prefs1, prefs2 = self.get_centered_prefs(item1,item2)
        sim = self.cosine_sim(prefs1, prefs2)

        if(np.isnan(sim)):
            sim = 0

        return sim

    def cosine_sim(self, v1, v2):
        try:
            return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))
        except:
            return 0
    def get_centered_prefs(self, item1, item2):
        #print item1
        #print item2

        prefs1 = self._model.preference_values_for_item(item1).toarray()
        prefs2 = self._model.preference_values_for_item(item2).toarray()

        avg1 = self._model.get_item_id_avg(item1)
        avg2 = self._model.get_item_id_avg(item2)

        # Center Cosine
        prefs1 = prefs1 - np.int8((prefs1>0)*avg1)
        prefs2 = prefs2 - np.int8((prefs2>0)*avg2)
        
        #print prefs1
        #print prefs2

        return prefs1, prefs2

