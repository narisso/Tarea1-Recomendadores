from .base_recommender import BaseRecommender
import numpy as np
import cPickle as pickle
import os

class SlopeOneRecommender(BaseRecommender):

    def __init__(self, model, min_ratings = 2):

        print "LOADING SLOPE ONE RECOMMENDER"

        self._model = model
        self.user_deviations = {}
        self.min_ratings = min_ratings # Parameter

        self.build_deviations()

        print "SLOPE ONE RECOMMENDER LOADED"

    def recomend(self, user_id, n):
        pass

    def predict(self, user_id, item_id):
        
        item_set = self._model.preference_values_from_user(user_id).keys()

        sum = 0
        tot = 0

        for itemno in item_set:
            iid = self._model.index_to_item_id(itemno)
            num_ratings = len(self._model.preference_values_for_item(iid))

            if num_ratings >= self.min_ratings:
                # weighted slope one
                sum = sum + (self.deviations[item_id][iid][0] + self._model.preference_value(user_id, iid))*self.deviations[item_id][iid][1] 
                tot = self.deviations[item_id][iid][1] 

        if tot == 0:
            return 0
        ret = float(sum) / float(tot)

        if ret > 10.0:
            return 10.0
        if ret < 0.0:
            return 0.0
        return ret


    def deviation(self,item1,item2):
       
        common, prefs1, prefs2 = self.get_common_users(item1, item2)

        if(len(common) == 0):
            return np.nan, 0

        return  float((prefs1 - prefs2).sum()) / float(len(common)), len(common)
    
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

    def build_deviations(self):

        if os.path.isfile('tmp/slope_one_deviations.pkl'):
            pkl_file = open('tmp/slope_one_deviations.pkl', 'rb')
            self.deviations = pickle.load(pkl_file)
            pkl_file.close()
            print "DEVIATIONS LOADED: %f %%" % (100.0)
        else:
            self.compute_all_deviations()
            output = open('tmp/slope_one_deviations.pkl', 'wb')
            pickle.dump(self.deviations, output)
            output.close()


    def compute_all_deviations(self):
        self.deviations = {}

        for n, item_id in enumerate(self._model.item_ids()):
            self.deviations[item_id] = {}

            print "DEVIATIONS PROGRESS: %f %%" % (float(n) * 100.0 / float(self._model.item_ids().size))

            prefs = self._model.preference_values_for_item(item_id).keys()

            for userno in prefs:
                user_id = self._model.index_to_user_id(userno)
                items = self._model.preference_values_from_user(user_id).keys()
                for itemno in items:
                    item_id2 = self._model.index_to_item_id(itemno)
                    d, l = self.deviation(item_id, item_id2)

                    if not np.isnan(d):
                        self.deviations[item_id][item_id2] = (d,l)

        print "DEVIATIONS PROGRESS: %f %%" % (100.0)
