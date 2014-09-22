from .base_recommender import BaseRecommender
import numpy as np
import cPickle as pickle
import os
import heapq

class SlopeOneRecommender(BaseRecommender):

    def __init__(self, model, min_ratings = 2):

        print "LOADING SLOPE ONE RECOMMENDER"

        self._model = model
        self.user_deviations = {}
        self.min_ratings = min_ratings # Parameter

        self.build_deviations()

        print "SLOPE ONE RECOMMENDER LOADED"

    def recommend(self, user_id, n= 10, test_set=[]):
        # user_prefs = self._model.preference_values_from_user(user_id).keys()
        reccomendations = []
        current_min_sim = -1.0
        
        prefs = self._model.preference_values_from_user(user_id).keys()
        for itemno in prefs:
            iid = self._model.index_to_item_id(itemno)
            score = self.predict(user_id,iid,test_set=test_set)
            if(len(reccomendations) < n and score > 0.0 ):
                heapq.heappush(reccomendations, (score, iid))
            elif(score > current_min_sim and score > 0.0):
                min_item = heapq.heappop(reccomendations)
                current_min_sim = min_item[0]
                heapq.heappush(reccomendations, (score, iid))
        
        rec_list = map(list, zip(*reccomendations))

        return rec_list

    def predict(self, user_id, item_id, test_set=[], default = None):
        
        item_set = self._model.preference_values_from_user(user_id).keys()

        sum = 0
        tot = 0

        if(default != None):
            default = default
        else:
            try:
                default1 = self._model.get_item_id_avg(item_id)
            except:
                default1 = 0.0
            try:
                default2 = self._model.get_user_id_avg(user_id)
            except:
                default2 = 0.0

            default = ( default1 + default2 ) / 2.0

        for itemno in item_set:
            iid = self._model.index_to_item_id(itemno)

            num_ratings = len(self._model.preference_values_for_item(iid))

            if num_ratings >= self.min_ratings:

                dev = self.get_deviation(item_id,iid)
                frq = self.get_freq(item_id,iid)

                if( np.isnan(dev) or np.isnan(frq) ):
                    pass
                elif(len(test_set) > 0 and test_set[itemno] == 1):
                    pass
                else:
                    # weighted slope one
                    sum = sum + (dev + self._model.preference_value(user_id, iid))*frq 
                    tot = tot + frq

        if tot == 0:
            return default
        ret = float(sum) / float(tot)

        if ret > 10.0:
            return 10.0
        if ret < 1.0 and ret != 0.0:
            return 1.0
        if ret == 0.0:
            return default
        return ret


    def get_freq(self,item1,item2):
        try:
            item1_index = self.slope_dict[item1]  
            item2_index = self.slope_dict[item2]

            ret = self.freqs[item1_index][item2_index]

            if ret == 0:
                return np.nan

            return ret
        except:
            return np.nan

    def get_deviation(self,item1,item2):
        try:

            item1_index = self.slope_dict[item1]   
            item2_index = self.slope_dict[item2]

            ret = self.deviations[item1_index][item2_index]

            if ret == 0:
                return np.nan

            return ret
        except:
            return np.nan

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

        if os.path.isfile('tmp/slope_one_diffs.pkl'):
            pkl_file = open('tmp/slope_one_diffs.pkl', 'rb')
            self.deviations = pickle.load(pkl_file)
            pkl_file.close()

            pkl_file = open('tmp/slope_one_freqs.pkl', 'rb')
            self.freqs = pickle.load(pkl_file)
            pkl_file.close()

            pkl_file2 = open('tmp/item_to_index_dict.pkl', 'rb')
            self.slope_dict = pickle.load(pkl_file2)
            self.rev_slope_dict = dict((reversed(item) for item in self.slope_dict.items()))
            pkl_file2.close()

            print "DEVIATIONS LOADED: %f %%" % (100.0)
        else:
            self.compute_all_deviations()
            output = open('tmp/slope_one_diffs.pkl', 'wb')
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
