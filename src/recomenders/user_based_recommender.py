from .base_recommender import BaseRecommender
import heapq
import cPickle as pickle
import os

class UserBasedRecommender(BaseRecommender):

    def __init__(self, model, similarity, k = 20):

        print "LOADING USER BASED RECOMMENDER"

        self._model = model
        self._similarity = similarity
        self._k_facor = float(k) #Len of user neighbors

        self.build_similarities()

        print "USER BASED RECOMMENDER LOADED"

    def build_similarities(self):

        if os.path.isfile('tmp/pearson_user_sim.pkl'):
            pkl_file = open('tmp/pearson_user_sim.pkl', 'rb')
            self.similarities = pickle.load(pkl_file)
            pkl_file.close()
            print "USER SIMILARITIES LOADED: %f %%" % (100.0)
        else:
            self.similarities = {}

            for userno, user_id in enumerate(self._model.user_ids()):
                self.similarities[user_id] = {}

                print "USER SIMILARITIES PROGRESS: %f %%" % (float(userno) * 100.0 / float(self._model.user_ids().size))

                prefs = self._model.preference_values_from_user(user_id).keys()

                for itemno in prefs:
                    item_id = self._model.index_to_item_id(itemno)
                    users = self._model.preference_values_for_item(item_id).keys()
                    for userno in users:
                        user_id2 = self._model.index_to_user_id(userno)
                        s = self._similarity.get_similarity(user_id, user_id2)

                        if s > 0:
                            self.similarities[user_id][user_id2] = s

            print "USER SIMILARITIES PROGRESS: %f %%" % (100.0)

            output = open('tmp/pearson_user_sim.pkl', 'wb')
            pickle.dump(self.similarities, output)
            output.close()

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


    def predict(self, user_id, item_id, test_set=[], default = None ):

        avg = self._model.get_user_id_avg(user_id)

        neighbors = self.get_neighbors(user_id, item_id, test_set)
        top_sum = 0
        bot_sum = 0

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

        try:
            for n in neighbors:
                n_avg = self._model.get_user_id_avg( n[1] )
                p = self._model.preference_value(n[1],item_id)
                if(p > 0):
                    top_sum = top_sum + n[0] * (p - n_avg)
                    bot_sum = bot_sum + n[0]

            if bot_sum == 0:
                return default
            else:
                ret = avg + top_sum / bot_sum

                if ret > 10.0:
                    return 10.0
                if ret < 1.0 and ret != 0.0:
                    return 1.0
                if ret == 0.0:
                    return default
                return ret
        except:
            return default


    def get_neighbors(self, user_id, item_id, test_set):
        neighbors = []
        current_min_sim = -1.0

        if len(test_set) == 0:

            for user, sim in self.similarities[user_id].items():

                if(len(neighbors) < self._k_facor and sim > 0 ):
                    heapq.heappush(neighbors, (sim, user))
                elif(sim > current_min_sim and sim > 0):
                    min_item = heapq.heappop(neighbors)
                    current_min_sim = min_item[0]
                    heapq.heappush(neighbors, (sim, user))
                
        else:

            for user, sim in self.similarities[user_id].items():
                idx = self._model.user_id_to_index(user)

                if(len(neighbors) < self._k_facor and sim > 0 ):
                    if(test_set[idx] == 1):
                        pass
                    else:
                        heapq.heappush(neighbors, (sim, user))
                elif(sim > current_min_sim and sim > 0):
                    if(test_set[idx] == 1):
                        pass
                    else:
                        min_item = heapq.heappop(neighbors)
                        current_min_sim = min_item[0]
                        heapq.heappush(neighbors, (sim, user))
                
                
        return neighbors
