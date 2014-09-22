from .base_recommender import BaseRecommender
import heapq
import cPickle as pickle
import os

class ItemBasedRecommender(BaseRecommender):

    def __init__(self, model, similarity, k = 20):

        print "LOADING ITEM BASED RECOMMENDER"

        self._model = model
        self._similarity = similarity
        self._k_facor = float(k) #Len of item neighbors

        self.build_similarities()

        print "ITEM BASED RECOMMENDER LOADED"

    def build_similarities(self):

        if os.path.isfile('tmp/cosine_item_sim.pkl'):
            pkl_file = open('tmp/cosine_item_sim.pkl', 'rb')
            self.similarities = pickle.load(pkl_file)
            pkl_file.close()
            
            pkl_file2 = open('tmp/item_to_index_dict.pkl', 'rb')
            self.cosine_dict = pickle.load(pkl_file2)
            self.rev_cosine_dict = dict((reversed(item) for item in self.cosine_dict.items()))
            pkl_file2.close()

            print "ITEM SIMILARITIES LOADED: %f %%" % (100.0)
        else:
            self.similarities = {}

            for n, item_id in enumerate(self._model.item_ids()):
                self.similarities[item_id] = {}

                print "ITEM SIMILARITIES PROGRESS: %f %%" % (float(n) * 100.0 / float(self._model.item_ids().size))

                prefs = self._model.preference_values_for_item(item_id).keys()

                for userno in prefs:
                    user_id = self._model.index_to_user_id(userno)
                    items = self._model.preference_values_from_user(user_id).keys()
                    for itemno in items:
                        item_id2 = self._model.index_to_item_id(itemno)
                        s = self._similarity.get_similarity(item_id, item_id2)

                        if item_id2 not in self.similarities:
                            if s > 0:
                                self.similarities[item_id][item_id2] = s

            print "ITEM SIMILARITIES PROGRESS: %f %%" % (100.0)

            output = open('tmp/cosine_item_sim.pkl', 'wb')
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

    def predict(self, user_id, item_id, test_set = [], default = None):

        neighbors = self.get_neighbors(item_id, test_set)
        top_sum = 0
        bot_sum = 0

        if(default != None):
            default = default
        else:
            try:
                default1 = self._model.get_item_id_avg(item_id)
            except:
                default1 = 0
            try:
                default2 = self._model.get_user_id_avg(user_id)
            except:
                default2 = 0

            default = ( default1 + default2 ) / 2
        try:

            for n in neighbors:
                p = self._model.preference_value(user_id, n[1])
                if(p > 0):
                    top_sum = top_sum + n[0] * (p)
                    bot_sum = bot_sum + n[0]

            if bot_sum == 0:
                return default
            else:
                ret = top_sum / bot_sum
                if ret > 10.0:
                    return 10.0
                if ret < 1.0 and ret != 0.0:
                    return 1.0
                if ret == 0.0:
                    return default
                return ret
        except:
            return default


    def get_neighbors(self, item_id, test_set):
        neighbors = []
        current_min_sim = -1.0

        try:
            item_index = self.cosine_dict[item_id]
        except:
            return neighbors

        if len(test_set) == 0:

            for itemno, sim in self.similarities[item_index].items():
                
                item_id2 = self.rev_cosine_dict[itemno]
    
                if(len(neighbors) < self._k_facor and sim > 0 ):
                    heapq.heappush(neighbors, (sim, item_id2))
                elif(sim > current_min_sim and sim > 0):
                    min_item = heapq.heappop(neighbors)
                    current_min_sim = min_item[0]
                    heapq.heappush(neighbors, (sim, item_id2))

        else:
            for itemno, sim in self.similarities[item_index].items():
                item_id2 = self.cosine_dict[itemno]
                idx = self._model.item_id_to_index(item_id2)
                if(len(neighbors) < self._k_facor and sim > 0 ):
                    if(test_set[idx] == 1):
                        pass
                    else:
                        heapq.heappush(neighbors, (sim, item_id2))
                elif(sim > current_min_sim and sim > 0):
                    if(test_set[idx] == 1):
                        pass
                    else:
                        min_item = heapq.heappop(neighbors)
                        current_min_sim = min_item[0]
                        heapq.heappush(neighbors, (sim, item_id2))

        return neighbors