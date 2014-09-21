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

                        if s > 0:
                            self.similarities[item_id][item_id2] = s

            print "ITEM SIMILARITIES PROGRESS: %f %%" % (100.0)

            output = open('tmp/cosine_item_sim.pkl', 'wb')
            pickle.dump(self.similarities, output)
            output.close()

    def get_similarity(self, item1, item2):

        try:

            if self.similarities:    
                return self.similarities[item1][item2]
        
        except Exception, e:
            print e
            return 0

        return self._similarity.get_similarity(item1, item2)

    def recomend(self, user_id, n):
        pass

    def predict(self, user_id, item_id):

        neighbors = self.get_neighbors(item_id)
        top_sum = 0
        bot_sum = 0

        try:

            for n in neighbors:
                p = self._model.preference_value(user_id, n[1])
                if(p > 0):
                    top_sum = top_sum + n[0] * (p)
                    bot_sum = bot_sum + n[0]

            if bot_sum == 0:
                return 0
            else:
                ret = top_sum / bot_sum
                if ret > 10.0:
                    return 10.0
                if ret < 0.0:
                    return 0.0
                return ret
        except:
            return 0


    def get_neighbors(self, item_id):
        item_ids = self._model.item_ids()
        neighbors = []
        current_min_sim = -1.0

        for idx, iid in enumerate(item_ids):
            sim = self.get_similarity(item_id, iid)
            if(len(neighbors) < self._k_facor and sim > 0 ):
                heapq.heappush(neighbors, (sim, iid))
            elif(sim > current_min_sim and sim > 0):
                min_item = heapq.heappop(neighbors)
                current_min_sim = min_item[0]
                heapq.heappush(neighbors, (sim, iid))
            
            if(idx % 1000 == 0):
                print "PROGRESS: %f %% (%i/%i - %s)" % ((float(idx) * 100.0 / float(item_ids.size)) , idx , item_ids.size, iid)

        print neighbors
        return neighbors