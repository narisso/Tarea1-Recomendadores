from .base_recommender import BaseRecommender
import heapq

class ItemBasedRecommender(BaseRecommender):

    def __init__(self, model, similarity, k = 20):

        self._model = model
        self._similarity = similarity
        self._k_facor = float(k) #Len of user neighbors

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
                return top_sum / bot_sum
        except:
            return 0


    def get_neighbors(self, item_id):
        item_ids = self._model.item_ids()
        neighbors = []
        current_min_sim = -1.0

        for idx, iid in enumerate(item_ids):
            sim = self._similarity.get_similarity(item_id, iid)
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