from .base_recommender import BaseRecommender
import heapq

class UserBasedRecommender(BaseRecommender):

    def __init__(self, model, similarity, k = 20):

        self._model = model
        self._similarity = similarity
        self._k_facor = float(k) #Len of user neighbors

    def recomend(self, user_id, n):
        pass

    def predict(self, user_id, item_id):

        avg = self._model.get_user_id_avg(user_id)

        neighbors = self.get_neighbors(user_id)
        top_sum = 0
        bot_sum = 0

        try:

            for n in neighbors:
                n_avg = self._model.get_user_id_avg( n[1] )
                p = self._model.preference_value(n[1],item_id)
                if(p > 0):
                    top_sum = top_sum + n[0] * (p - n_avg)
                    bot_sum = bot_sum + n[0]

            if bot_sum == 0:
                return 0
            else:
                return avg + top_sum / bot_sum
        except:
            return 0


    def get_neighbors(self, user_id):
        user_ids = self._model.user_ids()
        neighbors = []
        current_min_sim = -1.0

        for idx, u in enumerate(user_ids):
            sim = self._similarity.get_similarity(user_id, u)
            if(len(neighbors) < self._k_facor and sim > 0 ):
                heapq.heappush(neighbors, (sim, u))
            elif(sim > current_min_sim and sim > 0):
                min_item = heapq.heappop(neighbors)
                current_min_sim = min_item[0]
                heapq.heappush(neighbors, (sim, u))
            
            if(idx % 1000 == 0):
                print "PROGRESS: %f %% (%i/%i - %s)" % ((float(idx) * 100.0 / float(user_ids.size)) , idx , user_ids.size, u)

        print neighbors
        return neighbors
