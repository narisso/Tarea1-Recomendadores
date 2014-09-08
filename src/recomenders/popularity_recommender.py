from .base_recommender import BaseRecommender

class PopularityRecommender(BaseRecommender):

    def __init__(self, model):

        self._model = model

    def recomend(self, user_id, n):
        pass

    def predict(self, user_id, item_id):
        try:
            return self._model.get_item_id_avg(item_id)
        except:
            print "Item no existe"
            return 0