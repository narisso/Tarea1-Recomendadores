
class BaseRecommender(object):
    def recommend(self, user_id, n):
        raise NotImplementedError("cannot instantiate Abstract Base Class")