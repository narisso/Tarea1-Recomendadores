from .base_recommender import BaseRecommender
import numpy as np
from scipy.sparse import *
from scipy import *

class SlopeOneRecommender(BaseRecommender):

    def __init__(self, model):
        self._model = model
        self.user_deviations = {}

    def recomend(self, user_id, n):
        pass

    def predict(self, user_id, item_id):
        item_set = self._model.keys_from_user(user_id)
        item_set.append(item_id)

        self.compute_deviations(user_id,item_set)

        sum = 0
        tot = 0

        for i in range(0,len(item_set)):
            if i == len(item_set) - 1:
                pass
            elif abs(self.user_deviations[user_id][len(item_set) - 1,i]) < 11:
                tot = tot + 1
                sum = sum + self.user_deviations[user_id][len(item_set) - 1,i] + self._model.preference_value(user_id, item_set[i])
                print "%f + %f " %(self.user_deviations[user_id][len(item_set) - 1,i],self._model.preference_value(user_id, item_set[i]))

        print sum
        print tot
        if tot == 0:
            return 0
        return float( 1.0 / float(tot) ) * float(sum)

    def deviation(self,x,y):
        item1 = self._model.index_to_item_id(x)
        item2 = self._model.index_to_item_id(y)

        vector1 = self._model.preference_values_for_item(item1).toarray()
        vector2 = self._model.preference_values_for_item(item2).toarray()

        vector1 = vector1 * (vector2>0)
        vector2 = vector2 * (vector1>0)

        if np.count_nonzero(vector1) == 0:
            return -100

        return  float((vector1 - vector2).sum()) / float(np.count_nonzero(vector1))

    def item_id_deviation(self,item1,item2):
        try:
            vector1 = self._model.preference_values_for_item(item1).toarray()
            vector2 = self._model.preference_values_for_item(item2).toarray()
        except:
            return -100
            
        vector1 = vector1 * (vector2>0)
        vector2 = vector2 * (vector1>0)

        if np.count_nonzero(vector1) == 0:
            return -100

        return  (vector1 - vector2).sum() / np.count_nonzero(vector1)

    def compute_deviations(self,user_id, item_set):
        n_items = len(item_set)
        self.user_deviations[user_id] = np.zeros(shape=(n_items,n_items), dtype=np.int8)
        #self.user_deviations[user_id][len(item_set) - 1,i]
        for j in range(0,len(item_set)):
            if j == len(item_set) - 1:
                pass
            else:
                dev = self.item_id_deviation(item_set[len(item_set) - 1],item_set[j])
                self.user_deviations[user_id][len(item_set) - 1,j] = dev
                self.user_deviations[user_id][j, len(item_set) - 1] = -dev
   

    def compute_all_deviations(self):
        n_items = len(self._model.item_ids())
        self.deviations = np.zeros(shape=(n_items,n_items), dtype=np.int8)

        print "COMPUTING DEV"
        for i in range(0,n_items):
            for j in range(i, n_items):
                if i == j:
                    pass
                else:
                    dev = self.deviation(i,j)
                    self.deviations[i,j] = dev
                    self.deviations[j,i] = -dev
                    print "%i  %i / %i  - %f %%" % (dev, j + n_items*i , n_items*n_items, float(j + n_items*i)*100.0/float(n_items*n_items) )
        output = open('tmp/item_dev_matrix.pkl', 'wb')
        #self.sparse = coo_matrix(self.deviations)
        pickle.dump(self.deviations, output)
        output.close()