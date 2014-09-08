
import numpy as np
import cPickle as pickle
from scipy.sparse import *
from scipy import *
import os
from .base_model import BaseDataModel, UserNotFoundError, ItemNotFoundError

class RatingPreferenceMatrix(BaseDataModel):
   
    def __init__(self, dataset,t_dataset=None, sparse_rows= True, is_dict = False):
        BaseDataModel.__init__(self)
        
        if is_dict:
            self.build_model(dataset,t_dataset,sparse_rows)
        else:
            raise Exception('Error', 'you need to specify dataset type')

    def __getitem__(self, user_id):
        return self.preferences_from_user(user_id)

    def __iter__(self):
        for index, user in enumerate(self.user_ids()):
            yield user, self[user]

    def __len__(self):
        return self.index.shape

    def build_model(self, dataset, t_dataset, sparse_rows):

        self.dataset = dataset
        self.t_dataset = t_dataset

        self._user_ids = np.asanyarray(dataset.keys())
        self._user_ids.sort()

        self._item_ids = []
        for items in dataset.itervalues():
            self._item_ids.extend(items.keys())

        self._item_ids = np.unique(np.array(self._item_ids))
        self._item_ids.sort()

        self.user_id_map = {}
        self.item_id_map = {}

        self.max_pref = -np.inf
        self.min_pref = np.inf

        for userno, user_id in enumerate(self._user_ids):
            self.user_id_map[user_id] = userno

        for itemno, item_id in enumerate(self._item_ids):
            self.item_id_map[item_id] = itemno

        self.rev_user_id_map = dict((reversed(item) for item in self.user_id_map.items()))
        self.rev_item_id_map = dict((reversed(item) for item in self.item_id_map.items()))

        if sparse_rows:
            self.build_sparse_rows_matrix(dataset)
        else:
            self.build_sparse_cols_matrix(dataset)

        self.build_user_avgs()
        self.build_item_avgs()

        #self.nonzero_index = {}
        # for uno, uid in enumerate(self._user_ids):
        #     x, self.nonzero_index[uid] = self.preference_values_from_user(uid).nonzero()

        print
        print "======================="
        print "MODEL LOADED"
        print "%d Users" % self._user_ids.size
        print "%d Items" % self._item_ids.size
        print "======================="
        print 

    def build_user_avgs(self):

        if os.path.isfile('tmp/user_avg_dict.pkl'):
            pkl_file = open('tmp/user_avg_dict.pkl', 'rb')
            self.usr_avgs = pickle.load(pkl_file)
            pkl_file.close()
            print "USER AVGS LOADED: %f %%" % (100.0)
        else:
            self.usr_avgs = {}
            i = 0
            for user_id in self._user_ids:
                i = i + 1
                print "%d: %s" % (i, user_id)
                self.usr_avgs[user_id] = self.calc_user_id_avg(user_id)

            output = open('tmp/user_avg_dict.pkl', 'wb')
            pickle.dump(self.usr_avgs, output)
            output.close()

    def build_item_avgs(self):

        if os.path.isfile('tmp/item_avg_dict.pkl'):
            pkl_file = open('tmp/item_avg_dict.pkl', 'rb')
            self.item_avgs = pickle.load(pkl_file)
            pkl_file.close()
            print "ITEM AVGS LOADED: %f %%" % (100.0)
        else:
            self.item_avgs = {}
            i = 0
            for item_id in self._item_ids:
                i = i + 1
                if item_id == '0060934727':
                    print "SDAFFADSFDSFASFASDFADS"
                print "%d: %s" % (i, item_id)
                self.item_avgs[item_id] = self.calc_item_id_avg(item_id)

            output = open('tmp/item_avg_dict.pkl', 'wb')
            pickle.dump(self.item_avgs, output)
            output.close()

    def build_sparse_rows_matrix(self,dataset):

        if os.path.isfile('tmp/row_sparse_matrix.pkl'):

            pkl_file = open('tmp/row_sparse_matrix.pkl', 'rb')
            self.index = pickle.load(pkl_file)
            pkl_file.close()

            print "MATRIX LOADED: %f %%" % (100.0)

        else:
            
            self.index = np.zeros(shape=(self._user_ids.size, self._item_ids.size), dtype=int8)

            for userno, user_id in enumerate(self._user_ids):

                if userno % 1000 == 0:
                    print "PROGRESS: %f %%" % (float(userno) * 100.0 / float(self._user_ids.size))
                for item_id in dataset[user_id].keys():
                    itemno = self.item_id_map[item_id]
                    # 0.0 represents no rating
                    r = dataset[user_id].get(item_id, 0.0)
                    self.index[userno, itemno] = r

            print "PROGRESS: %f %%" % (100.0)

            if self.index.size:
                self.max_pref = 1.0
                self.min_pref = 10.0

            self.sparse_matrix = csr_matrix( self.index , dtype=int8)
            self.sparse_matrix.eliminate_zeros()
            output = open('tmp/row_sparse_matrix.pkl', 'wb')
            pickle.dump(self.sparse_matrix, output)
            output.close()
            self.index = self.sparse_matrix
            self.sparse_matrix = None

    def build_sparse_cols_matrix(self,dataset):

        if os.path.isfile('tmp/col_sparse_matrix.pkl'):

            pkl_file = open('tmp/col_sparse_matrix.pkl', 'rb')
            self.index = pickle.load(pkl_file)
            pkl_file.close()

            print "MATRIX LOADED: %f %%" % (100.0)

        else:
            
            self.index = np.zeros(shape=(self._user_ids.size, self._item_ids.size), dtype=int8)

            for userno, user_id in enumerate(self._user_ids):

                if userno % 1000 == 0:
                    print "PROGRESS: %f %%" % (float(userno) * 100.0 / float(self._user_ids.size))
                for item_id in dataset[user_id].keys():
                    itemno = self.item_id_map[item_id]
                    # 0.0 represents no rating
                    r = dataset[user_id].get(item_id, 0.0)
                    self.index[userno, itemno] = r

            print "PROGRESS: %f %%" % (100.0)

            if self.index.size:
                self.max_pref = 1.0
                self.min_pref = 10.0

            self.sparse_matrix = csc_matrix( self.index , dtype=int8)
            self.sparse_matrix.eliminate_zeros()
            output = open('tmp/col_sparse_matrix.pkl', 'wb')
            pickle.dump(self.sparse_matrix, output)
            output.close()
            self.index = self.sparse_matrix
            self.sparse_matrix = None

    def user_ids(self):
        return self._user_ids

    def item_ids(self):
        return self._item_ids

    def index_to_user_id(self,idx):
        return self.rev_user_id_map[idx]

    def index_to_item_id(self,idx):
        return self.rev_item_id_map[idx]

    def user_id_to_index(self,uid):
        return self.user_id_map[uid]

    def item_id_to_index(self,iid):
        return self.item_id_map[iid]

    def get_user_id_avg(self, user_id):
        return self.usr_avgs[user_id]

    def get_item_id_avg(self, item_id):
        return self.item_avgs[item_id]

    def calc_user_id_avg(self, user_id):

        prefs = self.dataset[user_id]
        sum = 0
        tot = 0

        for item, value in prefs.items():
            sum = sum + value
            tot = tot + 1

        if(tot > 0):
            return float(sum)/float(tot)

        raise Exception()

    def calc_item_id_avg(self, item_id):

        if (self.t_dataset):
            prefs = self.t_dataset[item_id]
            sum = 0
            tot = 0

            for item, value in prefs.items():
                sum = sum + value
                tot = tot + 1

            if(tot > 0):
                return float(sum)/float(tot)

        raise Exception()

    def keys_from_user(self, user_id):
        try :
            keys = self.dataset[user_id].keys()
        except :
            raise UserNotFoundError("User not found")

        return keys

    def preference_values_for_item(self, item_id):
        try :
            item_id_loc = self.item_id_map[item_id]
        except :
            raise ItemNotFoundError("Item not found")

        preferences = self.index[:,item_id_loc]

        return preferences

    def preference_values_from_user(self, user_id):
        try :
            user_id_loc = self.user_id_map[user_id]
        except :
            raise UserNotFoundError("User not found")

        preferences = self.index[user_id_loc]

        return preferences

    def preferences_from_user(self, user_id):
        preferences = self.preference_values_from_user(user_id)

        data = zip(self._item_ids, preferences.flatten())

        return [(item_id, preference) for item_id, preference in data]
                        

    def has_preference_values(self):
        return True

    def maximum_preference_value(self):
        return self.max_pref

    def minimum_preference_value(self):
        return self.min_pref

    def users_count(self):
        return self._user_ids.size

    def items_count(self):
        return self._item_ids.size

    def items_from_user(self, user_id):
        preferences = self.preferences_from_user(user_id)
        return [key for key, value in preferences]

    def preferences_for_item(self, item_id):
        try:
            item_id_loc = self.item_id_map[item_id]
        except:
            raise ItemNotFoundError('Item not found')
        preferences = self.index[:, item_id_loc]

       
        data = zip(self._user_ids, preferences.flatten())

        return [(user_id, preference)  for user_id, preference in data]

    def preference_value(self, user_id, item_id):
        try:
            item_id_loc = self.item_id_map[item_id]
        except:
            raise ItemNotFoundError('Item not found')
        try :
            user_id_loc = self.user_id_map[user_id]
        except :
            raise UserNotFoundError("User not found")

        return self.index[user_id_loc, item_id_loc].flatten()[0]


    def __repr__(self):
        return "<MatrixPreferenceDataModel (%d by %d)>" % (self.index.shape[0],
                        self.index.shape[1])

