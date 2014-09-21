
import numpy as np
import cPickle as pickle
from scipy.sparse import *
from scipy import *
import os
from .base_model import BaseDataModel, UserNotFoundError, ItemNotFoundError

class RatingPreferenceMatrix(BaseDataModel):
   
    def __init__(self, dataset, t_dataset):
        
        BaseDataModel.__init__(self)
        self.build_model(dataset,t_dataset)

    def __getitem__(self, user_id):
        return self.preferences_from_user(user_id)

    def __iter__(self):
        for index, user in enumerate(self.user_ids()):
            yield user, self[user]

    def __len__(self):
        return self.index.shape

    def build_model(self, dataset, t_dataset):

        self._user_ids = np.asanyarray(dataset.keys())
        self._user_ids.sort()

        self._item_ids = []
        for items in dataset.itervalues():
            self._item_ids.extend(items.keys())

        self._item_ids = np.unique(np.array(self._item_ids))
        self._item_ids.sort()

        self.user_id_map = {}
        self.item_id_map = {}

        for userno, user_id in enumerate(self._user_ids):
            self.user_id_map[user_id] = userno

        for itemno, item_id in enumerate(self._item_ids):
            self.item_id_map[item_id] = itemno

        self.rev_user_id_map = dict((reversed(item) for item in self.user_id_map.items()))
        self.rev_item_id_map = dict((reversed(item) for item in self.item_id_map.items()))

        self.build_user_dict(dataset)
        self.build_item_dict(t_dataset)

        self.build_user_avgs()
        self.build_item_avgs()

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
                print "%d: %s" % (i, item_id)
                self.item_avgs[item_id] = self.calc_item_id_avg(item_id)

            output = open('tmp/item_avg_dict.pkl', 'wb')
            pickle.dump(self.item_avgs, output)
            output.close()

    def build_user_dict(self,dataset):

        if os.path.isfile('tmp/user_dict.pkl'):

            pkl_file = open('tmp/user_dict.pkl', 'rb')
            self.user_dict = pickle.load(pkl_file)
            pkl_file.close()

            print "USER DICT LOADED: %f %%" % (100.0)

        else:
            
            self.user_dict = {}

            for userno, user_id in enumerate(self._user_ids):
                self.user_dict[userno] = {}
                if userno % 1000 == 0:
                    print "USER DICT PROGRESS: %f %%" % (float(userno) * 100.0 / float(self._user_ids.size))
                for item_id in dataset[user_id].keys():
                    itemno = self.item_id_map[item_id]
                    self.user_dict[userno][itemno] = dataset[user_id][item_id]

            print "USER DICT PROGRESS: %f %%" % (100.0)

            output = open('tmp/user_dict.pkl', 'wb')
            pickle.dump(self.user_dict, output)
            output.close()

    def build_item_dict(self,t_dataset):

        if os.path.isfile('tmp/item_dict.pkl'):

            pkl_file = open('tmp/item_dict.pkl', 'rb')
            self.item_dict = pickle.load(pkl_file)
            pkl_file.close()

            print "ITEM DICT LOADED: %f %%" % (100.0)

        else:
            
            self.item_dict = {}

            for itemno, item_id in enumerate(self._item_ids):
                self.item_dict[itemno] = {}
                if itemno % 1000 == 0:
                    print "ITEM DICT PROGRESS: %f %%" % (float(itemno) * 100.0 / float(self._item_ids.size))
                for user_id in t_dataset[item_id].keys():                    
                    userno = self.user_id_map[user_id]
                    self.item_dict[itemno][userno] = t_dataset[item_id][user_id]

            print "ITEM DICT PROGRESS: %f %%" % (100.0)

            output = open('tmp/item_dict.pkl', 'wb')
            pickle.dump(self.item_dict, output)
            output.close()


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

        userno = self.user_id_to_index(user_id)
        prefs = self.user_dict[userno]
        sum = 0
        tot = 0

        for item, value in prefs.items():
            sum = sum + value
            tot = tot + 1

        if(tot > 0):
            return float(sum)/float(tot)

        raise Exception()

    def calc_item_id_avg(self, item_id):

        itemno = self.item_id_to_index(item_id)
        prefs = self.item_dict[itemno]
        sum = 0
        tot = 0

        for item, value in prefs.items():
            sum = sum + value
            tot = tot + 1

        if(tot > 0):
            return float(sum)/float(tot)

        raise Exception()

    def preference_values_for_item(self, item_id):
        try :
            itemno = self.item_id_to_index(item_id)
        except :
            raise ItemNotFoundError("Item not found")

        preferences = self.item_dict[itemno]

        return preferences

    def preference_values_from_user(self, user_id):
        try :
            userno = self.user_id_to_index(user_id)
        except :
            raise UserNotFoundError("User not found")

        preferences = self.user_dict[userno]

        return preferences                    

    def users_count(self):
        return self._user_ids.size

    def items_count(self):
        return self._item_ids.size

    def preference_value(self, user_id, item_id):
        try:
            itemno = self.item_id_to_index(item_id)
        except:
            raise ItemNotFoundError('Item not found')
        try :
            userno = self.user_id_to_index(user_id)
        except :
            raise UserNotFoundError("User not found")

        try:
            ret = self.user_dict[userno][itemno]
        except:
            ret = 0
        
        return ret

    def preference_value_from_index(self, userno, itemno):

        try:
            ret = self.user_dict[userno][itemno]
        except:
            ret = 0
        
        return ret
