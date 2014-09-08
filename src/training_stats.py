import csv
import numpy as np
import pickle
import os
import operator

from data_models.rating_preference_matrix import RatingPreferenceMatrix

def get_train_dataset():
    with open('../train/ratings.csv', 'rb') as csvfile:
        data = {}
        reader = csv.reader(csvfile,delimiter=';')
        for row in reader:
            if(row[0] not in data):
                data[row[0]] = {}
            data[row[0]][row[1]] = float(row[2])

        return data

def get_train_dataset_traspose():
    with open('../train/ratings.csv', 'rb') as csvfile:
        data = {}
        stats = {}
        stats["total"] = 0
        stats["plain"] = []
        reader = csv.reader(csvfile,delimiter=';')
        for row in reader:
            if(row[1] not in data):
                data[row[1]] = {}
            data[row[1]][row[0]] = float(row[2])
            stats["total"] = stats["total"] + 1
            stats["plain"].append(float(row[2]))

        return stats, data


def main():
    data = get_train_dataset()
    stats, t_data = get_train_dataset_traspose()
    non_zeros = stats["total"]
    print "Loaded %d rows" % len(data)
    
    model = RatingPreferenceMatrix(data,t_data, is_dict=True)
    total = float(model.user_ids().size * model.item_ids().size)

    avg = (float(non_zeros)/ float(model.user_ids().size))
    plain = stats["plain"]

    avg_score =  np.average(plain)
    std_score =  np.std(plain)

    views_array = []
    for uid in model.dataset.keys():
        x= 0
        for iid in model.dataset[uid].keys():
            x = x+1
        views_array.append(x)

    avg_views =  np.average(views_array)
    std_vies  =  np.std(views_array)

    # if os.path.isfile('tmp/stats.pkl'):
    #     pkl_file = open('tmp/stats.pkl', 'rb')
    #     means = pickle.load(pkl_file)
    #     pkl_file.close()
    # else:
    #     for uid in model.user_ids():
    #         print uid
    #         means[uid] = {}
    #         prefs = model.preference_values_from_user(uid).toarray()
    #         nz = np.count_nonzero(prefs)
    #         means[uid]["count"] = nz
    #         means[uid]["tot"] = len(prefs)

    #     output = open('tmp/stats.pkl', 'wb')
    #     pickle.dump(means, output)
    #     output.close()

    print
    print "=======================" 
    print "TRAINING DATA STATISTICS"
    print "%d Users" % model.user_ids().size
    print "%d Items" % model.item_ids().size
    print "%d non_zeros " % (non_zeros ) 
    print "%f %% Sparse " % (non_zeros * 100.0/ total) 
    print "%f Avg User Views " % avg
    print "%f Std User Views " % std_vies
    print "%f Avg User Score " % avg_score
    print "%f Std User Score " % std_score
    print "=======================" 
    print

    best_items = {}
    for iid in model.t_dataset.keys():
        best_items[iid] = 0
        x = 0
        for uid in model.t_dataset[iid].keys():
            x = x+1
        best_items[iid] = x

    sorted_x = sorted(best_items.iteritems(), key=operator.itemgetter(1))
    best_items = sorted_x[-51:-1]
    print len(best_items)

    output = open('tmp/best_items.pkl', 'wb')
    pickle.dump(best_items, output)
    output.close()

if __name__ == "__main__":
    main()
