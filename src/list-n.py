import csv

from scipy.sparse import *
from scipy import *
import pickle
from data_models.rating_preference_matrix import RatingPreferenceMatrix
from recomenders.slope_one_recommender import SlopeOneRecommender
from recomenders.popularity_recommender import PopularityRecommender
import operator


def get_train_dataset():
    with open('../train/ratings.csv', 'rb') as csvfile:
        data = {}
        reader = csv.reader(csvfile,delimiter=';')
        for row in reader:
            if(row[0] not in data):
                data[row[0]] = {}
            data[row[0]][row[1]] = float(row[2])

        return data

def get_predictions():
    with open('../prediction/list-top-n.txt', 'rb') as csvfile:
        data = {}
        reader = csv.reader(csvfile,delimiter=';')
        for row in reader:
            if(row[0] not in data):
                data[row[0]] = {}
            data[row[0]] = 1

        return data

def get_best_items():
    pkl_file = open('tmp/best_items.pkl', 'rb')
    best_items = pickle.load(pkl_file)
    pkl_file.close()

    return best_items

def generate_predicitons(data, best, recommender, recommender2):
    pred = {}
    for user in data.keys():
        pred[user] = {}
        for item_id, n_item in best:
            print
            print "Predicting " + item_id + " for " + user
            print
            pred[user][item_id] = recommender.predict(user,item_id)
            if pred[user][item_id] < 1.0 and pred[user][item_id] > 0:
                pred[user][item_id] = 1.0
            elif pred[user][item_id] > 10.0:
                pred[user][item_id] = 10.0
            elif pred[user][item_id] == 0:
                pred[user][item_id] = recommender2.predict(user,item_id)
            print "Finish %f" % (pred[user][item_id])
            

    f = open('../prediction/list-n-prediction.txt','wb')
    for user in pred.keys():
        best_items = pred[user]
        sorted_items = sorted(best_items.iteritems(), key=operator.itemgetter(1))
        best_10 = sorted_items[-11:-1]

        f.write('"'+user+'"\n')
        for item in best_10:
            f.write('\t"'+item[0]+'"\n')
    f.close()

def slope_one_deviations(data):
    model = RatingPreferenceMatrix(data, sparse_rows = False, is_dict=True)
    recommender = SlopeOneRecommender(model)
    recommender.compute_deviations()
    

def main():
    data = get_train_dataset();
    print "Loaded %d rows" % len(data)
    
    model = RatingPreferenceMatrix(data, is_dict=True)
    recommender = SlopeOneRecommender(model)
    recommender2 = PopularityRecommender(model)
    prediction_set = get_predictions()

    best = get_best_items()
    generate_predicitons(prediction_set, best, recommender, recommender2)


if __name__ == "__main__":
    main()
