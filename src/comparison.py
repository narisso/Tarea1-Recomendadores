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

def get_test_data(model):

    test = {}
    i = 0
    for uid in model.dataset.keys():
        i = i + 1
        test[uid] = {}
        test[uid] = model.dataset[uid]
        if i == 50:
            break

    return test

def generate_predicitons(data, recommender, recommender2):
    pred = {}
    for user in data.keys():
        pred[user] = {}
        for item_id in data[user].keys():
            print
            print "Predicting " + item_id + " for " + user
            print
            pred[user][item_id] = recommender.predict(user,item_id)
            if pred[user][item_id] < 1.0 and pred[user][item_id] > 0:
                pred[user][item_id] = 1.0
            elif pred[user][item_id] > 10.0:
                pred[user][item_id] = 10.0
            print "Finish %f" % (pred[user][item_id])

def main():
    data = get_train_dataset();
    print "Loaded %d rows" % len(data)
    
    model = RatingPreferenceMatrix(data, is_dict=True)
    recommender = SlopeOneRecommender(model)
    recommender2 = PopularityRecommender(model)
    test_set = get_test_data()

    get_test_data(test_set, recommender, recommender2)


if __name__ == "__main__":
    main()
