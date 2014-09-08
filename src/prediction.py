import csv

from scipy.sparse import *
from scipy import *

from data_models.rating_preference_matrix import RatingPreferenceMatrix
from recomenders.slope_one_recommender import SlopeOneRecommender
from recomenders.popularity_recommender import PopularityRecommender

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
    with open('../prediction/prediction.csv', 'rb') as csvfile:
        data = {}
        reader = csv.reader(csvfile,delimiter=';')
        for row in reader:
            if(row[0] not in data):
                data[row[0]] = {}
            data[row[0]][row[1]] = float(row[2])

        return data

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
            elif pred[user][item_id] == 0:
                pred[user][item_id] = recommender2.predict(user,item_id)
            print "Finish %f" % (pred[user][item_id])
        

    with open('../prediction/prediction.csv', 'wb') as f:
        writer = csv.writer(f,delimiter=';')
        for user in pred.keys():
            for item, value in pred[user].items():
                writer.writerow([user+'', item+'', value])

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
    generate_predicitons(prediction_set, recommender, recommender2)


if __name__ == "__main__":
    main()
