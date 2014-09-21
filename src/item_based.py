import csv

from scipy.sparse import *
from scipy import *

from data_models.rating_preference_matrix import RatingPreferenceMatrix
from similarities.adj_cosine_item_similarity import AdjCosineItemSimilarity
from recomenders.item_based_recommender import ItemBasedRecommender

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
        reader = csv.reader(csvfile,delimiter=';')
        for row in reader:
            if(row[1] not in data):
                data[row[1]] = {}
            data[row[1]][row[0]] = float(row[2])

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

def generate_predictons(data, recommender):
    pred = {}
    for user in data.keys():
        pred[user] = {}
        for item_id in data[user].keys():
            print
            print "Predicting " + item_id + " for " + user
            print
            pred[user][item_id] = recommender.predict(user,item_id)

    with open('../prediction/item_based_prediction.csv', 'wb') as f:
        writer = csv.writer(f,delimiter=';')
        for user in pred.keys():
            for item, value in pred[user].items():
                writer.writerow([user, item, value])
    

def main():
    data = get_train_dataset();
    t_data = get_train_dataset_traspose()
    
    model = RatingPreferenceMatrix(data,t_data)
    similarity = AdjCosineItemSimilarity(model)
    recommender = ItemBasedRecommender(model, similarity)

    prediction_set = get_predictions()
    generate_predictons(prediction_set, recommender)

if __name__ == "__main__":
    main()
