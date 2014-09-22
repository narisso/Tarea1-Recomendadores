import csv

from scipy.sparse import *
from scipy import *
import pickle
from data_models.rating_preference_matrix import RatingPreferenceMatrix
from recomenders.slope_one_recommender import SlopeOneRecommender
from recomenders.popularity_recommender import PopularityRecommender
from similarities.weighted_pearson_user_similarity import WeightedPearsonUserSimilarity
from recomenders.user_based_recommender import UserBasedRecommender
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
    with open('../prediction/list-top-n.txt', 'rb') as csvfile:
        data = {}
        reader = csv.reader(csvfile,delimiter=';')
        for row in reader:
            if(row[0] not in data):
                data[row[0]] = {}
            data[row[0]] = 1

        return data

def generate_predicitons(data, recommender):
    
    print 'GENERATING PREDICTIONS'
    f = open('../prediction/list-n-prediction.txt','wb')
    for user in data.keys():
        recommended_scores, recommended_list = recommender.recommend(user)
        best_10 = recommended_list
        print "WRITING "+user
        f.write('"'+user+'"\n')
        for item in best_10:
            f.write('\t"'+item+'"\n')
    f.close()
    

def main():
    data = get_train_dataset()
    t_data = get_train_dataset_traspose()
    
    model = RatingPreferenceMatrix(data, t_data)
    
    # similarity = WeightedPearsonUserSimilarity(model)
    # recommender = UserBasedRecommender(model, similarity)
    recommender = SlopeOneRecommender(model)

    prediction_set = get_predictions()

    generate_predicitons(prediction_set, recommender)


if __name__ == "__main__":
    main()
