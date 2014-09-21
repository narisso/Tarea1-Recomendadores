import csv

from scipy.sparse import *
from scipy import *

from data_models.rating_preference_matrix import RatingPreferenceMatrix

from recomenders.slope_one_recommender import SlopeOneRecommender
from recomenders.popularity_recommender import PopularityRecommender
from similarities.weighted_pearson_user_similarity import WeightedPearsonUserSimilarity
from recomenders.user_based_recommender import UserBasedRecommender
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


def main():
    data = get_train_dataset()
    t_data = get_train_dataset_traspose()
    
    model = RatingPreferenceMatrix(data, t_data)

    slope_recommender = SlopeOneRecommender(model)
    pop_recommender = PopularityRecommender(model)

    item_sim    = AdjCosineItemSimilarity(model)
    item_recommender = ItemBasedRecommender(model, item_sim)

    user_sim = WeightedPearsonUserSimilarity(model)
    user_recommender = UserBasedRecommender(model, user_sim)
    

if __name__ == "__main__":
    main()
