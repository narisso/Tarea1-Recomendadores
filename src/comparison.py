import csv
import numpy as np

from sklearn import cross_validation

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

def cross_validate(data, t_data, model,  recommender, n_folds = 5):
    kf = cross_validation.KFold(len(data), n_folds)

    stats= {}

    stats['NAME'] = recommender.__class__.__name__

    # CALC MSE
    stats['MSE'] = 0
    stats['MSE_LIST'] = []

    for train_index, test_index in kf:
        # model.index_to_user_id(train_index)
        # model.index_to_user_id(test_index)
        top_sum = 0
        i = 0.0
        for userno in test_index:
            user = model.index_to_user_id(userno)
            item_list = data[user]
            print "PROGRESS: %f %%" % (i / float(len(test_index)) * 100.0) 
            for item in item_list.keys():
                predicted = float(recommender.predict(user,item,test_index))
                real = float(data[user][item])
                top_sum = top_sum + pow( predicted - real ,2)
            i = i + 1.0

        stats['MSE_LIST'].append( np.sqrt(top_sum) / len(data) )
    stats['MSE'] = np.average(stats['MSE_LIST'])
    print stats


def main():
    data = get_train_dataset()
    t_data = get_train_dataset_traspose()

    model = RatingPreferenceMatrix(data, t_data)

    # slope_recommender = SlopeOneRecommender(model)
    # pop_recommender = PopularityRecommender(model)

    # item_sim    = AdjCosineItemSimilarity(model)
    # item_recommender = ItemBasedRecommender(model, item_sim)

    user_sim = WeightedPearsonUserSimilarity(model)
    user_recommender = UserBasedRecommender(model, user_sim)

    cross_validate(data,t_data,model,user_recommender)
    

if __name__ == "__main__":
    main()
