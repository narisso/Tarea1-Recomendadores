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

def cross_validate_users(data, t_data, model,  recommender, n_folds = 5):
    kf = cross_validation.KFold(len(data), n_folds)

    stats= {}

    stats['NAME'] = recommender.__class__.__name__

    stats['RMSE'] = 0
    stats['RMSE_LIST'] = []
    stats['MAP_LIST'] = []
    stats['MAR_LIST'] = []

    for train_index, test_index in kf:

        test_set = np.zeros(len(data))
        for i in test_index:
            test_set[i] = 1

        top_sum = 0
        bot_sum = 0
        i = 0.0

        p_list = []
        r_list = []

        for userno in test_index:
            user = model.index_to_user_id(userno)
            
            recommended_scores, recommended_list = recommender.recommend(user, test_set=test_set)
            relevants = get_relevant_items(model,user)

            p, r = get_precision_recall(recommended_list, relevants)
            p_list.append(p)
            r_list.append(r)

            item_list = data[user]
            print "PROGRESS %s: %f %%" % (user,i / float(len(test_index)) * 100.0) 
            for item in item_list.keys():
                predicted = float(recommender.predict(user,item, test_set=test_set ))
                real = float(data[user][item])
                top_sum = top_sum + pow( predicted - real ,2)
                bot_sum = bot_sum + 1.0
            i = i + 1.0
        stats['RMSE_LIST'].append( np.sqrt(top_sum / bot_sum) )
        stats['MAP_LIST'].append( np.average(p_list) )
        stats['MAR_LIST'].append( np.average(r_list) )
    stats['RMSE'] = np.average(stats['RMSE_LIST'])
    stats['MAP'] = np.average(stats['MAP_LIST'])
    stats['MAR'] = np.average(stats['MAR_LIST'])

    print stats

def get_relevant_items(model,user_id):
    avg = np.average( model.preference_values_from_user(user_id).values() )
    std = np.std( model.preference_values_from_user(user_id).values() )
    limit = avg + std

    relevant = []

    for itemno, rating in model.preference_values_from_user(user_id).items():
        iid = model.index_to_item_id(itemno)
        if rating >= limit:
            relevant.append(iid)

    return relevant

def get_precision_recall(predicted, relevant):
    if len(predicted) == 0:
        precision = 0.0
    else:
        precision = float(len(np.intersect1d(predicted, relevant, assume_unique = True))) / float(10)
    if len(relevant) == 0:
        recall = 0.0
    else:
        recall = float(len(np.intersect1d(predicted, relevant, assume_unique = True))) / float(len(relevant))
    return precision, recall


def cross_validate_items(data, t_data, model,  recommender, n_folds = 5):
    kf = cross_validation.KFold(len(t_data), n_folds)

    stats= {}

    stats['NAME'] = recommender.__class__.__name__

    stats['RMSE'] = 0
    stats['RMSE_LIST'] = []

    for train_index, test_index in kf:
        top_sum = 0
        bot_sum = 0
        i = 0.0

        for itemno in test_index:
            item = model.index_to_item_id(itemno)
            user_list = t_data[item]
            print "PROGRESS %s: %f %%" % (item,i / float(len(test_index)) * 100.0)
            for user in user_list.keys():
                predicted = float(recommender.predict(user,item, test_set=[] ))
                real = float(t_data[item][user])
                top_sum = top_sum + pow( predicted - real ,2)
                bot_sum = bot_sum + 1.0
            i = i + 1.0

        stats['RMSE_LIST'].append( np.sqrt(top_sum / bot_sum) )
    stats['RMSE'] = np.average(stats['RMSE_LIST'])

    p_list = []
    r_list = []
    i = 0.0
    length = float(len(data.keys()))
    for user_id in data.keys():

        recommended_scores, recommended_list = recommender.recommend(user_id)
        relevants = get_relevant_items(model,user_id)

        p, r = get_precision_recall(recommended_list, relevants)
        print "PROGRESS %s: %f %%" % (user_id, i / length * 100.0)
        p_list.append(p)
        r_list.append(r)
        i = i + 1.0
    
    stats['MAP'] = np.average(np.average(p_list))
    stats['MAR'] = np.average(np.average(r_list))

    print stats

def main():
    data = get_train_dataset()
    t_data = get_train_dataset_traspose()

    model = RatingPreferenceMatrix(data, t_data)

    slope_recommender = SlopeOneRecommender(model)
    cross_validate_items(data,t_data,model,slope_recommender)

    item_sim    = AdjCosineItemSimilarity(model)
    item_recommender = ItemBasedRecommender(model, item_sim)
    cross_validate_items(data,t_data,model,item_recommender)

    user_sim = WeightedPearsonUserSimilarity(model)
    user_recommender = UserBasedRecommender(model, user_sim)
    cross_validate_users(data,t_data,model,user_recommender)
    

if __name__ == "__main__":
    main()
