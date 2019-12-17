# SCSE RecSys Group
# Minh N.
# 19.07.01
# This module takes a list of user-item-ratings and predicts ratings for each row based off of 
# the similarity and utility matrix.

import pandas as pd
import numpy as np
import math
import json
import time

def load_files(training_um_path, training_sm_path, test_set_path):
    with open(training_um_path, 'r') as f:
        training_um = json.load(f)
    with open(training_sm_path, 'r') as f:
        training_sm = json.load(f)
    test_set = pd.read_csv(test_set_path)
    return training_um, training_sm, test_set

def predict(training_um, training_sm, test_set, output_file_path):
    
    test_set['prediction'] = math.nan

    for num, (business, user) in enumerate(zip(test_set['business_id'], test_set['user_id']), start = 0):
        if business not in training_sm.keys():
            continue
        similarities = training_sm[business]

        # CURRENT ISSUES:
        # The training/test sets aren't split such that there is a guarantee of a person being in the training set

        ratings = []
        # This loop generates a list of lists where the inner lists has two values
        # The first is the similarity coefficient to the user and the second is the rating they've given the business
        for key, value in similarities.items():
            if user in training_um[key]:
                ratings.append([value, training_um[key][user]])

        # Checks if ratings are empty
        if len(ratings) == 0:
            continue
        else:
            # This determines how many ratings are taken into account
            #print('Predicting user ' + str(user) + ' rating on business ' + str(business))
            k = 30
            numerator = 0
            denominator = 0
            ratings.sort(key=lambda x : x[0])
            ratings = ratings[::-1]
            ratings = ratings[0:k]
            #print(ratings)
            #time.sleep(5)
            for rating in ratings:
                numerator += float(rating[0] * rating[1])
                denominator += float(abs(rating[0]))
        test_set.at[num, 'prediction'] = max(1, (numerator / (denominator + 0.000000000001)))
    print(test_set)
    return test_set

def main():
    training_um, training_sm, test_set = load_files('../datasets/yelp_dataset/utility-matrix/yelp_utility_matrix_uc_user_tr1.json','yelp_user_sim_matrix_cosine_tr1.json','../datasets/yelp_dataset/yelp_review_uc_testing_1.csv')
    predict(training_um, training_sm, test_set, 'yelp_user_prediction_cosine_1.csv')
    with open('yelp_user_sim_matrix_pearson_tr1.json', 'r') as f:
        training_sm = json.load(f)
    predict(training_um, training_sm, test_set, 'yelp_user_prediction_pearson_1.csv')    

if __name__ == '__main__':
    main()