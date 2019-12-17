# SCSE RecSys Group
# Minh N. and Chen T.
# 19.06.18
# This module builds two user similarity matrices from a Yelp utility matrix
# The similiarities are cosine distance and Pearson Correlation.
# Can be used to create both item and user similarities matricies.

import json
import math
import time

def dict_squared_sum(dict_to_sum):
    summation = 0
    for i in dict_to_sum.values():
        summation += i * i
    return summation

def similarity_cosine(um_dict, output_json):
    sim_dict = {}
    for key_i, value_i in um_dict.items():
        curr_dict = {}
        for key_j, value_j in um_dict.items():
            if key_j in sim_dict:
                if key_i in sim_dict[key_j]:
                    curr_dict[key_j] = sim_dict[key_j][key_i]
                else:
                    continue
            else:
                intersection = value_i.keys() & value_j.keys()
                if len(intersection) == 0:
                    continue
                else:
                    dot_product = 0
                    magnitudes = 0
                    # intersection = list(intersection)
                    for i in intersection:
                        dot_product += value_i[i] * value_j[i]
                    magnitudes = math.sqrt(dict_squared_sum(value_i)) * math.sqrt(dict_squared_sum(value_j))
                curr_dict[key_j] = dot_product / magnitudes
        sim_dict[key_i] = curr_dict
    with open(output_json, 'w') as f:
        json_dump = json.dumps(sim_dict)
        f.write(json_dump)
        f.close()
    return sim_dict

def similarity_pearson(um_dict, output_json):
    sim_dict = {}
    for key_i, value_i in um_dict.items():
        curr_dict = {}
        for key_j, value_j in um_dict.items():
            if key_j in sim_dict:
                if key_i in sim_dict[key_j]:
                    curr_dict[key_j] = sim_dict[key_j][key_i]
                else:
                    continue
            else:
                intersection = value_i.keys() & value_j.keys()
                if len(intersection) == 0:
                    continue
                else:
                    # Isolating corrated cases
                    corrated_i = [value_i[i] for i in intersection]
                    corrated_j = [value_j[j] for j in intersection]

                    # Getting the average
                    average_i = sum(corrated_i) / len(corrated_i)
                    average_j = sum(corrated_j) / len(corrated_j)

                    #updating to the necessary pre-components
                    corrated_i = [i - average_i for i in corrated_i]
                    corrated_j = [j - average_j for j in corrated_j]

                    numerator = sum((i * j for i, j in zip(corrated_i, corrated_j)))
                    denomiator = math.sqrt(sum((i * i for i in corrated_i))) * math.sqrt(sum((j * j for j in corrated_j)))
                    if denomiator:
                        curr_dict[key_j] = numerator/denomiator
                    else:
                        continue
        sim_dict[key_i] = curr_dict
    with open(output_json, 'w') as f:
        json_dump = json.dumps(sim_dict)
        f.write(json_dump)
        f.close()
    return sim_dict

def read_json(path):
    f = open(path, 'r')
    return json.load(f)

def main():
    um_loaded = read_json('../datasets/yelp_dataset/utility-matrix/yelp_utility_matrix_uc_user_tr5.json')
    user_similarity_cosine(um_loaded, 'yelp_user_sim_matrix_cosine_tr5.json')
    user_similarity_pearson(um_loaded, 'yelp_user_sim_matrix_pearson_tr5.json')

if __name__ == '__main__':
    main()