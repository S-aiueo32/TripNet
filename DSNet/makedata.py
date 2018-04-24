import glob
import os
import copy
import itertools
import random
import pandas as pd


import argparse
parser = argparse.ArgumentParser(description='This Program Makes the List of Triplets.')
parser.add_argument("--data_dir", type=str, help="Specify the Data Directory.")
parser.add_argument("--loop", type=int, default=3 ,help="(Optional) Specify the Loop Count.")
args = parser.parse_args()


def get_attr_names(prefix, nums):
    names = []
    for i in nums:
        names.append(prefix + "%02d" % i)
    return names


def get_same_genre(all, target):
    same_genre = []
    for row in all:
        len_sub = len(set(row) - set(target))
        if len_sub == 2:
            same_genre.append(row)
    # if same_genre != []:    print(same_genre , "\n")
    return same_genre


def get_same_trend(all, target):
    same_trend = []
    for row in all:
        len_sub = len(set(row) - set(target))
        if len_sub == 1:
            same_trend.append(row)
    # if same_trend != []:    print(same_trend , "\n")
    return same_trend


def pick_random(others):
    return others[random.randint(0, len(others) - 1)]


def pick_random_list(pair, others, loop):
    negative = set([])
    while(len(negative) < loop):
        negative.add(pick_random(others))
    negative = list(negative)

    result = []
    for i in range(loop):
        row = []
        random.shuffle(pair)
        row.extend(pair)
        row.append(list(negative[i]))
        result.append(row)

    return result


def get_triplets(targets, others, loop=args.loop):
    positive_pairs = list(itertools.combinations(tuple(targets), 2))
    result = []
    for pair in positive_pairs:
        result.extend(pick_random_list(list(pair), others, loop))
    """
    result = []
    for pair in positive_pairs:
        result.extend(pick_random_list(list(pair), others, loop))
    """
    # if positive_pairs != []:
    #    print(positive_pairs, "\n")
    # return positive_pairs
    return result


def hashable(list):
    return map(tuple, list)


if __name__ == "__main__":
    data_dir = args.data_dir
    files = glob.glob(os.path.join(data_dir,'*.jpg'))

    attr_list = files
    for i, file in enumerate(attr_list):
        attr_list[i] = (file.split("\\")[1]).replace(".jpg", "").split("_")

    Y = get_attr_names('Y', range(15, 19))
    S = get_attr_names('S', range(1, 3))
    C = get_attr_names('C', range(1, 5))
    G = get_attr_names('G', range(1, 10))
    T = get_attr_names('T', range(1, 32))

    """
    same_genre = get_same_genre(attr_list, ['Y15', 'S02', 'C04', 'G05'])
    same_trend = get_same_trend(attr_list, ['Y15', 'S02', 'C04', 'G05', 'T19'])
    print(list(set(map(tuple,same_genre))-set(map(tuple,same_trend))))
    """
    result = []
    for y in Y:
        for s in S:
            for c in C:
                for g in G:
                    same_genre = get_same_genre(attr_list, [y, s, c, g])
                    for t in T:
                        same_trend = get_same_trend(attr_list, [y, s, c, g, t])
                        others = list(set(hashable(same_genre)) -
                                      set(hashable(same_trend)))
                        if len(others) != 0:
                            # print(pick_random(others))
                            triplets = get_triplets(same_trend, others)
                            if triplets != []:
                                result.extend(triplets)

    for i, row in enumerate(result):
        for j, name in enumerate(row):
            result[i][j] = "_".join(name) + ".jpg"

    # print(result)
    df = pd.DataFrame(result)
    df.to_csv("./triplets.csv", index=False, header=False)
