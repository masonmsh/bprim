""" An example of using this library to calculate related artists
from the last.fm dataset. More details can be found
at http://www.benfrederickson.com/matrix-factorization/
This code will automically download a HDF5 version of the dataset from
GitHub when it is first run. The original dataset can also be found at
http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html.
"""
import argparse
from time import time
from Dataset import Dataset
import numpy as np
import math

from implicit.bpr import BayesianPersonalizedRanking

MODELS = {"bpr": BayesianPersonalizedRanking}


def parse_args():
    parser = argparse.ArgumentParser(description="run bpr")
    parser.add_argument('--model', type=str, default='bpr', dest='model',
                        help='model to calculate (%s)' % "/".join(MODELS.keys()))
    parser.add_argument('--dataset', type=str, default='Home_and_Kitchen', help='dataset')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--factors', type=int, default=64, help='factors')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--reg', type=float, default=0,
                        help='regularization')
    parser.add_argument('--iter', type=int, default=10000, help='iterations')

    return parser.parse_args()


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


def calculate_recommendations_all(model_name="bpr", K=10, epochs=100, factors=64, iter=100, lr=0.001, reg=0.01):
    """ Generates artist recommendations for each user in the dataset """
    # train the model based off input params
    # create a model from the input data
    model = get_model(model_name, factors, iter, lr, reg)
    dataset = Dataset('data/'+args.dataset)
    train, train_filter, testRatings, testNegatives = dataset.trainMatrix, dataset.trainfilter, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print(num_users, num_items)
    filter_items = {}
    for idx in range(num_users):
        filter_items[idx] = set(testNegatives[idx] + [testRatings[idx][1]])
    plays = train.T.tocsr()
    # this is actually disturbingly expensive:
    plays = plays.tocsr()
    best_hit, best_ndcg, best_epoch = 0, 0, 0
    for num in range(epochs):
        t1 = time()
        model.fit(plays, show_progress=False)
        # generate recommendations for each user and write out to a file
        hits, ndcgs = [], []
        itemid = model.recommend_all(user_items=train.tocsr(
        ), filter_already_liked_items=False, N=num_items, show_progress=False)
        for idx in range(num_users):
            ranklist = []
            for i in range(len(itemid[idx])):
                if (itemid[idx][i] in filter_items[idx]) and (itemid[idx][i] not in train_filter[idx])and len(ranklist) < K:
                    ranklist.append(itemid[idx][i])
                elif len(ranklist) >= K:
                    break
            gtItem = testRatings[idx][1]
            hr = getHitRatio(ranklist, gtItem)
            ndcg = getNDCG(ranklist, gtItem)
            hits.append(hr)
            ndcgs.append(ndcg)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Epoch %d [%.1f s]: HR = %.4f, NDCG = %.4f' %
              (num, time()-t1, hr, ndcg))
        if hr > best_hit:
            best_hit, best_ndcg, best_epoch = hr, ndcg, num
    print('The best epoch is %d, HR = %.4f, NDCG = %.4f' %
          (best_epoch, best_hit, best_ndcg))


def get_model(model_name, factors=64, iter=100, lr=0.001, reg=0.01):
    print("getting model %s" % model_name)
    model_class = MODELS.get(model_name)
    if not model_class:
        raise ValueError("Unknown Model '%s'" % model_name)
    params = {'factors': factors, 'iterations': iter, 'learning_rate': lr,
              'regularization': reg, 'verify_negative_samples': True}
    return model_class(**params)


if __name__ == "__main__":
    args = parse_args()

    calculate_recommendations_all(model_name=args.model, K=10, epochs=args.epochs,
                                  factors=args.factors, iter=args.iter, lr=args.lr, reg=args.reg)
