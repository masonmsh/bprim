import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from time import time
from time import strftime
from time import localtime
import logging

from implicit.bpr import BayesianPersonalizedRanking


def parse_args():
    parser = argparse.ArgumentParser(description="BPR")
    parser.add_argument('--dataset', type=str, default='_', help='d')
    parser.add_argument('--dim', type=int, default=16, help='d')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--reg', type=float, default=0, help='r')
    parser.add_argument('--epochs', type=int, default=2, help='e')
    parser.add_argument('--iters', type=int, default=10000, help='iter')
    parser.add_argument('--testbatch', type=int, default=2, help="t")
    parser.add_argument('--topk', type=int, default=10, help="t")
    parser.add_argument('--gpu', type=int, default=0, help="g")
    return parser.parse_args()


def default_args():
    return 'Musical_Patio'


def load_data(dataset):
    trainp = 'data/%s/train.csv' % dataset
    testp = 'data/%s/test.csv' % dataset
    fp = 'data/%s/info.txt' % dataset

    with open(fp, 'r') as f:
        _, _, maxuser = map(int, f.readline().strip().split())
        It_idx, maxitem = map(int, f.readline().strip().split())

    train = pd.read_csv(trainp, ',', names=[
                        'u', 'i', 'r', 't'], engine='python')
    test = pd.read_csv(testp, ',', names=['u', 'i', 'r', 't'], engine='python')

    train = train[['u', 'i', 'r']]
    test = test[['u', 'i']]
    train = [[int(x[0]), int(x[1]), 1.0] for x in train.values]
    test = [[int(x[0]), int(x[1])] for x in test.values]

    mat = sp.dok_matrix((maxuser+1, maxitem+1), dtype=np.float32)
    filter_dict = {}
    for i in range(len(train)):
        user, item, rating = train[i]
        mat[user, item] = rating
        if user not in filter_dict:
            filter_dict[user] = set()
        filter_dict[user].add(item)
    return mat, filter_dict, test, It_idx


def get_model(dim, lr, reg, iters, gpu):
    model_class = BayesianPersonalizedRanking
    params = {'factors': dim, 'learning_rate': lr, 'regularization': reg,
              'iterations': iters, 'verify_negative_samples': True, 'use_gpu': True if gpu else False}
    return model_class(**params)


def get_label(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def recall_precision(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def ndcg_k(test_data, r, k):
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(X, k):
    sorted_items = X[0]
    groundTrue = X[1]
    r = get_label(groundTrue, sorted_items)
    ret = recall_precision(groundTrue, r, k)
    ndcg = ndcg_k(groundTrue, r, k)
    return {'precision': ret['precision'], 'recall': ret['recall'], 'ndcg': ndcg}


def main(dataset, dim, lr, reg, epochs, iters, testbatch, topk, gpu):
    train, train_filter, test, i_begin = load_data(dataset)
    model = get_model(dim, lr, reg, iters, gpu)
    num_items = train.shape[1]
    test_df = pd.DataFrame(test, columns=['u', 'i'])
    results = {'precision': 0, 'recall': 0, 'ndcg': 0}
    n_user = max(test_df['u'])+1

    best = [0, 0, 0]
    for epoch in range(epochs):
        print('===EPOCH %d===' % epoch)
        logging.info('===EPOCH %d===' % epoch)
        start = time()
        plays = train.T.tocsr()
        model.fit(plays, show_progress=False)
        print('Train Time [%.1f]' % (time()-start))
        logging.info('Train Time [%.1f]' % (time()-start))
        start = time()
        ratings = []
        truths = []
        rating = []
        truth = []
        for user in range(n_user):
            items = test_df[test_df['u'] == user]['i'].values - i_begin
            itemid = model.recommend(userid=user, user_items=train.tocsr(
            ), filter_already_liked_items=False, N=num_items)
            rating_A = [itemid[i][0] for i in range(len(itemid))]
            rating_K = []

            for i in range(len(rating_A)):
                if (rating_A[i] >= i_begin) and (rating_A[i] not in train_filter[user]) and (len(rating_K) < topk):
                    rating_K.append(rating_A[i])
                elif len(rating_K) >= topk:
                    break
            if len(rating) < testbatch:
                rating.append(rating_K)
                truth.append(items.tolist())
            else:
                ratings.append(np.array(rating)-i_begin)
                truths.append(truth)
                rating = [rating_K]
                truth = [items.tolist()]
                print('%d / %d [%.1f]' % (user, n_user, time()-start))
                logging.info('%d / %d [%.1f]' % (user, n_user, time()-start))
                start = time()
        if len(rating) > 0:
            ratings.append(np.array(rating)-i_begin)
            truths.append(truth)
        X = zip(ratings, truths)
        test_results = []
        for x in X:
            test_results.append(test_one_batch(x, topk))
        for result in test_results:
            results['precision'] += result['precision']
            results['recall'] += result['recall']
            results['ndcg'] += result['ndcg']
        results['precision'] /= float(n_user)
        results['recall'] /= float(n_user)
        results['ndcg'] /= float(n_user)
        print('[%d] precision = %.4f, recall = %.4f, NDCG = %.4f' %
              (topk, results['precision'], results['recall'], results['ndcg']))
        logging.info('[%d] precision = %.4f, recall = %.4f, NDCG = %.4f' % (
            topk, results['precision'], results['recall'], results['ndcg']))
        if best[1] < results['recall']:
            best[0] = results['precision']
            best[1] = results['recall']
            best[2] = results['ndcg']
    print('BEST[%d] precision = %.4f, recall = %.4f, NDCG = %.4f' %
          (topk, best[0], best[1], best[2]))
    logging.info('BEST[%d] precision = %.4f, recall = %.4f, NDCG = %.4f' %
                 (topk, best[0], best[1], best[2]))


if __name__ == "__main__":
    args = parse_args()
    # args.dataset = default_args()
    log_dir = "log/%s/" % args.dataset
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, "%s_%s_%s" % (
        args.dataset, args.dim, strftime('%Y-%m-%d--%H-%M-%S', localtime()))), level=logging.INFO)

    main(args.dataset, args.dim, args.lr, args.reg, args.epochs,
         args.iters, args.testbatch, args.topk, args.gpu)
