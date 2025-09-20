import numpy as np
import random
import torch

from rank_bm25 import BM25Okapi, BM25Plus, BM25L
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

import data


def bm25(test_srcs, train_srcs, train_idxs=[], mode='BM25Okapi', n=4):
    test_srcs = [test_src.lower().split() for test_src in test_srcs]
    train_srcs = [train_src.lower().split() for train_src in train_srcs]

    if mode == 'BM25Okapi':
        bm25 = BM25Okapi(train_srcs)
    elif mode == 'BM25Plus':
        bm25 = BM25Plus(train_srcs)
    elif mode == 'BM25L':
        bm25 = BM25L(train_srcs)
    else:
        raise ValueError("mode must be one of BM25Okapi, BM25Plus, BM25L")
    
    top_n_index = []
    for i in tqdm(range(len(test_srcs)), ncols=60):
        test_src = test_srcs[i]
        score = bm25.get_scores(test_src)
        if train_idxs == []:
            train_idx = list(range(len(train_srcs)))
        else:
            train_idx = train_idxs[i]
        score_ = []
        for i in train_idx:
            score_.append(score[i])
        top_n_index.append(np.argsort(score_)[::-1][:n])

    return top_n_index


def cls(test_srcs, train_srcs, bert_dir, train_idxs=[], n=1000, device='cuda:0'):
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    model = BertModel.from_pretrained(bert_dir)
    model.eval()
    model.to(device)

    test_srcs = [test_src.lower().split() for test_src in test_srcs]
    train_srcs = [train_src.lower().split() for train_src in train_srcs]

    train_vecs, test_vecs = [], []
    top_n_index = []
    with torch.no_grad():
        for train_src in tqdm(train_srcs, ncols=60):
            marked_text = ["[CLS]"] + train_src + ["[SEP]"]
            indexed_tokens = tokenizer.convert_tokens_to_ids(marked_text)
            segments_ids = [1] * len(marked_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            tokens_tensor = tokens_tensor.to(device)
            segments_tensors = segments_tensors.to(device)
            outputs = model(tokens_tensor, segments_tensors)
            cls_head = outputs[1][0]
            train_vecs.append(cls_head.cpu())
        for test_src in tqdm(test_srcs, ncols=60):
            marked_text = ["[CLS]"] + test_src + ["[SEP]"]
            indexed_tokens = tokenizer.convert_tokens_to_ids(marked_text)
            segments_ids = [1] * len(marked_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            tokens_tensor = tokens_tensor.to(device)
            segments_tensors = segments_tensors.to(device)
            outputs = model(tokens_tensor, segments_tensors)
            cls_head = outputs[1][0]
            test_vecs.append(cls_head.cpu())
        
        for i in tqdm(range(len(test_vecs)), ncols=60):
            test_vec = test_vecs[i]
            score = []
            if train_idxs == []:
                train_idx = list(range(len(train_vecs)))
            else:
                train_idx = train_idxs[i]
            for j in train_idx:
                train_vec = train_vecs[j]
                score.append(torch.cosine_similarity(test_vec, train_vec, dim=0))
            top_n_index.append(np.argsort(score)[::-1][:n])

    return top_n_index


def write_idx(fn, idxs):
    with open(fn, 'w', encoding='utf8') as out:
        for i in range(len(idxs)):
            out.write(' '.join([str(j) for j in idxs[i]]) + '\n')


def selection4all(args):
    test_data_list = ['bea19', 'conll14']
    n = args.n

    bert_dir = args.bert_dir
    device = args.device

    for test_data in test_data_list:
        print('-'*80)
        print(test_data)
        test_fn = '../data/' + test_data + '/test.src'
        out_dir = '../data/' + test_data + '/index/'
        train_fn = '../data/wi+locness/train.src'

        test_srcs = data.read_lines(test_fn)
        train_srcs = data.read_lines(train_fn)

        random_n_index = [random.sample(range(len(train_srcs)), n) for _ in range(len(test_srcs))]
        write_idx(out_dir + 'random.txt', random_n_index)

        cls_n_index = cls(test_srcs, train_srcs, n=n, bert_dir=bert_dir, device=device)
        write_idx(out_dir + 'bert.txt', cls_n_index)

        bm25_n_index = bm25(test_srcs, train_srcs, n=n)
        write_idx(out_dir + 'bm25.txt', bm25_n_index)

        print('='*80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()
    selection4all(args)