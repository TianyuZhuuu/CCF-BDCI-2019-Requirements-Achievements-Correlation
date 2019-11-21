import os
import random
import re
from collections import Counter
from functools import partial

import jieba
import numpy as np
import pandas as pd
import scipy as sp
import torch
import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import Dataset, Sampler
from transformers import BertTokenizer

import config_local


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def clean_str(s):
    regex = '[\n\t\\\\n]'
    s = re.sub(regex, ' ', s)
    s = re.sub(' +', ' ', s)
    return s


def read_requirements_or_achievements(path):
    guid2data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            indices = [i for i, tok in enumerate(line) if tok == '\'']
            assert len(indices) == 6
            guid = line[indices[0] + 1:indices[1]]
            title = clean_str(line[indices[2] + 1:indices[3]])
            content = clean_str(line[indices[4] + 1:indices[5]])
            guid2data[guid] = (title, content)
    return guid2data


def read_interrelation(path):
    relations = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            relations.append(tuple(line.strip().split(',')))
    return relations


class Instance:
    def __init__(self, guid, aid, rid, a_title, a_text, r_title, r_text, a_title_tokens, a_text_tokens, r_title_tokens,
                 r_text_tokens, input_ids, token_type_ids, level, label):
        self.guid = guid
        self.aid = aid
        self.rid = rid

        self.a_title = a_title
        self.a_text = a_text
        self.r_title = r_title
        self.r_text = r_text

        self.a_title_tokens = a_title_tokens
        self.a_text_tokens = a_text_tokens
        self.r_title_tokens = r_title_tokens
        self.r_text_tokens = r_text_tokens

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids

        self.level = level
        self.label = label

    def __str__(self):
        return f'{self.guid} || {self.aid} || {self.rid} || {self.a_title} || {self.a_text} || {self.r_title} || {self.r_text} || ' \
               f'{self.input_ids} || {self.token_type_ids} || {self.level} || {self.label}'


def prepare_instances(requirements_path, achievements_path, interrelation_path, tokenizer: BertTokenizer,
                      label_fn=None, filter=False, chunk=False, swap=False):
    rid2data = read_requirements_or_achievements(requirements_path)
    aid2data = read_requirements_or_achievements(achievements_path)
    interrelation = read_interrelation(interrelation_path)

    instances = []

    if chunk:
        text2chunkid = {}
        chunks = []
    else:
        text2chunkid = None
        chunks = None

    maxlength = 0

    with tqdm.tqdm(total=len(interrelation), desc='Preparing instances') as pbar:

        for idx, tup in enumerate(interrelation):
            guid, aid, rid, level = tup
            level = '0' if level == '' else level

            level = int(level)
            label = label_fn(guid, level)

            a_title, a_text = aid2data[aid]
            r_title, r_text = rid2data[rid]

            a_title_tokens = ' '.join(jieba.cut(a_title))
            a_text_tokens = ' '.join(jieba.cut(a_text))
            r_title_tokens = ' '.join(jieba.cut(r_title))
            r_text_tokens = ' '.join(jieba.cut(r_text))

            if chunk:
                if (a_title, a_text, r_title, r_text) not in text2chunkid:
                    text2chunkid[(a_title, a_text, r_title, r_text)] = len(chunks)
                    chunks.append([idx])
                else:
                    chunkid = text2chunkid[(a_title, a_text, r_title, r_text)]
                    chunks[chunkid].append(idx)

            if not swap:
                encode_rt = tokenizer.encode_plus(text=a_title, text_pair=r_title, add_special_tokens=True)
            else:
                encode_rt = tokenizer.encode_plus(text=r_title, text_pair=a_title, add_special_tokens=True)
            input_ids = encode_rt['input_ids']
            token_type_ids = encode_rt['token_type_ids']

            length = len(input_ids)
            if 'num_truncated_tokens' in encode_rt:
                length += encode_rt['num_truncated_tokens']
            maxlength = max(maxlength, length)

            instances.append(
                Instance(guid, aid, rid, a_title, a_text, r_title, r_text, a_title_tokens,
                         a_text_tokens, r_title_tokens, r_text_tokens, input_ids, token_type_ids, level, label))
            pbar.update(1)

    print(f'Max Length: {maxlength}')

    if chunk:
        if filter:
            inconsist_indices = set()
            non_redundant_indices = set()
            for chunk in chunks:
                inst_ids = chunk
                labels = [instances[i].level for i in inst_ids]
                if len(set(labels)) > 1:
                    inconsist_indices.update(inst_ids)
                non_redundant_indices.add(inst_ids[0])

            print(f'Found {len(inconsist_indices)} inconsistent instances')
            print(f'Found {len(non_redundant_indices)} non redundant instances')

            valid_instances = []
            for idx, instance in enumerate(instances):
                if idx not in inconsist_indices and idx in non_redundant_indices:
                    valid_instances.append(instance)
            instances = valid_instances

            text2chunkid = {}
            chunks = []
            for idx, inst in enumerate(instances):
                a_title, a_text = inst.a_title, inst.a_text
                r_title, r_text = inst.r_title, inst.r_text

                if (a_title, a_text, r_title, r_text) not in text2chunkid:
                    text2chunkid[(a_title, a_text, r_title, r_text)] = len(chunks)
                    chunks.append([idx])
                else:
                    chunkid = text2chunkid[(a_title, a_text, r_title, r_text)]
                    chunks[chunkid].append(idx)

    return instances, text2chunkid, chunks


def chunk_k_fold(instances, chunks, n_folds, biased=False):
    kf_train_indices = []
    kf_val_indices = []

    if chunks is None:
        dummy = np.zeros((len(instances), 1))

        if biased:
            def bias_mapping(level):
                if level == 1 or level == 4:
                    return 3
                else:
                    return level - 1

            labels = np.array([bias_mapping(inst.level) for inst in instances])
        else:
            labels = np.array([inst.level for inst in instances])

        kf = StratifiedKFold(n_folds, shuffle=True, random_state=config_local.kfold_split_seed)
        for fold, (train_inst_indices, val_inst_indices) in enumerate(kf.split(dummy, labels)):
            kf_train_indices.append(train_inst_indices)
            kf_val_indices.append(val_inst_indices)
    else:
        chunk_levels = []
        for chunk in chunks:
            insts_levels = [instances[idx].level for idx in chunk]
            label = max(insts_levels, key=insts_levels.count)
            chunk_levels.append(label)

        dummy = np.zeros((len(chunks), 1))
        chunk_levels = np.array(chunk_levels)
        kf = StratifiedKFold(n_folds, shuffle=True, random_state=config_local.seed1)

        for fold, (train_chunk_indices, val_chunk_indices) in enumerate(kf.split(dummy, chunk_levels)):
            train_inst_indices = [inst_id for chunk_id in train_chunk_indices for inst_id in chunks[chunk_id]]
            val_inst_indices = [inst_id for chunk_id in val_chunk_indices for inst_id in chunks[chunk_id]]
            kf_train_indices.append(train_inst_indices)
            kf_val_indices.append(val_inst_indices)

    return kf_train_indices, kf_val_indices


def get_bert_dir_and_config(model_name):
    assert model_name in ('bert', 'roberta_base', 'roberta_large')
    if model_name == 'bert':
        model_dir = config_local.bert_base_dir
        config_path = config_local.bert_base_config_path
    elif model_name == 'roberta_base':
        model_dir = config_local.roberta_base_dir
        config_path = config_local.roberta_base_config_path
    else:
        model_dir = config_local.roberta_large_dir
        config_path = config_local.roberta_large_config_path
    return model_dir, config_path


def get_xlnet_dir_and_config(model_name):
    assert model_name in ('xlnet_base', 'xlnet_mid')
    if model_name == 'xlnet_base':
        model_dir = config_local.pretrained_xlnet_base_dir
        config_path = config_local.pretrained_xlnet_base_config_path
    else:
        model_dir = config_local.pretrained_xlnet_mid_dir
        config_path = config_local.pretrained_xlnet_mid_config_path
    return model_dir, config_path


def get_ckpt_template(approach, swap, filter, chunk, radam, bias_split, multidrop, model_name, learning_rate,
                      batch_size, n_folds, n_epochs, warmup, wd, seed, submit_csv_path, pseudo_label_path, **kwargs):
    if submit_csv_path is not None:
        pseudo_part = f'hard.pseudo.'
    elif pseudo_label_path is not None:
        pseudo_part = f'soft.pseudo.'
    else:
        pseudo_part = ''

    ckpt_template = '../new_ckpt/' + pseudo_part + ('swap.' if swap else '') + ('biased_split.' if bias_split else '') + \
                    ('filter.' if filter else '') + ('chunk.' if chunk else '') + ('radam.' if radam else '') + \
                    ('msd.' if multidrop else '') + approach + '.' + model_name + \
                    (f'.beta{kwargs["beta"]}' if "beta" in kwargs and kwargs["beta"] is not None else '') + \
                    f'.lr{learning_rate}.bs{batch_size}.nfold{n_folds}.nepoch{n_epochs}.warmup{warmup}.wd{wd}.seed{seed}/' + \
                    'fold{}.epoch{}.loss{}.score{}.ckpt'

    return ckpt_template


def model_transformation(model, device, n_gpu, ckpt_path=None):
    model.to(device)

    if n_gpu > 1:
        model = nn.DataParallel(model)

    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
        print(f'Load checkpoint from {ckpt_path}')

    return model


class NLIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def getkeys(self):
        return np.fromiter([len(example.input_ids) for example in self.examples],
                           dtype=np.int)


def bert_pad_sequences(examples):
    batch = len(examples)
    max_len = max(len(example.input_ids) for example in examples)
    padded_input_ids = torch.zeros(batch, max_len).long()
    padded_token_type_ids = torch.zeros(batch, max_len).long()
    padded_attention_mask = torch.zeros(batch, max_len).long()
    for i, example in enumerate(examples):
        _len = len(example.input_ids)
        padded_input_ids[i, :_len] = torch.LongTensor(example.input_ids)
        padded_token_type_ids[i, :_len] = torch.LongTensor(example.token_type_ids)
        padded_attention_mask[i, :_len] = 1
    return padded_input_ids, padded_token_type_ids, padded_attention_mask


# bucket iterator
def divide_chunks(l, n):
    if n == len(l):
        yield np.arange(len(l), dtype=np.int32), l
    else:
        for i in range(0, len(l), n):
            data = l[i:i + n]
            yield np.arange(i, i + len(data), dtype=np.int32), data


def prepare_buckets(lens, bucket_size, batch_size, shuffle_data=True, indices=None):
    lens = -lens
    assert bucket_size % batch_size == 0 or bucket_size == len(lens)
    if indices is None:
        if shuffle_data:
            indices = shuffle(np.arange(len(lens), dtype=np.int32))
            lens = lens[indices]
        else:
            indices = np.arange(len(lens), dtype=np.int32)
    new_indices = []
    extra_batch = None
    for chunk_index, chunk in divide_chunks(lens, bucket_size):
        indices_sorted = chunk_index[np.argsort(chunk, axis=-1)]
        batches = []
        for _, batch in divide_chunks(indices_sorted, batch_size):
            if len(batch) == batch_size:
                batches.append(batch.tolist())
            else:
                assert extra_batch is None
                assert batch is not None
                extra_batch = batch
        if shuffle_data:
            batches = shuffle(batches)
        for batch in batches:
            new_indices.extend(batch)
    if extra_batch is not None:
        new_indices.extend(extra_batch)
    return indices[new_indices]


class BucketSampler(Sampler):
    def __init__(self, data_source, sort_keys, batch_size, bucket_size=None, shuffle_data=True):
        super(BucketSampler, self).__init__(data_source)
        self.shuffle = shuffle_data
        self.batch_size = batch_size
        self.sort_keys = sort_keys
        self.bucket_size = bucket_size if bucket_size is not None else len(sort_keys)
        if not shuffle_data:
            self.index = prepare_buckets(self.sort_keys, bucket_size=self.bucket_size, batch_size=self.batch_size,
                                         shuffle_data=shuffle_data)
        else:
            self.index = None

    def __iter__(self):
        if self.shuffle:
            self.index = prepare_buckets(self.sort_keys, bucket_size=self.bucket_size, batch_size=self.batch_size,
                                         shuffle_data=self.shuffle)
        return iter(self.index)

    def __len__(self):
        return len(self.sort_keys)


def eval_function(predictions, true_labels):
    predictions = np.array(predictions).astype(np.int)
    true_labels = np.array(true_labels).astype(np.int)
    abs_diff = np.abs(predictions - true_labels)
    mae = np.mean(abs_diff)
    score = 1.0 / (1.0 + mae)
    return score


def generate_soft_label(label, num_labels):
    padded_label = torch.arange(1, num_labels + 1).float()  # [4]
    padded_label = torch.abs(padded_label - label)
    padded_label = torch.exp(-2 * padded_label)
    padded_label = padded_label / padded_label.sum(dim=-1, keepdim=True)
    return padded_label


def eval_function(predictions, true_labels):
    predictions = np.array(predictions).astype(np.int)
    true_labels = np.array(true_labels).astype(np.int)
    abs_diff = np.abs(predictions - true_labels)
    mae = np.mean(abs_diff)
    score = 1.0 / (1.0 + mae)
    return score


class OptimizedRounder1(object):
    def __init__(self):
        self.coef_ = 0

    def _mae_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 1
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 2
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        score = eval_function(X_p, y)
        return -score

    def fit(self, X, y):
        loss_partial = partial(self._mae_loss, X=X, y=y)
        initial_coef = [1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 1
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 2
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i + 1
    return len(borders) + 1


class OptimizedRounder2(object):
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        ll = -eval_function(X_p, y)
        return ll

    def fit(self, X, y):
        coef = [2.0, 2.5, 3.0]
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [(1.5, 2.5), (2, 3), (2.5, 3.5)]
        for it1 in range(10):
            for idx in range(3):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                coef[idx] = a
                la = self._loss(coef, X, y)
                coef[idx] = b
                lb = self._loss(coef, X, y)
                for it in range(20):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        coef[idx] = a
                        la = self._loss(coef, X, y)
                    else:
                        b = b - (b - a) * golden2
                        coef[idx] = b
                        lb = self._loss(coef, X, y)
        self.coef_ = {'x': coef}

    def predict(self, X, coef):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        return X_p

    def coefficients(self):
        return self.coef_['x']


def distribution_rounder(guid2score):
    df = pd.read_csv(config_local.interrelation_path)
    vals = df.Level.values
    counter = Counter(vals)
    counts = np.array([counter[i] for i in range(1, 5)])
    distributions = counts / counts.sum()
    distributions = np.cumsum(distributions)

    guids = np.array(list(guid2score.keys()))
    scores = np.array(list(guid2score.values()))
    sorted_indices = np.argsort(scores)

    pivots = (distributions * len(guid2score)).astype(np.int) - 1
    thresholds = scores[sorted_indices][pivots][:-1]
    levels = np.digitize(scores, thresholds) + 1
    guid2level = {guid: level for guid, level in zip(guids, levels)}

    return guid2level


def thresholds_rounder(guid2score, thresholds):
    guids = np.array(list(guid2score.keys()))
    scores = np.array(list(guid2score.values()))
    levels = np.digitize(scores, thresholds) + 1
    guid2level = {guid: level for guid, level in zip(guids, levels)}
    return guid2level


def get_confident_test_example_guids(*submit_paths):
    submit_files = [pd.read_csv(path) for path in submit_paths]
    guids = submit_files[0].Guid.values
    preds = [list(file.Level.values) for file in submit_files]
    inst_preds = zip(*preds)
    conf_guids = set()
    for guid, pred in zip(guids, inst_preds):
        if len(set(pred)) == 1:
            conf_guids.add(guid)
    return conf_guids


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(config_local.bert_base_dir)
    instances, _, _ = prepare_instances(config_local.requirements_path, config_local.train_achievements_path,
                                        config_local.interrelation_path,
                                        tokenizer, label_fn=lambda guid, level: int(level), filter=True, chunk=False)
    kf_train_indices, kf_val_indices = chunk_k_fold(instances, None, 10, biased=True)

    for a, b in zip(kf_train_indices, kf_val_indices):
        print(b[:10])

    # inst = instances[0]
    # print(inst.a_title_tokens)
    # print(inst.a_text_tokens)
    # print(inst.r_title_tokens)
    # print(inst.r_text_tokens)
    # print(len(instances))
    pass
