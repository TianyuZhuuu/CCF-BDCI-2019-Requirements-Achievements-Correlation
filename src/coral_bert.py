import os
import pickle
import random

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, AdamW, WarmupLinearSchedule, BertModel, \
    BertTokenizer

import config_local
from radam import RAdam
from utils import NLIDataset, BucketSampler, eval_function, bert_pad_sequences, seed_everything, \
    get_bert_dir_and_config, get_ckpt_template, model_transformation, prepare_instances, chunk_k_fold

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def extend_label(label):
    # label: 1,2,3,4
    padded_label = torch.zeros(config_local.num_labels - 1)
    padded_label[:label - 1] = 1.0
    return padded_label


def collate_fn(examples):
    guids = [example.guid for example in examples]
    labels = [example.label for example in examples]
    input_ids, token_type_ids, attention_mask = bert_pad_sequences(examples)
    labels = torch.stack(labels, dim=0)
    return input_ids, token_type_ids, attention_mask, labels, guids


def get_task_weight(examples):
    S = torch.zeros(config_local.num_labels - 1)
    for example in examples:
        S += example.label
    M = torch.max(S, len(examples) - S)
    lamda = torch.sqrt(M)
    lamda = lamda / lamda.max()
    lamda = lamda.unsqueeze(0)
    return lamda


class CORALBERT(nn.Module):
    def __init__(self, model_dir, config_path):
        super(CORALBERT, self).__init__()
        bert_config = BertConfig.from_pretrained(config_path)
        self.bert = BertModel.from_pretrained(model_dir, config=bert_config)
        self.classifier = nn.Linear(bert_config.hidden_size, config_local.num_labels - 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, config_local.num_labels - 1))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        pooler_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[1]
        logits = self.classifier(pooler_output) + self.bias
        return logits


class MSDCORALBERT(nn.Module):
    def __init__(self, model_dir, config_path):
        super(MSDCORALBERT, self).__init__()
        bert_config = BertConfig.from_pretrained(config_path)
        self.bert = BertModel.from_pretrained(model_dir, config=bert_config)
        self.classifier = nn.Linear(bert_config.hidden_size, config_local.num_labels - 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, config_local.num_labels - 1))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, task_weight=None, n_drops=10,
                p_drop=0.1):
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        pooler_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[1]

        # Training
        if labels is not None and task_weight is not None:
            for i in range(n_drops):
                logits = self.classifier(F.dropout(pooler_output, p_drop, training=self.training)) + self.bias
                if i == 0:
                    loss = (loss_fn(logits, labels) * task_weight).sum(dim=-1).mean()
                else:
                    loss += (loss_fn(logits, labels) * task_weight).sum(dim=-1).mean()
            loss /= n_drops
            return loss
        else:
            # Eval
            logits = self.classifier(pooler_output) + self.bias
            return logits,


def get_coral_bert(model_dir, config_path, multidrop=True):
    if multidrop:
        model = MSDCORALBERT(model_dir, config_path)
    else:
        model = CORALBERT(model_dir, config_path)
    return model


def kfold_training(n_folds, swap, filter, chunk, radam, biased_split, learning_rate, batch_size, warmup, wd, n_epochs,
                   multidrop=True, model_name='bert', submit_csv_path=None, pseudo_label_path=None,
                   conf_guids_path=None, retrain_fold=None, seed=42):
    seed_everything(seed)
    model_dir, config_path = get_bert_dir_and_config(model_name)

    ckpt_template = get_ckpt_template('coral', swap, filter, chunk, radam, biased_split, multidrop, model_name,
                                      learning_rate, batch_size, n_folds, n_epochs, warmup, wd, seed, submit_csv_path,
                                      pseudo_label_path)

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    examples, text2chunkid, chunks = prepare_instances(config_local.requirements_path, config_local.train_achievements_path,
                                                       config_local.interrelation_path, tokenizer,
                                                       label_fn=lambda guid, label: extend_label(label), chunk=chunk,
                                                       swap=swap)

    pseudo_exmaples = []
    if submit_csv_path:
        submit_df = pd.read_csv(submit_csv_path)
        guids = submit_df.Guid.values
        levels = submit_df.Level.values
        guid2level = {guid: level for guid, level in zip(guids, levels)}
        pseudo_label_fn = lambda guid, level: extend_label(guid2level[guid])
        pseudo_exmaples, _, _ = prepare_instances(config_local.requirements_path, config_local.test_achievements_path,
                                                  config_local.test_prediction_path, tokenizer, label_fn=pseudo_label_fn,
                                                  swap=swap)

        if conf_guids_path:
            with open(conf_guids_path, 'rb') as f:
                conf_guids = pickle.load(f)
            pseudo_exmaples = [example for example in pseudo_exmaples if example.guid in conf_guids]
            print(f'Use {len(pseudo_exmaples)} confident pseudo examples')

    if pseudo_label_path:
        with open(pseudo_label_path, 'rb') as f:
            guid2pred = pickle.load(f)
        pseudo_label_fn = lambda guid, level: guid2pred[guid]

        pseudo_exmaples, _, _ = prepare_instances(config_local.requirements_path, config_local.test_achievements_path,
                                                  config_local.test_prediction_path, tokenizer, label_fn=pseudo_label_fn,
                                                  chunk=False, swap=swap)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f'Let\'s use {n_gpu} GPUs!!!')

    kf_train_indices, kf_val_indices = chunk_k_fold(examples, chunks, n_folds, biased=biased_split)
    for fold in range(len(kf_train_indices)):
        if retrain_fold is not None and (fold + 1) != retrain_fold:
            continue

        if retrain_fold == 1:
            seed_everything(seed + 1)

        train_index = kf_train_indices[fold]
        val_index = kf_val_indices[fold]

        print(f'FOLD {fold + 1}')

        train_examples = [examples[i] for i in train_index] + pseudo_exmaples
        valid_examples = [examples[i] for i in val_index]

        print(f'# of train examples: {len(train_examples)}')
        print(f'# of val   examples: {len(valid_examples)}')

        task_weight = get_task_weight(train_examples).to(device)

        train_dataset = NLIDataset(train_examples)
        valid_dataset = NLIDataset(valid_examples)

        gradient_accumulation_steps = batch_size // 8
        if model_name == 'roberta_large':
            gradient_accumulation_steps = batch_size // 4
        batch_size = batch_size // gradient_accumulation_steps
        batch_per_bucket = max(1, int(len(train_dataset) / batch_size / 20))

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=BucketSampler(train_dataset, train_dataset.getkeys(),
                                                        batch_size=batch_size,
                                                        bucket_size=batch_size * batch_per_bucket),
                                  collate_fn=collate_fn)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                sampler=BucketSampler(valid_dataset, valid_dataset.getkeys(),
                                                      batch_size=batch_size, shuffle_data=False),
                                collate_fn=collate_fn)

        t_total = np.ceil(len(train_loader) / gradient_accumulation_steps) * n_epochs
        warmup_steps = int(t_total * warmup)

        model = get_coral_bert(model_dir, config_path, multidrop)
        model = model_transformation(model, device, n_gpu)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': wd},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if not radam:
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                              correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        else:
            print('Using RAdam')
            optimizer = RAdam(optimizer_grouped_parameters, lr=learning_rate)

        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)  # PyTorch scheduler

        loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        best_score = -1
        prev_best_ckpt = None

        for epoch in range(1, n_epochs + 1):
            avg_loss = 0.0

            model.train()
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch} Train') as pbar:
                for step, data in enumerate(train_loader):
                    input_ids, token_type_ids, attention_mask, labels = tuple(t.to(device) for t in data[:4])

                    if multidrop:
                        n_drops = random.randint(1, 10)
                        p_drop = min(n_drops / 10.0, 0.5)
                        loss = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                     labels=labels, task_weight=task_weight, n_drops=n_drops, p_drop=p_drop)
                    else:
                        logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
                        loss = (loss_fn(logits, labels) * task_weight).sum() / logits.size(0)

                    if n_gpu > 1:
                        loss = loss.mean()
                    if gradient_accumulation_steps > 1:
                        loss /= gradient_accumulation_steps
                    loss.backward()
                    avg_loss += loss.item() / len(train_loader)
                    if (step + 1) % gradient_accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    pbar.update(1)
            avg_loss *= gradient_accumulation_steps

            val_loss = 0
            model.eval()
            epoch_preds, epoch_labels = [], []
            with tqdm(total=len(val_loader), desc=f'Epoch {epoch} Val  ') as pbar:
                with torch.no_grad():
                    for step, data in enumerate(val_loader):
                        # input_ids, token_type_ids, attention_mask, labels, guids
                        input_ids, token_type_ids, attention_mask, labels = tuple(t.to(device) for t in data[:4])

                        logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
                        loss = (loss_fn(logits, labels) * task_weight).sum() / logits.size(0)

                        if n_gpu > 1:
                            loss = loss.mean()
                        val_loss += loss.item() / len(val_loader)

                        probs = torch.sigmoid(logits)
                        preds = (torch.sum(probs > 0.5, dim=-1) + 1)
                        labels = (labels.sum(dim=-1) + 1)

                        epoch_preds.append(preds.cpu().numpy())
                        epoch_labels.append(labels.cpu().numpy())

                        pbar.update(1)

            epoch_preds = np.concatenate(epoch_preds)
            epoch_labels = np.concatenate(epoch_labels)

            score = eval_function(epoch_preds, epoch_labels)

            if score > best_score:
                best_score = score

                # fold{}.epoch{}.{}.ckpt
                ckpt_path = ckpt_template.format(fold + 1, epoch, str(val_loss)[2:6], str(score)[2:6])

                if not os.path.exists(os.path.dirname(ckpt_path)):
                    os.makedirs(os.path.dirname(ckpt_path))

                model_to_save = model.module if n_gpu > 1 else model
                torch.save(model_to_save.state_dict(), ckpt_path)

                if prev_best_ckpt and os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)

                prev_best_ckpt = ckpt_path

            print(f'train loss: {avg_loss:.4f} val loss: {val_loss:.4f}')
            print(f'eval result: {score:.4f}')

        del train_examples, valid_examples
        del train_dataset, valid_dataset
        del train_loader, val_loader
        del model
        del optimizer, scheduler, loss_fn
        torch.cuda.empty_cache()
        gc.collect()

        print(f'Fold {fold + 1} done')


def inference(model, loader, device):
    guid2probs = {}
    guid2label = {}

    model.eval()
    with tqdm(total=len(loader), desc='Inference') as pbar:
        with torch.no_grad():
            for step, data in enumerate(loader):
                # input_ids, token_type_ids, attention_mask, labels, guids
                input_ids, token_type_ids, attention_mask, labels = tuple(t.to(device) for t in data[:4])
                batch_guids = data[-1]

                logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
                logits = torch.sigmoid(logits)

                for guid, score, label in zip(batch_guids, logits.cpu(), labels.cpu()):
                    guid2probs[guid] = score
                    guid2label[guid] = label

                pbar.update(1)

    return guid2probs, guid2label


def merge_predictions(*preds):
    merged_guid2pred = {}
    assert len(set([len(guid2pred) for guid2pred in preds])) == 1
    for guid in preds[0]:
        avg = sum([guid2pred[guid] for guid2pred in preds]) / len(preds)
        merged_guid2pred[guid] = avg
    return merged_guid2pred


def train_inference(ckpt_dir, multidrop=True, model_name='bert', filter=False, chunk=False, biased_split=False,
                    swap=False, seed=42):
    seed_everything(seed)

    model_dir, config_path = get_bert_dir_and_config(model_name)

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    examples, text2chunkid, chunks = prepare_instances(config_local.requirements_path, config_local.train_achievements_path,
                                                       config_local.interrelation_path, tokenizer,
                                                       label_fn=lambda guid, label: extend_label(label),
                                                       filter=filter, chunk=chunk, swap=swap)

    num_ckpts = sum([int(name.endswith('.ckpt')) for name in os.listdir(ckpt_dir)])
    ckpts = [name for name in os.listdir(ckpt_dir) if name.endswith('.ckpt')]
    ckpts.sort(key=lambda name: int(name.split('.')[0][4:]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f'Let\'s use {n_gpu} GPUs!!!')

    guid2score = {}
    guid2label = {}

    model = get_coral_bert(model_dir, config_path, multidrop)

    kf_train_indices, kf_val_indices = chunk_k_fold(examples, chunks, num_ckpts, biased=biased_split)
    for fold in range(len(kf_train_indices)):
        val_index = kf_val_indices[fold]

        print(f'Evaluating FOLD {fold + 1}')

        valid_examples = [examples[i] for i in val_index]
        valid_dataset = NLIDataset(valid_examples)
        print(f'# of val   examples: {len(valid_examples)}')

        batch_size = 8
        if model_name == 'roberta_large':
            batch_size = 4

        val_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                sampler=BucketSampler(valid_dataset, valid_dataset.getkeys(),
                                                      batch_size=batch_size, shuffle_data=True),
                                collate_fn=collate_fn)

        ckpt_path = os.path.join(ckpt_dir, ckpts[fold])
        model = model_transformation(model, device, n_gpu, ckpt_path=ckpt_path)
        scores, labels = inference(model, val_loader, device)
        guid2score.update(scores)
        guid2label.update(labels)

        del valid_examples, valid_dataset, val_loader
        torch.cuda.empty_cache()
        gc.collect()

        print(f'Fold {fold + 1} done')

    return guid2score, guid2label


def test_inference_single_model(ckpt_dir, swap, multidrop=True, model_name='bert', pseudo_label_name=None):
    model_dir, config_path = get_bert_dir_and_config(model_name)

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    examples, text2chunkid, chunks = prepare_instances(config_local.requirements_path, config_local.test_achievements_path,
                                                       config_local.test_prediction_path, tokenizer,
                                                       label_fn=lambda guid, label: extend_label(label),
                                                       swap=swap)
    test_dataset = NLIDataset(examples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f'Let\'s use {n_gpu} GPUs!!!')

    batch_size = 8
    if model_name == 'roberta_large':
        batch_size = 4

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             sampler=BucketSampler(test_dataset, test_dataset.getkeys(),
                                                   batch_size=batch_size),
                             collate_fn=collate_fn)

    model = get_coral_bert(model_dir, config_path, multidrop)

    ckpt_preds = []

    for name in os.listdir(ckpt_dir):
        if not name.endswith('.ckpt'):
            continue
        model = model_transformation(model, device, n_gpu, ckpt_path=os.path.join(ckpt_dir, name))
        guid2probs, _ = inference(model, test_loader, device)
        ckpt_preds.append(guid2probs)
    guid2probs = merge_predictions(*ckpt_preds)

    if pseudo_label_name:
        pseudo_label_path = f'../pseudo_labels/{pseudo_label_name}.pkl'
        if not os.path.exists(os.path.dirname(pseudo_label_path)):
            os.makedirs(os.path.dirname(pseudo_label_path))
        with open(pseudo_label_path, 'wb') as f:
            pickle.dump(guid2probs, f)

    return guid2probs


def test_inference(ckpt_dirs, swaps, multidrop=True, model_name='bert', pseudo_label_name=None):
    guid2preds = []
    for ckpt_dir, swap in zip(ckpt_dirs, swaps):
        guid2preds.append(test_inference_single_model(ckpt_dir, swap, multidrop, model_name))
    guid2pred = merge_predictions(*guid2preds)
    if pseudo_label_name:
        pseudo_label_path = f'../pseudo_labels/{pseudo_label_name}.pkl'
        if not os.path.exists(os.path.dirname(pseudo_label_path)):
            os.makedirs(os.path.dirname(pseudo_label_path))
        with open(pseudo_label_path, 'wb') as f:
            pickle.dump(guid2pred, f)
    return guid2pred


def get_eval_result(guid2pred, guid2label, write_to=None):
    guids = list(guid2pred.keys())
    oof_preds = [torch.sum(guid2pred[guid] > 0.5) + 1 for guid in guids]
    oof_labels = [torch.sum(guid2label[guid] > 0.5) + 1 for guid in guids]

    score = eval_function(oof_preds, oof_labels)

    if write_to is not None:
        with open(os.path.join(write_to, 'eval_result.txt'), 'w') as f:
            f.write(f'{score}\n')

    print(f'{score}')


def make_submission(guid2probs, output_path):
    guids = list(guid2probs)
    # oof_preds = [torch.argmax(guid2probs[guid]) + 1 for guid in guids]
    # oof_labels = [torch.argmax(guid2probs[guid]) + 1 for guid in guids]

    guid2level = {}
    for guid, prob in guid2probs.items():
        # guid2level[guid] = list(guid2probs)
        pred = torch.sum(prob > 0.5) + 1
        guid2level[guid] = pred.item()

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    submit = pd.read_csv(config_local.submit_example_path)
    submit['Level'] = submit['Guid'].apply(lambda guid: guid2level[guid])
    submit.to_csv(output_path, index=False)


if __name__ == '__main__':
    # Stage 2
    # CV: 0.7924199380165289
    kfold_training(10, swap=False, filter=False, chunk=False, radam=False, biased_split=False, learning_rate=2e-5,
                   batch_size=32, warmup=0.1, wd=0.01, n_epochs=6, seed=config_local.seed1)
    guid2pred1, guid2label = train_inference(
        '../new_ckpt/msd.coral.bert.lr2e-05.bs32.nfold10.nepoch6.warmup0.1.wd0.01.seed20170712',
        filter=False, chunk=False, biased_split=False, swap=False, seed=config_local.seed1)
    get_eval_result(guid2pred1, guid2label,
                    write_to='../new_ckpt/msd.coral.bert.lr2e-05.bs32.nfold10.nepoch6.warmup0.1.wd0.01.seed20170712')

    # CV: 0.7906841901816777
    kfold_training(10, swap=True, filter=False, chunk=False, radam=False, biased_split=False, learning_rate=2e-5,
                   batch_size=32, warmup=0.1, wd=0.01, n_epochs=6, seed=config_local.seed1)
    guid2pred2, guid2label = train_inference(
        '../new_ckpt/swap.msd.coral.bert.lr2e-05.bs32.nfold10.nepoch6.warmup0.1.wd0.01.seed20170712',
        filter=False, chunk=False, biased_split=False, swap=True, seed=config_local.seed1)
    get_eval_result(guid2pred2, guid2label,
                    write_to='../new_ckpt/swap.msd.coral.bert.lr2e-05.bs32.nfold10.nepoch6.warmup0.1.wd0.01.seed20170712')

    # CV: 0.7963792096554408 LB: 0.80115831
    guid2pred = merge_predictions(guid2pred1, guid2pred2)
    get_eval_result(guid2pred, guid2label)

    guid2pred = test_inference(['../new_ckpt/msd.coral.bert.lr2e-05.bs32.nfold10.nepoch6.warmup0.1.wd0.01.seed20170712',
                                '../new_ckpt/swap.msd.coral.bert.lr2e-05.bs32.nfold10.nepoch6.warmup0.1.wd0.01.seed20170712'],
                               [False, True],
                               pseudo_label_name='2msd.coral.bert.lr2e-05.bs32.nfold10.nepoch6.warmup0.1.wd0.01.seed20170712')

    guid2pred1 = test_inference_single_model(
        '../new_ckpt/msd.coral.bert.lr2e-05.bs32.nfold10.nepoch6.warmup0.1.wd0.01.seed20170712',
        swap=False)
    guid2pred2 = test_inference_single_model(
        '../new_ckpt/swap.msd.coral.bert.lr2e-05.bs32.nfold10.nepoch6.warmup0.1.wd0.01.seed20170712',
        swap=True
    )
    guid2pred = merge_predictions(guid2pred1, guid2pred2)
    make_submission(guid2pred,
                    '../new_output/2swap.msd.coral.bert.lr2e-05.bs32.nfold10.nepoch6.warmup0.1.wd0.01.seed20170712.csv')
