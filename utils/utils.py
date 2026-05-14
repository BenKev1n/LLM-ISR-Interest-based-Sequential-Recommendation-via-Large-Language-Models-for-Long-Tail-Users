# here put the import lib
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def set_seed(seed):
    '''Fix all of random seed for reproducible training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # only add when conv in your model


def get_n_params(model):
    '''Get the number of parameters of model'''
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def get_n_params_(parameter_list):
    '''Get the number of parameters of model'''
    pp = 0
    for p in list(parameter_list):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def unzip_data(data, aug=True, aug_num=0):

    res = []
    
    if aug:
        for user in tqdm(data):

            user_seq = data[user]
            seq_len = len(user_seq)

            for i in range(aug_num+2, seq_len+1):
                
                res.append(user_seq[:i])
    else:
        for user in tqdm(data):

            user_seq = data[user]
            res.append(user_seq)

    return res


def unzip_data_with_user(data, aug=True, aug_num=0):
    """Expand user sequence and keep the real global user_id mapping.
    - data: dict[user_id -> seq]
    - return: (expanded sequence list, corresponding global user_id list)
    """

    res = []
    users = []

    if aug:
        # Traverse the real user_id and its sequence in data
        for user_id, user_seq in tqdm(data.items()):
            seq_len = len(user_seq)
            for i in range(aug_num + 2, seq_len + 1):
                res.append(user_seq[:i])
                users.append(user_id)
    else:
        for user_id, user_seq in tqdm(data.items()):
            res.append(user_seq)
            users.append(user_id)

    return res, users


def concat_data(data_list):

    res = []

    if len(data_list) == 2:

        train = data_list[0]
        valid = data_list[1]

        for user in train:

            res.append(train[user]+valid[user])
    
    elif len(data_list) == 3:

        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]

        for user in train:

            res.append(train[user]+valid[user]+test[user])

    else:

        raise ValueError

    return res


def concat_aug_data(data_list):

    res = []

    train = data_list[0]
    valid = data_list[1]

    for user in train:

        if len(valid[user]) == 0:
            res.append([train[user][0]])
        
        else:
            res.append(train[user]+valid[user])

    return res


def concat_data_with_user(data_list):
    """Concatenate data by user and keep the real global user_id mapping.
    - data_list: [train, valid] or [train, valid, test] each is dict[user_id -> seq]
    - return: (concatenated sequence list, corresponding global user_id list)
    """

    res = []
    users = []

    if len(data_list) == 2:
        train, valid = data_list
        for user_id in train:
            res.append(train[user_id] + valid[user_id])
            users.append(user_id)

    elif len(data_list) == 3:
        train, valid, test = data_list
        for user_id in train:
            res.append(train[user_id] + valid[user_id] + test[user_id])
            users.append(user_id)

    else:
        raise ValueError

    return res, users


def filter_data(data, thershold=5):
    '''Filter out the sequence shorter than threshold'''
    res = []

    for user in data:

        if len(user) > thershold:
            res.append(user)
        else:
            continue
    
    return res



def random_neq(l, r, s=[]):    # Sample a random number in range l-r except the numbers in s
    
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t



def metric_report(data_rank, topk=10):

    ndcg_1, ndcg_5, ndcg_10, ndcg_20 = 0.0, 0.0, 0.0, 0.0
    HT = 0
    for rank in data_rank:
        if rank < 1:
            ndcg_1 += 1 / np.log2(rank + 2)
        if rank < 5:
            ndcg_5 += 1 / np.log2(rank + 2)
        if rank < 10:
            ndcg_10 += 1 / np.log2(rank + 2)
        if rank < 20:
            ndcg_20 += 1 / np.log2(rank + 2)
        if rank < topk:
            HT += 1

    n = len(data_rank)
    return {
        'NDCG@1': ndcg_1 / n,
        'NDCG@5': ndcg_5 / n,
        'NDCG@10': ndcg_10 / n,
        'NDCG@20': ndcg_20 / n,
        'HR@10': HT / n,
    }



# def metric_len_report(data_rank, data_len, topk=10, aug_len=0, args=None):

#     if args is not None:
#         ts_short = args.ts_short
#         ts_long = args.ts_long
#     else:
#         ts_short = 10
#         ts_long = 20

#     NDCG_s, HT_s = 0, 0
#     NDCG_m, HT_m = 0, 0
#     NDCG_l, HT_l = 0, 0
#     count_s = len(data_len[data_len<ts_short+aug_len])
#     count_l = len(data_len[data_len>=ts_long+aug_len])
#     count_m = len(data_len) - count_s - count_l

#     for i, rank in enumerate(data_rank):

#         if rank < topk:

#             if data_len[i] < ts_short+aug_len:
#                 NDCG_s += 1 / np.log2(rank + 2)
#                 HT_s += 1
#             elif data_len[i] < ts_long+aug_len:
#                 NDCG_m += 1 / np.log2(rank + 2)
#                 HT_m += 1
#             else:
#                 NDCG_l += 1 / np.log2(rank + 2)
#                 HT_l += 1

#     return {'Short NDCG@10': NDCG_s / count_s if count_s!=0 else 0, # avoid division of 0
#             'Short HR@10': HT_s / count_s if count_s!=0 else 0,
#             'Medium NDCG@10': NDCG_m / count_m if count_m!=0 else 0,
#             'Medium HR@10': HT_m / count_m if count_m!=0 else 0,
#             'Long NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
#             'Long HR@10': HT_l / count_l if count_l!=0 else 0,}


def metric_len_report(data_rank, data_len, topk=10, aug_len=0, args=None):

    if args is not None:
        ts_user = args.ts_user
    else:
        ts_user = 10

    ndcg1_s, ndcg5_s, ndcg10_s, ndcg20_s, hr10_s = 0.0, 0.0, 0.0, 0.0, 0
    ndcg1_l, ndcg5_l, ndcg10_l, ndcg20_l, hr10_l = 0.0, 0.0, 0.0, 0.0, 0
    count_s = len(data_len[data_len<ts_user+aug_len])
    count_l = len(data_len[data_len>=ts_user+aug_len])

    for i, rank in enumerate(data_rank):
        if data_len[i] < ts_user+aug_len:
            if rank < 1:
                ndcg1_s += 1 / np.log2(rank + 2)
            if rank < 5:
                ndcg5_s += 1 / np.log2(rank + 2)
            if rank < 10:
                ndcg10_s += 1 / np.log2(rank + 2)
            if rank < 20:
                ndcg20_s += 1 / np.log2(rank + 2)
            if rank < 10:
                hr10_s += 1
        else:
            if rank < 1:
                ndcg1_l += 1 / np.log2(rank + 2)
            if rank < 5:
                ndcg5_l += 1 / np.log2(rank + 2)
            if rank < 10:
                ndcg10_l += 1 / np.log2(rank + 2)
            if rank < 20:
                ndcg20_l += 1 / np.log2(rank + 2)
            if rank < 10:
                hr10_l += 1

    return {
        'Short NDCG@1': ndcg1_s / count_s if count_s!=0 else 0,
        'Short NDCG@5': ndcg5_s / count_s if count_s!=0 else 0,
        'Short NDCG@10': ndcg10_s / count_s if count_s!=0 else 0,
        'Short NDCG@20': ndcg20_s / count_s if count_s!=0 else 0,
        'Short HR@10': hr10_s / count_s if count_s!=0 else 0,
        'Long NDCG@1': ndcg1_l / count_l if count_l!=0 else 0,
        'Long NDCG@5': ndcg5_l / count_l if count_l!=0 else 0,
        'Long NDCG@10': ndcg10_l / count_l if count_l!=0 else 0,
        'Long NDCG@20': ndcg20_l / count_l if count_l!=0 else 0,
        'Long HR@10': hr10_l / count_l if count_l!=0 else 0,
    }


def metric_pop_report(data_rank, pop_dict, target_items, topk=10, aug_pop=0, args=None):
    """
    Report the metrics according to target item's popularity
    item_pop: the array of the target item's popularity
    """
    if args is not None:
        ts_tail = args.ts_item
    else:
        ts_tail = 20

    ndcg1_s, ndcg5_s, ndcg10_s, ndcg20_s, hr10_s = 0.0, 0.0, 0.0, 0.0, 0
    ndcg1_l, ndcg5_l, ndcg10_l, ndcg20_l, hr10_l = 0.0, 0.0, 0.0, 0.0, 0
    item_pop = pop_dict[target_items.astype("int64")]
    count_s = len(item_pop[item_pop<ts_tail+aug_pop])
    count_l = len(item_pop[item_pop>=ts_tail+aug_pop])

    for i, rank in enumerate(data_rank):
        if i == 0:
            continue
        if item_pop[i] < ts_tail+aug_pop:
            if rank < 1:
                ndcg1_s += 1 / np.log2(rank + 2)
            if rank < 5:
                ndcg5_s += 1 / np.log2(rank + 2)
            if rank < 10:
                ndcg10_s += 1 / np.log2(rank + 2)
            if rank < 20:
                ndcg20_s += 1 / np.log2(rank + 2)
            if rank < 10:
                hr10_s += 1
        else:
            if rank < 1:
                ndcg1_l += 1 / np.log2(rank + 2)
            if rank < 5:
                ndcg5_l += 1 / np.log2(rank + 2)
            if rank < 10:
                ndcg10_l += 1 / np.log2(rank + 2)
            if rank < 20:
                ndcg20_l += 1 / np.log2(rank + 2)
            if rank < 10:
                hr10_l += 1

    return {
        'Tail NDCG@1': ndcg1_s / count_s if count_s!=0 else 0,
        'Tail NDCG@5': ndcg5_s / count_s if count_s!=0 else 0,
        'Tail NDCG@10': ndcg10_s / count_s if count_s!=0 else 0,
        'Tail NDCG@20': ndcg20_s / count_s if count_s!=0 else 0,
        'Tail HR@10': hr10_s / count_s if count_s!=0 else 0,
        'Popular NDCG@1': ndcg1_l / count_l if count_l!=0 else 0,
        'Popular NDCG@5': ndcg5_l / count_l if count_l!=0 else 0,
        'Popular NDCG@10': ndcg10_l / count_l if count_l!=0 else 0,
        'Popular NDCG@20': ndcg20_l / count_l if count_l!=0 else 0,
        'Popular HR@10': hr10_l / count_l if count_l!=0 else 0,
    }



def metric_len_5group(pred_rank, 
                      seq_len, 
                      thresholds=[5, 10, 15, 20], 
                      topk=10):

    NDCG = np.zeros(5)
    HR = np.zeros(5)    
    for i, rank in enumerate(pred_rank):

        target_len = seq_len[i]
        if rank < topk:

            if target_len < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1

            elif target_len < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1

            elif target_len < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1

            elif target_len < thresholds[3]:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1

            else:
                NDCG[4] += 1 / np.log2(rank + 2)
                HR[4] += 1

    count = np.zeros(5)
    count[0] = len(seq_len[seq_len>=0]) - len(seq_len[seq_len>=thresholds[0]])
    count[1] = len(seq_len[seq_len>=thresholds[0]]) - len(seq_len[seq_len>=thresholds[1]])
    count[2] = len(seq_len[seq_len>=thresholds[1]]) - len(seq_len[seq_len>=thresholds[2]])
    count[3] = len(seq_len[seq_len>=thresholds[2]]) - len(seq_len[seq_len>=thresholds[3]])
    count[4] = len(seq_len[seq_len>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count



def metric_pop_5group(pred_rank, 
                      pop_dict, 
                      target_items, 
                      thresholds=[10, 30, 60, 100], 
                      topk=10):

    NDCG = np.zeros(5)
    HR = np.zeros(5)    
    for i, rank in enumerate(pred_rank):

        target_pop = pop_dict[int(target_items[i])]
        if rank < topk:

            if target_pop < thresholds[0]:
                NDCG[0] += 1 / np.log2(rank + 2)
                HR[0] += 1

            elif target_pop < thresholds[1]:
                NDCG[1] += 1 / np.log2(rank + 2)
                HR[1] += 1

            elif target_pop < thresholds[2]:
                NDCG[2] += 1 / np.log2(rank + 2)
                HR[2] += 1

            elif target_pop < thresholds[3]:
                NDCG[3] += 1 / np.log2(rank + 2)
                HR[3] += 1

            else:
                NDCG[4] += 1 / np.log2(rank + 2)
                HR[4] += 1

    count = np.zeros(5)
    pop = pop_dict[target_items.astype("int64")]
    count[0] = len(pop[pop>=0]) - len(pop[pop>=thresholds[0]])
    count[1] = len(pop[pop>=thresholds[0]]) - len(pop[pop>=thresholds[1]])
    count[2] = len(pop[pop>=thresholds[1]]) - len(pop[pop>=thresholds[2]])
    count[3] = len(pop[pop>=thresholds[2]]) - len(pop[pop>=thresholds[3]])
    count[4] = len(pop[pop>=thresholds[3]])

    for j in range(5):
        NDCG[j] = NDCG[j] / count[j]
        HR[j] = HR[j] / count[j]

    return HR, NDCG, count



def seq_acc(true, pred):

    true_num = np.sum((true==pred))
    total_num = true.shape[0] * true.shape[1]

    return {'acc': true_num / total_num}


def load_pretrained_model(pretrain_dir, model, logger, device):

    logger.info("Loading pretrained model ... ")
    checkpoint_path = os.path.join(pretrain_dir, 'pytorch_model.bin')

    model_dict = model.state_dict()

    # To be compatible with the new and old version of model saver
    try:
        pretrained_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    except:
        pretrained_dict = torch.load(checkpoint_path, map_location=device)

    # filter out required parameters
    new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    # Print the number of parameters loaded
    logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
    model.load_state_dict(model_dict)

    return model


def record_csv(args, res_dict, path='log'):
    path = os.path.join(path, args.dataset)
    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    model_name = args.aug_file + '-' + args.now_str
    res_dict["model_name"] = model_name
    desired_cols = [
        "model_name",
        "NDCG@10",
        "HR@10",
        "Short NDCG@10",
        "Short HR@10",
        "Long NDCG@10",
        "Long HR@10",
        "Tail NDCG@10",
        "Tail HR@10",
        "Popular NDCG@10",
        "Popular HR@10",
        "NDCG@1",
        "NDCG@5",
        "NDCG@20",
    ]
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    allowed_cols = [col for col in desired_cols if col in new_res_dict]

    add_df = pd.DataFrame(new_res_dict)
    add_df = add_df.reindex(columns=allowed_cols)

    if not os.path.exists(csv_path):
        add_df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
        merged_cols = list(df.columns)
        for col in allowed_cols:
            if col not in merged_cols:
                merged_cols.append(col)
        # Fill in the missing columns in existing pandas df
        for col in merged_cols:
            if col not in df.columns:
                df[col] = np.nan
        # Add the new rows to the existing pandas df
        add_df = add_df.reindex(columns=merged_cols)
        df = pd.concat([df.reindex(columns=merged_cols), add_df], ignore_index=True)
        df.to_csv(csv_path, index=False)



def record_group(args, res_dict, path='log'):
    
    path = os.path.join(path, args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    model_name = args.aug_file + '-' + args.now_str
    columns = list(res_dict.keys())
    columns.insert(0, "model_name")
    res_dict["model_name"] = model_name
    # columns = ["model_name", "HR@10", "NDCG@10", "Short HR@10", "Short NDCG@10", "Medium HR@10", "Medium NDCG@10", "Long HR@10", "Long NDCG@10",]
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    
    if not os.path.exists(csv_path):

        df = pd.DataFrame(new_res_dict)
        df = df[columns]    # reindex the columns
        df.to_csv(csv_path, index=False)

    else:

        df = pd.read_csv(csv_path)
        add_df = pd.DataFrame(new_res_dict)
        df = pd.concat([df, add_df])
        df.to_csv(csv_path, index=False)
