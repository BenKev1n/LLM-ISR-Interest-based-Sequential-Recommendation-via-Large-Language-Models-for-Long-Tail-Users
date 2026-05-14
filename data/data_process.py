from collections import defaultdict
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
import tqdm
import os
from tqdm import tqdm

true=True
false=False
def parse(path): # for Amazon
    g = gzip.open(path, 'rb')
    inter_list = []
    for l in tqdm(g):
        inter_list.append(json.loads(l.decode()))
        # inter_list.append(eval(l))

    return inter_list


def parse_meta(path): # for Amazon
    g = gzip.open(path, 'rb')
    inter_list = []
    for l in tqdm(g):
        inter_list.append(eval(l))

    return inter_list


def parse_meta_data(path):  # for Amazon data-type gz file (actually plain JSON text)
    inter_list = []
    with open(path, 'r', encoding='utf-8') as f:  # 使用文本模式和utf-8编码
        for l in tqdm(f):
            line = l.strip()
            if not line:  # 跳过空行
                continue
            # 处理可能的行尾逗号（Amazon数据集常见问题）
            if line.endswith(','):
                line = line[:-1]
            try:
                data = json.loads(line)  # 安全且标准的JSON解析
                inter_list.append(data)
            except json.JSONDecodeError as e:
                # 可选：打印错误行，便于调试
                # print(f"JSON解析错误，跳过该行。错误信息: {e}， 行内容: {line[:100]}...")
                # 根据你的需求，可以选择跳过或抛出异常
                continue
    return inter_list


# return (user item timestamp) sort in get_interaction
def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    # older Amazon
    data_flie = './data/' + str(dataset_name) + '/raw/' + str(dataset_name) + '.json.gz'
    # latest Amazon
    # data_flie = '/home/hui_wang/data/new_Amazon/' + dataset_name + '.json.gz'
    for inter in parse(data_flie):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas


def New_Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    # older Amazon
    data_flie = './data/' + str(dataset_name) + '/raw/' + str(dataset_name) + '.json.gz'
    # latest Amazon
    # data_flie = '/home/hui_wang/data/new_Amazon/' + dataset_name + '.json.gz'
    for inter in parse(data_flie):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas


def Amazon_meta(dataset_name, data_maps):
    datas = {}
    base = './data/' + str(dataset_name) + '/raw/'
    gz_path = base + 'meta_' + str(dataset_name) + '.json.gz'
    plain_path = base + 'meta_' + str(dataset_name) + '.json'
    item_asins = list(data_maps['item2id'].keys())
    try:
        for info in tqdm(parse_meta(gz_path)):
            asin = info.get('asin')
            if asin not in item_asins:
                continue
            datas[asin] = info
        return datas
    except Exception:
        pass
    try:
        for info in tqdm(parse_meta_data(plain_path)):
            asin = info.get('asin')
            if asin not in item_asins:
                continue
            datas[asin] = info
        return datas
    except Exception:
        pass
    try:
        for info in tqdm(parse_meta_data(gz_path)):
            asin = info.get('asin')
            if asin not in item_asins:
                continue
            datas[asin] = info
        return datas
    except Exception:
        pass
    return datas

def Yelp(date_min, date_max, rating_score): # take out inters in [date_min, date_max] and the score < rating_score
    datas = []
    data_flie = './data/yelp/raw/yelp_academic_dataset_review.json'
    lines = open(data_flie).readlines()
    for line in tqdm(lines):
        review = json.loads(line.strip())
        user = review['user_id']
        item = review['business_id']
        rating = review['stars']
        # 2004-10-12 10:13:32 2019-12-13 15:51:19
        date = review['date']
        # 剔除一些例子
        if date < date_min or date > date_max or float(rating) <= rating_score:
            continue
        time = date.replace('-','').replace(':','').replace(' ','') 
        datas.append((user, item, int(time)))
    return datas


def Yelp_meta(datamaps):
    meta_infos = {}
    meta_file = './data/yelp/raw/yelp_academic_dataset_business.json'
    item_ids = list(datamaps['item2id'].keys())
    # Robust parsing: support JSON Lines with potential trailing commas or JSON array
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':  # JSON array
                data = json.load(f)
                for info in tqdm(data):
                    bid = info.get('business_id')
                    if bid in item_ids:
                        meta_infos[bid] = info
            else:  # JSON Lines
                skipped = 0
                for line in tqdm(f):
                    l = line.strip()
                    if not l or l in {'[', ']', ','}:
                        continue
                    if l.endswith(','):
                        l = l[:-1]
                    try:
                        info = json.loads(l)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    bid = info.get('business_id')
                    if bid in item_ids:
                        meta_infos[bid] = info
                if skipped:
                    print(f"[Yelp_meta] skipped {skipped} malformed JSON line(s)")
    except json.JSONDecodeError:
        # If overall format is inconsistent, silently return what has been parsed
        pass
    return meta_infos


def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

# categories 和 brand is all attribute
def get_attribute_Amazon(meta_infos, datamaps, attribute_core):
    attributes = defaultdict(int)
    for iid, info in tqdm(meta_infos.items()):
        b = info.get('brand')
        if isinstance(b, str) and len(b) > 0:
            attributes[b] += 1
        cats = info.get('categories')
        if isinstance(cats, list):
            for cates in cats:
                if isinstance(cates, list):
                    s = 1 if len(cates) > 0 else 0
                    for cate in cates[s:]:
                        if isinstance(cate, str) and len(cate) > 0:
                            attributes[cate] += 1
    new_meta = {}
    for iid, info in tqdm(meta_infos.items()):
        arr = []
        b = info.get('brand')
        if isinstance(b, str) and len(b) > 0 and attributes[b] >= attribute_core:
            arr.append(b)
        cats = info.get('categories')
        if isinstance(cats, list):
            for cates in cats:
                if isinstance(cates, list):
                    s = 1 if len(cates) > 0 else 0
                    for cate in cates[s:]:
                        if isinstance(cate, str) and len(cate) > 0 and attributes[cate] >= attribute_core:
                            arr.append(cate)
        new_meta[iid] = arr
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    for iid, arr in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in arr:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    if len(attribute_lens) > 0:
        print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    else:
        print(f'attributes len, Min:0, Max:0, Avg.:0.0000')
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), (np.mean(attribute_lens) if len(attribute_lens) > 0 else 0.0), datamaps, items2attributes


def get_attribute_Yelp(meta_infos, datamaps, attribute_core):
    attributes = defaultdict(int)
    for iid, info in tqdm(meta_infos.items()):
        try:
            cates = [cate.strip() for cate in info['categories'].split(',')]
            for cate in cates:
                attributes[cate] +=1
        except:
            pass
    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm(meta_infos.items()):
        new_meta[iid] = []
        try:
            cates = [cate.strip() for cate in info['categories'].split(',') ]
            for cate in cates:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
        except:
            pass
    # 做映射
    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'after delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # 更新datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes

def get_interaction(datas): # sort the interactions based on timestamp
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # 已经保证Kcore

# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core): # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def filter_common(user_items, user_t, item_t):

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, item, _ in user_items:
        user_count[user] += 1
        item_count[item] += 1

    User = {}
    for user, item, timestamp in user_items:
        if user_count[user] < user_t or item_count[item] < item_t:
            continue
        if user not in User.keys():
            User[user] = []
        User[user].append((item, timestamp))

    new_User = {}
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[1])
        new_hist = [i for i, t in User[userid]]
        new_User[userid] = new_hist

    return new_User



def id_map(user_items): # user_items dict

    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps


def get_counts(user_items):

    user_count = {}
    item_count = {}

    for user, items in user_items.items():
        user_count[user] = len(items)
        for item in items:
            if item not in item_count.keys():
                item_count[item] = 1
            else:
                item_count[item] += 1

    return user_count, item_count


def filter_minmum(user_items, min_len=3):

    new_user_items = {}
    for user, items in user_items.items():
        if len(items) >= min_len:
            new_user_items[user] = items

    return new_user_items



def main(data_name, data_type='Amazon', user_core=3, item_core=3):
    assert data_type in {'Amazon', 'Yelp', 'New_Amazon'}
    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    attribute_core = 0

    if data_type == 'Yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2000-01-01 00:00:00'
        datas = Yelp(date_min, date_max, rating_score)
    elif data_type == "New_Amazon":
        datas = New_Amazon(data_name, rating_score=rating_score)
    else:
        datas = Amazon(data_name, rating_score=rating_score)

    # datas = datas[:int(len(datas)*0.1)] # for electronics and game
    if data_type != "New_Amazon":
        user_items = get_interaction(datas)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    user_items = filter_common(datas, user_t=user_core, item_t=item_core)
    # user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items, user_num, item_num, data_maps = id_map(user_items)  # new_num_id
    user_items = filter_minmum(user_items, min_len=3)
    # user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count, item_count = get_counts(user_items)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)


    print('Begin extracting meta infos...')

    if data_type == 'Amazon':
        meta_infos = Amazon_meta(data_name, data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(meta_infos, data_maps, attribute_core)
    elif data_type == "New_Amazon":
        meta_infos = Amazon_meta(data_name, data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(meta_infos, data_maps, attribute_core)
    else:
        meta_infos = Yelp_meta(data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Yelp(meta_infos, data_maps, attribute_core)

    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')

    # -------------- Save Data ---------------
    handled_path = 'data/' + data_name + '/handled/'
    if not os.path.exists(handled_path):
        os.makedirs(handled_path)

    data_file = handled_path + 'inter_seq.txt'
    item2attributes_file = handled_path + 'item2attributes.json'
    id_file = handled_path + "id_map.json"

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    json_str = json.dumps(meta_infos)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)
    with open(id_file, "w") as f:
        json.dump(data_maps, f)



def LastFM():
    user_core = 5
    item_core = 5
    datas = []
    data_file = '/path/lastfm/2k/user_attributegedartists-timestamps.dat'
    lines = open(data_file).readlines()
    for line in tqdm.tqdm(lines[1:]):
        user, item, attribute, timestamp = line.strip().split('\t')
        datas.append((user, item, int(timestamp)))

    # 有重复item
    user_seq = {}
    user_seq_notime = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            if item not in user_seq_notime[user]:
                user_seq[user].append((item, time))
                user_seq_notime[user].append(item)
            else:
                continue
        else:
            user_seq[user] = []
            user_seq_notime[user] = []

            user_seq[user].append((item, time))
            user_seq_notime[user].append(item)

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items

    user_items = filter_Kcore(user_seq, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items, user_num, item_num, data_maps = id_map(user_items)  # new_num_id
    user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    attribute_file = './data_path/artist2attributes.json'

    meta_item2attribute = json.loads(open(attribute_file).readline())

    # 做映射
    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    item2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in meta_item2attribute.items():
        if iid in list(data_maps['item2id'].keys()):
            item_id = data_maps['item2id'][iid]
            item2attributes[item_id] = []
            for attribute in attributes:
                if attribute not in attribute2id:
                    attribute2id[attribute] = attribute_id
                    id2attribute[attribute_id] = attribute
                    attribute_id += 1
                item2attributes[item_id].append(attribute2id[attribute])
            attribute_lens.append(len(item2attributes[item_id]))
    print(f'after delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # 更新datamap
    data_maps['attribute2id'] = attribute2id
    data_maps['id2attribute'] = id2attribute

    data_name = 'LastFM'
    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(len(attribute2id))}&'
          f'{np.mean(attribute_lens):.1f} \\')

    # -------------- Save Data ---------------
    # one user one line
    data_file = 'data/' + data_name + '.txt'
    item2attributes_file = 'data/' + data_name + '_item2attributes.json'

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')

    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)

amazon_datas = ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games']


if __name__ == "__main__":

    # Run Yelp only; comment Amazon runs until raw files exist
    main('musical_instruments', data_type='Amazon', user_core=3, item_core=3)
    # main("fashion", data_type="Amazon")
    # main("beauty", data_type="Amazon", user_core=3, item_core=3)
