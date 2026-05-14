# here put the import lib
import os
import copy
import random
import numpy as np
from torch.utils.data import Dataset
from utils.utils import random_neq
import pickle


class SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation'''

    def __init__(self, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.var_name = ["seq", "pos", "neg", "positions"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions
    


class SeqDatasetAllUser(SeqDataset):
    '''The train dataset for Sequential recommendation'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__(data, item_num, max_len, neg_num)
        self.sim_user_num = args.sim_user_num
        self.sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", "sim_user_100.pkl"), "rb"))
        # Whether to include similar users in output (train: True, eval: False)
        self.include_sim = True
        # Optional mapping: global user_id -> full sequence for neighbor retrieval
        self.user2seq = None
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        # use the real global user_id
        user_id = getattr(self, 'users', None)
        if user_id is not None:
            user_id = self.users[index]
        else:
            user_id = index  # fallback

        if self.include_sim:
            # sim_users saved as 0-based index; convert global user_id(1-based) to 0-based access
            src_idx = int(user_id) - 1 if int(user_id) >= 1 else int(user_id)
            if src_idx < 0:
                src_idx = 0
            if isinstance(self.sim_users, np.ndarray):
                max_src = self.sim_users.shape[0] - 1
            else:
                max_src = len(self.sim_users) - 1
            if src_idx > max_src:
                src_idx = max_src
            sim_users = self.sim_users[src_idx][:self.sim_user_num]
            # Convert similar user ID to 1-based to match the user2seq key
            if isinstance(sim_users, np.ndarray):
                sim_users = (sim_users + 1).tolist()
            else:
                sim_users = [int(u) + 1 for u in sim_users]
            sim_seq, sim_positions = [], []
            for sim_user in sim_users:
                meta_seq, meta_positions = self._get_user_seq(sim_user)
                sim_seq.append(meta_seq)
                sim_positions.append(meta_positions)
            sim_seq = np.array(sim_seq)
            sim_positions = np.array(sim_positions)
            return seq, pos, neg, positions, user_id, sim_seq, sim_positions
        else:
            return seq, pos, neg, positions, user_id
    

    def _get_user_seq(self, user):

        # Prefer full sequence from mapping if available
        if self.user2seq is not None and user in self.user2seq:
            inter = self.user2seq[user]
        else:
            inter = self.data[user]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions



class Seq2SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation with seq-to-seq loss'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.aug_seq = args.aug_seq
        self.aug_seq_len = args.aug_seq_len
        self.var_name = ["seq", "pos", "neg", "positions"]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if self.aug_seq:
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions



class Seq2SeqDatasetAllUser(Seq2SeqDataset):

    def __init__(self, args, data, item_num, max_len, neg_num=1):

        super().__init__(args, data, item_num, max_len, neg_num)
        self.sim_user_num = args.sim_user_num
        # Required: basic similar user collection
        self.sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled", "sim_user_100.pkl"), "rb"))
        # Whether to include similar users in output (train: True, eval: False)
        self.include_sim = True
        # Optional mapping: global user_id -> full sequence
        self.user2seq = None
        # Optional: head/tail user augmentation (if files exist)
        self._use_head_tail = False
        if self._use_head_tail:
            try:
                self.head_sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled", "head_sim_user_100.pkl"), "rb"))  # 头部 sim dict
                self.tail_sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled", "tail_sim_user_100.pkl"), "rb"))  # 尾部 sim dict
                with open(os.path.join("./data/"+args.dataset+"/handled", "head_user_id.txt"), 'r') as f:
                    self.head_user_ids = [line.strip() for line in f.readlines()]
                print("head_user num:", len(self.head_user_ids))
                with open(os.path.join("./data/"+args.dataset+"/handled/", "tail_user_id.txt"), 'r') as f:
                    self.tail_user_ids = [line.strip() for line in f.readlines()]
                print("tail_user num:", len(self.tail_user_ids))
            except Exception as e:
                self._use_head_tail = False
                self.head_sim_users, self.tail_sim_users = None, None
                self.head_user_ids, self.tail_user_ids = [], []
                print(f"[Data][Warn] Head/Tail sim_user files missing: {e}. Fallback to regular sim_users only.")

        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions"]


    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if self.aug_seq:
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        # use the real global user_id
        user_id = getattr(self, 'users', None)
        if user_id is not None:
            user_id = self.users[index]
        else:
            user_id = index

        # Convert user_id to global 1-based user_id list if head/tail augmentation is used
        if self._use_head_tail:
            sim_users = None
            if str(user_id) in self.head_user_ids:
                group_idx = self.head_user_ids.index(str(user_id))
                raw_idx = self.head_sim_users[group_idx][:self.sim_user_num]
                # Convert 0-based index to global user_id
                sim_users = []
                for gi in raw_idx:
                    if 0 <= int(gi) < len(self.head_user_ids):
                        sim_users.append(int(self.head_user_ids[int(gi)]))
            elif str(user_id) in self.tail_user_ids:
                group_idx = self.tail_user_ids.index(str(user_id))
                raw_idx = self.tail_sim_users[group_idx][:self.sim_user_num]
                sim_users = []
                for gi in raw_idx:
                    if 0 <= int(gi) < len(self.tail_user_ids):
                        sim_users.append(int(self.tail_user_ids[int(gi)]))
            # Fallback to regular similar_users if user_id is not in head/tail list or files are missing
            if sim_users is None or len(sim_users) == 0:
                src_idx = int(user_id) - 1 if int(user_id) >= 1 else int(user_id)
                src_idx = max(0, src_idx)
                max_src = (self.sim_users.shape[0] - 1) if isinstance(self.sim_users, np.ndarray) else (len(self.sim_users) - 1)
                src_idx = min(src_idx, max_src)
                raw_idx = self.sim_users[src_idx][:self.sim_user_num]
                sim_users = (raw_idx + 1).tolist() if isinstance(raw_idx, np.ndarray) else [int(u) + 1 for u in raw_idx]
        else:
            # Convert global 0-based user_id to global 1-based
            src_idx = int(user_id) - 1 if int(user_id) >= 1 else int(user_id)
            src_idx = max(0, src_idx)
            max_src = (self.sim_users.shape[0] - 1) if isinstance(self.sim_users, np.ndarray) else (len(self.sim_users) - 1)
            src_idx = min(src_idx, max_src)
            raw_idx = self.sim_users[src_idx][:self.sim_user_num]
            sim_users = (raw_idx + 1).tolist() if isinstance(raw_idx, np.ndarray) else [int(u) + 1 for u in raw_idx]
        if self.include_sim:
            sim_seq, sim_positions = [], []
            for sim_user in sim_users:
                meta_seq, meta_positions = self._get_user_seq(sim_user)
                sim_seq.append(meta_seq)
                sim_positions.append(meta_positions)
            sim_seq = np.array(sim_seq)
            sim_positions = np.array(sim_positions)
            return seq, pos, neg, positions, user_id, sim_seq, sim_positions
        else:
            return seq, pos, neg, positions, user_id
    

    def _get_user_seq(self, user):

        if self.user2seq is not None and user in self.user2seq:
            inter = self.user2seq[user]
        else:
            inter = self.data[user]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions
    


class BertRecTrainDatasetAllUser(Dataset):
    '''The train dataset for Bert4Rec'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.mask_prob = args.mask_prob
        self.sim_user_num = args.sim_user_num
        self.mask_token = item_num + 1
        self.sim_users = pickle.load(open(os.path.join("./data/"+args.dataset+"/handled/", "sim_user_100.pkl"), "rb"))
        self.var_name = ["seq", "pos", "neg", "positions", "user_id", "sim_seq", "sim_positions"]


    def __len__(self):

        return 2 * len(self.data)

    def __getitem__(self, index):

        tokens = []
        labels, neg_labels = [], []

        if index >= len(self.data):
            seq = self.data[index - len(self.data)]
            for s in seq:
                tokens.append(s)
                labels.append(0)
                neg_labels.append(0)
            labels[-1] = tokens[-1]
            neg_labels[-1] = random_neq(1, self.item_num+1, seq)
            tokens[-1] = self.mask_token
        else:
            seq = self.data[index]
   
            for s in seq:
                prob = random.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(random.randint(1, self.item_num))
                    else:
                        tokens.append(s)

                    labels.append(s)
                    neg = random_neq(1, self.item_num+1, seq)
                    neg_labels.append(neg)

                else:
                    tokens.append(s)
                    labels.append(0)
                    neg_labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        neg_labels = neg_labels[-self.max_len:]
        pos = list(range(1, len(tokens)+1))
        pos= pos[-self.max_len:]

        mask_len = self.max_len - len(tokens)
        
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        neg_labels = [0] * mask_len + neg_labels
        pos = [0] * mask_len + pos

        if index >= len(self.data):
            user_id = index - len(self.data)
        else:
            user_id = index

        ### get the sequence of similar user
        sim_users = self.sim_users[user_id][:self.sim_user_num]
        sim_seq, sim_positions = [], []
        for sim_user in sim_users:
            meta_seq, meta_positions = self._get_user_seq(sim_user)
            sim_seq.append(meta_seq)
            sim_positions.append(meta_positions)
        
        sim_seq = np.array(sim_seq)
        sim_positions = np.array(sim_positions)

        return np.array(tokens), np.array(labels), np.array(neg_labels), np.array(pos), user_id, sim_seq, sim_positions


    def _get_user_seq(self, user):

        ### get the sequence of required user
        inter = self.data[user]
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions = positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, positions

