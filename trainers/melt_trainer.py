import time
import torch
import numpy as np
from tqdm import tqdm
from trainers.sequence_trainer import SeqTrainer
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset

class MELTTrainer(SeqTrainer):
    def __init__(self, args, logger, writer, device, generator):
        super().__init__(args, logger, writer, device, generator)
        self.u_head_idx = []
        self.i_head_idx = []
        self.user_context = {}
        self.item_context = {}
        self.n_item_context = {}
        self._build_context()

    def _build_context(self):
        user_train = self.generator.train
        user_thres = getattr(self.args, 'ts_user', 12)
        item_thres = getattr(self.args, 'ts_item', 13)
        max_len = self.args.max_len
        
        # build user_context
        for u, seq in user_train.items():
            pad_len = max_len - len(seq)
            if pad_len > 0:
                padded_seq = [0] * pad_len + seq
            else:
                padded_seq = seq[-max_len:]
            self.user_context[u] = padded_seq
            if len(seq) >= user_thres:
                self.u_head_idx.append(u)
                
        # build item_context
        item_users = defaultdict(list)
        for u, seq in user_train.items():
            for i in seq:
                item_users[i].append(u)
                
        for i, users in item_users.items():
            # item_context stores sequences of users who interacted with item i
            self.item_context[i] = np.array([self.user_context[u] for u in users])
            self.n_item_context[i] = len(users)
            if len(users) >= item_thres:
                self.i_head_idx.append(i)
                
        self.u_head_idx = np.array(self.u_head_idx)
        self.i_head_idx = np.array(self.i_head_idx)
        
        # DataLoaders for branches
        if len(self.u_head_idx) > 0 and len(self.train_loader) > 0:
            u_batch_size = max(len(self.u_head_idx) // max((len(self.train_loader)-1), 1), 1)
            u_dataset = TensorDataset(torch.tensor(self.u_head_idx))
            self.u_h_loader = DataLoader(u_dataset, batch_size=u_batch_size, shuffle=True, drop_last=False)
        else:
            self.u_h_loader = None
            
        if len(self.i_head_idx) > 0 and len(self.train_loader) > 0:
            i_batch_size = max(len(self.i_head_idx) // max((len(self.train_loader)-1), 1), 1)
            i_dataset = TensorDataset(torch.tensor(self.i_head_idx))
            self.i_h_loader = DataLoader(i_dataset, batch_size=i_batch_size, shuffle=True, drop_last=False)
        else:
            self.i_h_loader = None

    def _train_one_epoch(self, epoch):
        tr_loss = 0
        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')
        
        u_iter = iter(self.u_h_loader) if self.u_h_loader is not None else None
        i_iter = iter(self.i_h_loader) if self.i_h_loader is not None else None

        for batch_idx, batch in enumerate(prog_iter):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_train_inputs(batch)
            inputs['global_step'] = self.global_step
            
            u_h_batch = None
            i_h_batch = None
            if u_iter is not None:
                try:
                    u_h_batch = next(u_iter)[0]
                except StopIteration:
                    u_iter = iter(self.u_h_loader)
                    u_h_batch = next(u_iter)[0]
                    
            if i_iter is not None:
                try:
                    i_h_batch = next(i_iter)[0]
                except StopIteration:
                    i_iter = iter(self.i_h_loader)
                    i_h_batch = next(i_iter)[0]
            
            inputs['u_h_idx'] = u_h_batch.to(self.device) if u_h_batch is not None else torch.tensor([]).to(self.device)
            inputs['i_h_idx'] = i_h_batch.to(self.device) if i_h_batch is not None else torch.tensor([]).to(self.device)
            inputs['user_context'] = self.user_context
            inputs['item_context'] = self.item_context
            inputs['n_item_context'] = self.n_item_context
            inputs['epoch'] = epoch
            
            self.optimizer.zero_grad()
            loss = self.model(**inputs)
            loss.backward()
            self.optimizer.step()
            
            tr_loss += loss.item()
            self.global_step += 1
            
        return tr_loss / max(len(self.train_loader), 1)
