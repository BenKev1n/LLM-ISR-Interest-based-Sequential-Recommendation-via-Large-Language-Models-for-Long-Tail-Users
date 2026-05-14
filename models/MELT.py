import torch
import torch.nn as nn
import numpy as np
from models.SASRec import SASRec_seq
from models.GRU4Rec import GRU4Rec
from models.Bert4Rec import Bert4Rec

class USERBRANCH(nn.Module):
    def __init__(self, hidden_size, device, u_L_max):
        super(USERBRANCH, self).__init__()
        self.hidden_size = hidden_size
        self.W_U = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.W_U.weight.data)
        self.criterion = nn.MSELoss()
        self.device = device
        self.u_L_max = u_L_max
        self.pi = np.pi
        
    def forward(self, u_head_idx, backbone, user_context, user_thres, epoch, e_max=200):
        if len(u_head_idx) == 0:
            return torch.tensor(0.0).to(self.device)
            
        full_seq = []
        w_u_list = []
        for u_h in u_head_idx:
            u_h_item = u_h.item()
            seq = user_context[u_h_item]
            full_seq.append(seq)
            # seq is padded, length is number of non-zero
            seq_length = np.count_nonzero(seq)
            # Calculate the loss coefficient
            w_u = (self.pi/2)*(epoch/e_max)+(self.pi/(2*max(self.u_L_max-user_thres-1, 1)))*max(seq_length-user_thres-1, 0)
            w_u = np.abs(np.sin(w_u))
            w_u_list.append(w_u)
            
        full_seq = torch.tensor(np.array(full_seq)).to(self.device)
        w_u_list = torch.FloatTensor(w_u_list).view(-1, 1).to(self.device)
        
        # Representations of full sequence
        # sasrec/gru4rec/bert4rec output is (B, L, D) -> we want the last one
        if hasattr(backbone, 'log2feats'):
            if isinstance(backbone, Bert4Rec):
                positions = torch.arange(full_seq.size(1), dtype=torch.long, device=self.device).unsqueeze(0).expand(full_seq.size(0), -1)
                full_seq_repre = backbone.log2feats(full_seq, positions)[:, -1, :]
            elif isinstance(backbone, SASRec_seq):
                positions = torch.arange(full_seq.size(1), dtype=torch.long, device=self.device).unsqueeze(0).expand(full_seq.size(0), -1)
                full_seq_repre = backbone.log2feats(full_seq, positions)[:, -1, :]
            elif isinstance(backbone, GRU4Rec):
                full_seq_repre = backbone.log2feats(full_seq)[:, -1, :]
        
        few_seq = np.zeros([len(u_head_idx), full_seq.shape[1]], dtype=np.int32)
        R = np.random.randint(1, max(user_thres, 2), len(full_seq))
        full_seq_np = full_seq.cpu().numpy()
        for i, l in enumerate(R):
            few_seq[i, -l:] = full_seq_np[i, -l:]
            
        few_seq = torch.tensor(few_seq).to(self.device)
        
        if hasattr(backbone, 'log2feats'):
            if isinstance(backbone, Bert4Rec):
                positions = torch.arange(few_seq.size(1), dtype=torch.long, device=self.device).unsqueeze(0).expand(few_seq.size(0), -1)
                few_seq_repre = backbone.log2feats(few_seq, positions)[:, -1, :]
            elif isinstance(backbone, SASRec_seq):
                positions = torch.arange(few_seq.size(1), dtype=torch.long, device=self.device).unsqueeze(0).expand(few_seq.size(0), -1)
                few_seq_repre = backbone.log2feats(few_seq, positions)[:, -1, :]
            elif isinstance(backbone, GRU4Rec):
                few_seq_repre = backbone.log2feats(few_seq)[:, -1, :]

        loss = (w_u_list * ((self.W_U(few_seq_repre) - full_seq_repre) ** 2)).mean()
        return loss

class ITEMBRANCH(nn.Module):
    def __init__(self, hidden_size, device, i_L_max):
        super(ITEMBRANCH, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.i_L_max = i_L_max
        self.pi = np.pi
        self.W_I = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.W_I.weight.data)
        
    def forward(self, i_head_idx, item_context, backbone, item_thres, W_U, epoch, n_item_context, e_max=200):
        if len(i_head_idx) == 0:
            return torch.tensor(0.0).to(self.device)
            
        target_embed = []
        subseq_set = []
        subseq_set_idx = [0]
        idx = 0
        w_i_list = []
        
        for i, h_i in enumerate(i_head_idx):
            h_i_item = h_i.item()
            item_context_list = item_context[h_i_item]
            n_context = min(self.i_L_max, n_item_context[h_i_item])
            
            # Calculate the loss coefficient
            w_i = (self.pi/2)*(epoch/e_max)+(self.pi/100)*max(n_context-(item_thres+1), 0)
            w_i = np.abs(np.sin(w_i))
            w_i_list.append(w_i)
            len_context = len(item_context_list)
            
            # Set upper bound of item freq.
            thres = min(len_context, item_thres)
            n_few_inter = np.random.randint(1, max(thres+1, 2))
            
            # Randomly sample the contexts
            K = np.random.choice(range(len_context), int(n_few_inter), replace=False)
            idx += len(K)
            
            subseq_set.append(item_context_list[K])
            target_embed.append(h_i_item)
            subseq_set_idx.append(idx)
            
        subseq_set_np = np.vstack(subseq_set)
        subseq_set_tensor = torch.tensor(subseq_set_np).to(self.device)
        
        if hasattr(backbone, 'log2feats'):
            if isinstance(backbone, Bert4Rec):
                positions = torch.arange(subseq_set_tensor.size(1), dtype=torch.long, device=self.device).unsqueeze(0).expand(subseq_set_tensor.size(0), -1)
                subseq_repre_set = backbone.log2feats(subseq_set_tensor, positions)[:, -1, :]
            elif isinstance(backbone, SASRec_seq):
                positions = torch.arange(subseq_set_tensor.size(1), dtype=torch.long, device=self.device).unsqueeze(0).expand(subseq_set_tensor.size(0), -1)
                subseq_repre_set = backbone.log2feats(subseq_set_tensor, positions)[:, -1, :]
            elif isinstance(backbone, GRU4Rec):
                subseq_repre_set = backbone.log2feats(subseq_set_tensor)[:, -1, :]
                
        subseq_repre_set = subseq_repre_set + W_U(subseq_repre_set)
        
        context_repre = []
        for i in range(len(subseq_set_idx)-1):
            context_repre.append(subseq_repre_set[subseq_set_idx[i]:subseq_set_idx[i+1]].mean(0))
            
        context_repre = torch.stack(context_repre)
        
        target_embed_tensor = torch.tensor(target_embed).to(self.device)
        true_embed = backbone._get_embedding(target_embed_tensor)
        
        w_i_list = torch.FloatTensor(w_i_list).view(-1, 1).to(self.device)
        loss = (w_i_list * ((self.W_I(context_repre) - true_embed) ** 2)).mean()
        return loss

class MELT_SASRec(SASRec_seq):
    def __init__(self, user_num, item_num, device, args):
        super(MELT_SASRec, self).__init__(user_num, item_num, device, args)
        self.user_branch = USERBRANCH(args.hidden_size, device, args.max_len)
        self.item_branch = ITEMBRANCH(args.hidden_size, device, args.max_len)
        self.W_U = self.user_branch.W_U
        self.W_I = self.item_branch.W_I
        self.lamb_u = getattr(args, 'lamb_u', 1.0)
        self.lamb_i = getattr(args, 'lamb_i', 1.0)
        self.ts_user = getattr(args, 'ts_user', 12)
        self.ts_item = getattr(args, 'ts_item', 13)
        self.num_train_epochs = args.num_train_epochs

    def forward(self, seq, pos, neg, positions, **kwargs):
        # Standard recommendation loss
        loss = super().forward(seq, pos, neg, positions, **kwargs)
        
        # MELT branches
        if "u_h_idx" in kwargs and "i_h_idx" in kwargs:
            u_h_idx = kwargs["u_h_idx"]
            i_h_idx = kwargs["i_h_idx"]
            user_context = kwargs["user_context"]
            item_context = kwargs["item_context"]
            epoch = kwargs["epoch"]
            n_item_context = kwargs["n_item_context"]
            
            user_loss = self.user_branch(u_h_idx, self, user_context, self.ts_user, epoch, self.num_train_epochs)
            item_loss = self.item_branch(i_h_idx, item_context, self, self.ts_item, self.W_U, epoch, n_item_context, self.num_train_epochs)
            
            loss = loss + self.lamb_u * user_loss + self.lamb_i * item_loss
            
        return loss

class MELT_GRU4Rec(GRU4Rec):
    def __init__(self, user_num, item_num, device, args):
        super(MELT_GRU4Rec, self).__init__(user_num, item_num, device, args)
        self.user_branch = USERBRANCH(args.hidden_size, device, args.max_len)
        self.item_branch = ITEMBRANCH(args.hidden_size, device, args.max_len)
        self.W_U = self.user_branch.W_U
        self.W_I = self.item_branch.W_I
        self.lamb_u = getattr(args, 'lamb_u', 1.0)
        self.lamb_i = getattr(args, 'lamb_i', 1.0)
        self.ts_user = getattr(args, 'ts_user', 12)
        self.ts_item = getattr(args, 'ts_item', 13)
        self.num_train_epochs = args.num_train_epochs

    def forward(self, seq, pos, neg, positions, **kwargs):
        loss = super().forward(seq, pos, neg, positions, **kwargs)
        if "u_h_idx" in kwargs and "i_h_idx" in kwargs:
            user_loss = self.user_branch(kwargs["u_h_idx"], self, kwargs["user_context"], self.ts_user, kwargs["epoch"], self.num_train_epochs)
            item_loss = self.item_branch(kwargs["i_h_idx"], kwargs["item_context"], self, self.ts_item, self.W_U, kwargs["epoch"], kwargs["n_item_context"], self.num_train_epochs)
            loss = loss + self.lamb_u * user_loss + self.lamb_i * item_loss
        return loss

class MELT_Bert4Rec(Bert4Rec):
    def __init__(self, user_num, item_num, device, args):
        super(MELT_Bert4Rec, self).__init__(user_num, item_num, device, args)
        self.user_branch = USERBRANCH(args.hidden_size, device, args.max_len)
        self.item_branch = ITEMBRANCH(args.hidden_size, device, args.max_len)
        self.W_U = self.user_branch.W_U
        self.W_I = self.item_branch.W_I
        self.lamb_u = getattr(args, 'lamb_u', 1.0)
        self.lamb_i = getattr(args, 'lamb_i', 1.0)
        self.ts_user = getattr(args, 'ts_user', 12)
        self.ts_item = getattr(args, 'ts_item', 13)
        self.num_train_epochs = args.num_train_epochs

    def forward(self, seq, pos, neg, positions, **kwargs):
        loss = super().forward(seq, pos, neg, positions, **kwargs)
        if "u_h_idx" in kwargs and "i_h_idx" in kwargs:
            user_loss = self.user_branch(kwargs["u_h_idx"], self, kwargs["user_context"], self.ts_user, kwargs["epoch"], self.num_train_epochs)
            item_loss = self.item_branch(kwargs["i_h_idx"], kwargs["item_context"], self, self.ts_item, self.W_U, kwargs["epoch"], kwargs["n_item_context"], self.num_train_epochs)
            loss = loss + self.lamb_u * user_loss + self.lamb_i * item_loss
        return loss

