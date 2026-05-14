# here put the import lib
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.GRU4Rec import GRU4Rec
from models.SASRec import SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.utils import Multi_CrossAttention



class DualLLMGRU4Rec(GRU4Rec):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att

        # load llm embedding as item embedding
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

        self._load_item_labels(args)

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb

    def _load_item_labels(self, args):
        path = os.path.join("data", args.dataset, "handled", "item_label.pkl")
        self.item_labels = None
        self.num_item_categories = 1
        try:
            with open(path, "rb") as f:
                label = pickle.load(f)
            labels = label.get("labels", None)
            if labels is not None:
                if isinstance(labels, np.ndarray):
                    labels_t = torch.from_numpy(labels).long()
                else:
                    labels_t = torch.as_tensor(labels, dtype=torch.long)
                self.item_labels = labels_t
                self.num_item_categories = int(labels_t.max().item()) + 1 if labels_t.numel() > 0 else 1
        except Exception:
            self.item_labels = None
            self.num_item_categories = 1

    def _get_item_categories(self, item_ids):
        if self.item_labels is None:
            return None
        idx = item_ids.detach().cpu().long()
        idx.clamp_(0, self.item_labels.shape[0] - 1)
        cats = self.item_labels[idx]
        return cats.to(self.device)


    def log2feats(self, log_seqs):

        id_seqs = self.id_item_emb(log_seqs)
        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
    


class DualLLMSASRec(SASRec_seq):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        self.device = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att

        # load llm embedding as item embedding
        # llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset, "pca_itm_emb_np.pkl"), "rb"))
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)

        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

        self._load_item_labels(args)

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb

    def _load_item_labels(self, args):
        path = os.path.join("data", args.dataset, "handled", "item_label.pkl")
        self.item_labels = None
        self.num_item_categories = 1
        try:
            with open(path, "rb") as f:
                label = pickle.load(f)
            labels = label.get("labels", None)
            if labels is not None:
                if isinstance(labels, np.ndarray):
                    labels_t = torch.from_numpy(labels).long()
                else:
                    labels_t = torch.as_tensor(labels, dtype=torch.long)
                self.item_labels = labels_t
                self.num_item_categories = int(labels_t.max().item()) + 1 if labels_t.numel() > 0 else 1
        except Exception:
            self.item_labels = None
            self.num_item_categories = 1

    def _get_item_categories(self, item_ids):
        if self.item_labels is None:
            return None
        idx = item_ids.detach().cpu().long()
        idx.clamp_(0, self.item_labels.shape[0] - 1)
        cats = self.item_labels[idx]
        return cats.to(self.device)
    
    def _get_item_embedding(self, item_id):
        id_item_emb = self.id_item_emb.weight[item_id]
        llm_item_emb = self.llm_item_emb.weight[item_id]
        llm_item_emb = self.adapter(llm_item_emb)
        item_emb = torch.cat([id_item_emb,llm_item_emb], dim=-1)
        return item_emb
    
    def _get_weight_(self, user_id):
        u_label = self.u2label[user_id.cpu()] #b,k
        S = torch.matmul(u_label, u_label.T)
        # Normalize the similarity matrix row-wise
        S_min, _ = torch.min(S, dim=1, keepdim=True)
        S_max, _ = torch.max(S, dim=1, keepdim=True)
        S_normalized = (S - S_min) / (S_max - S_min)
        # Ensure the normalized values are in (0, 1)
        S_normalized = torch.clamp(S_normalized, 0, 1)
        return S_normalized.to(self.device)



    def log2feats(self, log_seqs, positions):
        # Collaborative view
        id_seqs = self.id_item_emb(log_seqs)    #B,L,D(64)
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)
        # Semantic view
        llm_seqs = self.llm_item_emb(log_seqs)  #B,L,D(1536)
        llm_seqs = self.adapter(llm_seqs)       ##B,L,D(64)--Linear layer transformation
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:# Cross-attention
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
            cross_id_seqs = 1 * cross_id_seqs + 0 * id_seqs
            cross_llm_seqs = 1 * cross_llm_seqs + 0 * llm_seqs
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)   # Collaborative view embedding
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs) # Semantic view embedding
        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)    # Final sequence embedding

        return log_feats
    


class DualLLMBert4Rec(Bert4Rec):

    def __init__(self, user_num, item_num, device, args):
        
        super().__init__(user_num, item_num, device, args)

        # self.user_num = user_num
        # self.item_num = item_num
        # self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads
        self.use_cross_att = args.use_cross_att

        # load llm embedding as item embedding
        # llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset, "pca_itm_emb_np.pkl"), "rb"))
        llm_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "itm_emb_np.pkl"), "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.llm_item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))    
        self.llm_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.adapter = nn.Linear(llm_item_emb.shape[1], args.hidden_size)

        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        id_item_emb = pickle.load(open(os.path.join("data/"+args.dataset+"/handled/", "pca64_itm_emb_np.pkl"), "rb"))
        id_item_emb = np.insert(id_item_emb, 0, values=np.zeros((1, id_item_emb.shape[1])), axis=0)
        id_item_emb = np.concatenate([id_item_emb, np.zeros((1, id_item_emb.shape[1]))], axis=0)
        self.id_item_emb = nn.Embedding.from_pretrained(torch.Tensor(id_item_emb))    
        self.id_item_emb.weight.requires_grad = True   # the grad is false in default
        # self.id_item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        if self.use_cross_att:
            self.llm2id = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)
            self.id2llm = Multi_CrossAttention(args.hidden_size, args.hidden_size, 2)

        if args.freeze: # freeze the llm embedding
            self.freeze_modules = ["llm_item_emb"]
            self._freeze()

        self.filter_init_modules = ["llm_item_emb", "id_item_emb"]
        self._init_weights()

        self._load_item_labels(args)

    
    def _get_embedding(self, log_seqs):

        id_seq_emb = self.id_item_emb(log_seqs)
        llm_seq_emb = self.llm_item_emb(log_seqs)
        llm_seq_emb = self.adapter(llm_seq_emb)

        item_seq_emb = torch.cat([id_seq_emb, llm_seq_emb], dim=-1)

        return item_seq_emb

    def _load_item_labels(self, args):
        path = os.path.join("data", args.dataset, "handled", "item_label.pkl")
        self.item_labels = None
        self.num_item_categories = 1
        try:
            with open(path, "rb") as f:
                label = pickle.load(f)
            labels = label.get("labels", None)
            if labels is not None:
                if isinstance(labels, np.ndarray):
                    labels_t = torch.from_numpy(labels).long()
                else:
                    labels_t = torch.as_tensor(labels, dtype=torch.long)
                self.item_labels = labels_t
                self.num_item_categories = int(labels_t.max().item()) + 1 if labels_t.numel() > 0 else 1
        except Exception:
            self.item_labels = None
            self.num_item_categories = 1

    def _get_item_categories(self, item_ids):
        if self.item_labels is None:
            return None
        idx = item_ids.detach().cpu().long()
        idx.clamp_(0, self.item_labels.shape[0] - 1)
        cats = self.item_labels[idx]
        return cats.to(self.device)


    def log2feats(self, log_seqs, positions):

        id_seqs = self.id_item_emb(log_seqs)
        id_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        id_seqs += self.pos_emb(positions.long())
        id_seqs = self.emb_dropout(id_seqs)

        llm_seqs = self.llm_item_emb(log_seqs)
        llm_seqs = self.adapter(llm_seqs)
        llm_seqs *= self.id_item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        llm_seqs += self.pos_emb(positions.long())
        llm_seqs = self.emb_dropout(llm_seqs)

        if self.use_cross_att:
            cross_id_seqs = self.llm2id(llm_seqs, id_seqs, log_seqs)
            cross_llm_seqs = self.id2llm(id_seqs, llm_seqs, log_seqs)
        else:
            cross_id_seqs = id_seqs
            cross_llm_seqs = llm_seqs

        id_log_feats = self.backbone(cross_id_seqs, log_seqs)
        llm_log_feats = self.backbone(cross_llm_seqs, log_seqs)

        log_feats = torch.cat([id_log_feats, llm_log_feats], dim=-1)

        return log_feats
