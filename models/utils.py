# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
    


class Contrastive_Loss2(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau


    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    


class CalculateAttention(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, Q, K, V, mask):

        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention



class Multi_CrossAttention(nn.Module):
    """
    cross-attention forward: x,y is two models' hidden layer, x as q, y as k and v
    """
    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # Input dimension
        self.all_head_size  = all_head_size     # Output dimension
        self.num_heads      = head_num          # Number of heads
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)


    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    

    def forward(self,x,y,log_seqs):
        """
        cross-attention forward: x,y is two models' hidden layer, x as q, y as k and v
        """

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # attention_mask = attention_mask.eq(0)
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output

class Update_Intr_Item(torch.nn.Module):
    def __init__(self):

        super(Update_Intr_Item, self).__init__()

        self.updata_strategy='pooling'
        self.wind_size = 11
        self.pool = torch.nn.AvgPool1d(self.wind_size, stride=1, padding=int((self.wind_size -1)/2)) # 
        self.pool.requires_grad = False  

    def forward(self, log_feats, log_seqs):
        if self.updata_strategy=='pooling':
            item_seq_emb = self.pool(log_feats.transpose(2,1)).transpose(2,1) # B,L,D->B,D,L->B,D,L1->B,L1,D
        else:
            item_seq_emb=log_feats

        index = torch.tensor(log_seqs.clone().detach()).view(-1)   # B,L-> B*L
        item_seq_emb = item_seq_emb.contiguous().view(-1, log_feats.size(-1))   # B,L1,D->B*L1,D ;Fusion the context information of the sequence
        # Create a bool mask to filter out the padding tokens
        non_zero_mask = index != 0
        index = index[non_zero_mask]    # Select the non-zero elements
        item_seq_emb = item_seq_emb[non_zero_mask]
        return log_feats[non_zero_mask], item_seq_emb#loss


from sklearn.metrics import pairwise_distances

class NeigClFuse(torch.nn.Module):
    def __init__(self, args, device):
        """
        Neighbor constrastive learning/Fusion module: Enhance the tail user representation
        """
        super(NeigClFuse, self).__init__()
        self.args = args
        self.device = device

        # User classifier
        num_gcn_layers = 2
        self.neig_enc = Encoder(num_gcn_layers, device).to(self.device)
        self.num_clusters = 10

    def forward(self, seq_emb):
        user_emb = self.neig_user_agg_layer(seq_emb)
        return user_emb
    
    def neig_user_agg_layer(self, seq_emb):
        user_relation_graph= construct_user_relation_graph_via_user(seq_emb)
        topk_user_relation_graph = select_topk_neighboehood(user_relation_graph, 0, neighborhood_threshold=1.0)
        agg_user_emb = self.neig_enc(topk_user_relation_graph, seq_emb)
        return agg_user_emb, topk_user_relation_graph

def construct_user_relation_graph_via_user(user_embedding, similarity_metric='cosine'):
    user_embedding = user_embedding.detach().cpu().numpy()
    # construct the user relation graph.
    adj = pairwise_distances(user_embedding, metric=similarity_metric)
    if similarity_metric == 'cosine':
        return adj
    else:
        return -adj

def select_topk_neighboehood(user_realtion_graph, neighborhood_size, neighborhood_threshold):
    topk_user_relation_graph = np.zeros(user_realtion_graph.shape, dtype='float32')
    if neighborhood_size > 0:
        for user in range(user_realtion_graph.shape[0]):
            user_neighborhood = user_realtion_graph[user]
            topk_indexes = user_neighborhood.argsort()[-neighborhood_size:][::-1]
            for i in topk_indexes:
                topk_user_relation_graph[user][i] = 1/neighborhood_size
    else:
        similarity_threshold = np.mean(user_realtion_graph)*neighborhood_threshold
        for i in range(user_realtion_graph.shape[0]):
            high_num = np.sum(user_realtion_graph[i] > similarity_threshold)
            if high_num > 0:
                for j in range(user_realtion_graph.shape[1]):
                    if user_realtion_graph[i][j] > similarity_threshold:
                        topk_user_relation_graph[i][j] = 1/high_num
            else:
                topk_user_relation_graph[i][i] = 1
    return topk_user_relation_graph


class Encoder(nn.Module):
    def __init__(self,num_gcn_layers, device):
        super(Encoder, self).__init__()
        self.device = device
        self.gcn_layers = nn.Sequential(*[GCNLayer() for i in range(num_gcn_layers)])

    def forward(self, encoder_adj, item_emb):
        encoder_adj = torch.tensor(encoder_adj).to(self.device)
        embeds = [item_emb]
        for i, gcn in enumerate(self.gcn_layers):
            embeds.append(gcn(encoder_adj, embeds[-1]))
        return sum(embeds)#, embeds

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)
    
