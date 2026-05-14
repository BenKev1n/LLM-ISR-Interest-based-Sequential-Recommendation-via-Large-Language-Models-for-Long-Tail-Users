# here put the import lib
import torch
import torch.nn as nn
from models.DualLLMSRS import DualLLMSASRec
from models.utils import Contrastive_Loss2, Update_Intr_Item, NeigClFuse
import torch.nn.functional as F
import pickle, os
import numpy as np

def contr_loss(log_feats, sim_log_feats):
    log_feats = F.normalize(log_feats, p=2, dim=1)
    sim_log_feats = F.normalize(sim_log_feats, p=2, dim=2)
    pos_scr = torch.mul(log_feats.unsqueeze(1), sim_log_feats).sum(dim=-1)  #B,DxB,K,D->B,K,D->B,K
    pos_scr = torch.exp(pos_scr/0.1).sum(dim=-1)    #B
    neg_scr = torch.matmul(log_feats, log_feats.transpose(0,1))  #B,DxD,B->B,B
    neg_scr = torch.exp(neg_scr/0.1).sum(dim=-1)    #B
    contr_loss = -torch.log(pos_scr/neg_scr).sum()
    return contr_loss 
    

class LLMISR_SASRec(DualLLMSASRec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.lambda1 = args.lambda1
        self.user_sim_func = args.user_sim_func
        # self.item_reg = args.item_reg
        self.device = device
        self.hidden_size = args.hidden_size
        # Cache frequently used args to avoid referencing a non-existent self.args
        self.dataset = args.dataset
        self.ts_user = getattr(args, 'ts_user', 10)
        self.fuse_gate = nn.Parameter(torch.tensor(float(getattr(args, 'fuse_init', 0.5))))
        self.fuse_warmup = int(getattr(args, 'fuse_warmup', 2000))
        self.fuse_len_gamma = float(getattr(args, 'fuse_len_gamma', 0.0))
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2*args.hidden_size + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.gate_ln = nn.LayerNorm(2*args.hidden_size + 2)
        self.fuse_floor = float(getattr(args, 'fuse_floor', 0.1))
        self.fuse_floor_warmup = int(getattr(args, 'fuse_floor_warmup', max(int(getattr(args, 'fuse_warmup', 2000)), 1) * 2))
        self.use_fuse_gate = getattr(args, 'use_fuse_gate', True)
        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        
        self.W_U = nn.Sequential(
            nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        )
        # self.i_transfer = args.i_transfer

        self.W_I = nn.Sequential(
            nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        )

        self.update_in_item = Update_Intr_Item()
        self.ncf = NeigClFuse(args, device)

        # if self.item_reg:
        #     self.theta = args.theta
        #     self.reg = Contrastive_Loss2()

        # MoE cluster centers loading and hyperparameters
        self.tau = args.tau
        moe_tau = float(getattr(args, 'moe_tau', -1.0))
        self.moe_tau = moe_tau if moe_tau > 0.0 else float(self.tau)
        self.moe_topk = getattr(args, 'moe_topk', 1)
        self.cluster_moe_eta = float(getattr(args, 'cluster_moe_eta', 1.0))
        # self.moe_apply_mode = str(getattr(args, 'moe_apply_mode', 'last'))
        self.moe_dyn_source = str(getattr(args, 'moe_dyn_source', 'seq_last'))
        self.moe_dyn_lambda = float(getattr(args, 'moe_dyn_lambda', 0.5))
        self.moe_gamma = float(getattr(args, 'moe_gamma', 1.0))
        # self.moe_conf_mode = str(getattr(args, 'moe_conf_mode', 'none'))
        self.moe_conf_thresh = float(getattr(args, 'moe_conf_thresh', 0.0))
        self.moe_broadcast_min_len = int(getattr(args, 'moe_broadcast_min_len', 0))
        self.disable_cluster_moe = bool(getattr(args, 'disable_cluster_moe', False))
        # Try to load cluster centers from data directory
        self._cluster_loaded = False
        self._cluster_centers = None  # (K, hidden_size)

        self.ema_m = float(getattr(args, 'ema_m', 0.9))
        self.min_ema_count = int(getattr(args, 'min_ema_count', 5))
        self.ema_warmup = int(getattr(args, 'ema_warmup', getattr(args, 'tail_warmup', 2000)))
        self.ema_center_eta = float(getattr(args, 'ema_center_eta', 0.3))
        # self.use_sinkhorn_ema = bool(getattr(args, 'use_sinkhorn_ema', False))
        # self.sinkhorn_iters = int(getattr(args, 'sinkhorn_iters', 3))
        # self.sinkhorn_eps = float(getattr(args, 'sinkhorn_eps', 1e-6))
        self.ema_noise_eps = float(getattr(args, 'ema_noise_eps', 1e-4))
        self.ema_noise_interval = int(getattr(args, 'ema_noise_interval', 2000))
        self.score_mix = float(getattr(args, 'score_mix', 0.0))
        self.score_mix_schedule = str(getattr(args, 'score_mix_schedule', 'decay'))
        self.score_tau = float(getattr(args, 'score_tau', 1.0))
        self.score_mix_warmup = int(getattr(args, 'score_mix_warmup', 0))
        self.score_fuse_mode = str(getattr(args, 'score_fuse_mode', 'logit'))
        self.score_mix_mode = str(getattr(args, 'score_mix_mode', 'const'))
        self.detach_moe_weights = bool(getattr(args, 'detach_moe_weights', False))
        self.moe_grad_warmup = int(getattr(args, 'moe_grad_warmup', 0))
        self.center_refresh_interval = int(getattr(args, 'center_refresh_interval', 1))
        self.moe_hard = bool(getattr(args, 'moe_hard', False))
        self.top1_margin_weight = float(getattr(args, 'top1_margin_weight', 0.0))
        self.top1_margin_warmup = int(getattr(args, 'top1_margin_warmup', 0))
        self.top1_margin_m = float(getattr(args, 'top1_margin_m', 0.0))
        self.alpha = float(getattr(args, 'alpha', 1e-4))
        self.tail_warmup = int(getattr(args, 'tail_warmup', 2000))
        self._user_to_score_row = None
        self._user_llm_emb_raw = None

        self._init_weights()

        # CateAux: category prediction head for next-item category supervision
        self.num_item_categories = getattr(self, 'num_item_categories', 1)
        
        self.cat_experts = nn.ModuleList([
            nn.Linear(2*args.hidden_size, 2*args.hidden_size) for _ in range(self.num_item_categories)
        ])

        gate_in = 2*args.hidden_size
        self.cat_gate = nn.Sequential(
            nn.Linear(gate_in, gate_in),
            nn.ReLU(),
            nn.Linear(gate_in, self.num_item_categories)
        )
        # self.cat_moe_beta = getattr(args, 'cat_moe_beta', 0.0)
        # self.cat_temp = getattr(args, 'cat_temp', 1.0)
        # self.hist_lambda = getattr(args, 'hist_lambda', 0.3)
        self.beta = getattr(args, 'beta', 0.0)
    
    def forward(self,
                user_id, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        S = getattr(self, 'ts_user', 10)

        # 1) get long-term interest LTI（full-length sequence）
        log_feats_lti = self.log2feats(seq, positions)  # (B, L, 2H)
        pos_embs = self._get_embedding(pos)             # (B, L, 2H)
        neg_embs = self._get_embedding(neg)             # (B, L, 2H)

        # 2) get short-term interest STI（only keep the last S interactions, rest zeros）
        seq_sti, pos_sti, neg_sti, positions_sti = self.build_sti_inputs(seq, pos, neg, positions, S)
        log_feats_sti = self.log2feats(seq_sti, positions_sti)  # (B, L, 2H)

        # 2.1) apply cluster MoE to STI features
        log_feats_sti_mixed = self.apply_cluster_moe(log_feats_sti, user_id, kwargs.get('global_step', 0))
        next_item_cat_logits = self.predict_next_item_category(log_feats_sti_mixed, seq)
        # enable_cat_moe = (self.num_item_categories > 1) and ((self.cat_moe_beta is not None and self.cat_moe_beta > 0.0) or (self.beta is not None and self.beta > 0.0))
        # next_item_cat_logits = None
        # if enable_cat_moe:
        #     next_item_cat_logits = self.predict_next_item_category(log_feats_sti_mixed, seq)
        #     cat_weights_prior = torch.softmax(next_item_cat_logits / max(self.cat_temp, 1e-6), dim=-1)
        #     hist_cat_weights = self.user_history_category_weights(seq)
        #     cat_weights = self.combine_cat_weights(cat_weights_prior, hist_cat_weights)
        #     if self.cat_moe_beta is not None and self.cat_moe_beta > 0.0:
        #         log_feats_lti = self.apply_category_moe_residual_to_feats(log_feats_lti, cat_weights)
        #         log_feats_sti_mixed = self.apply_category_moe_residual_to_feats(log_feats_sti_mixed, cat_weights)
        H = self.hidden_size
        last_lti = log_feats_lti[:, -1, :]
        last_sti = log_feats_sti_mixed[:, -1, :]
        last_lti_llm = last_lti[:, H:]
        last_sti_llm = last_sti[:, H:]
        lengths = torch.sum(seq > 0, dim=1).float().unsqueeze(-1)
        len_norm = (lengths / max(1.0, float(seq.shape[1])))
        centers = self.get_cluster_centers(kwargs.get('global_step', 0))
        cluster_idx = self.assign_user_cluster(user_id=user_id, llm_feature=last_sti_llm, global_step=kwargs.get('global_step', 0)) if user_id is not None else torch.zeros(seq.shape[0], device=self.device, dtype=torch.long)
        center_vec = centers[cluster_idx]
        sim_center = torch.sum(F.normalize(last_sti_llm, dim=-1) * F.normalize(center_vec, dim=-1), dim=-1)  # [-1,1]
        sim_center = (sim_center + 1.0) * 0.5  # [0,1]
        
        if self.use_fuse_gate:
            gate_in = torch.cat([last_lti_llm, last_sti_llm, len_norm, sim_center.unsqueeze(-1)], dim=-1)
            gate_in = self.gate_ln(gate_in)
            w_dyn = torch.sigmoid(self.fuse_mlp(gate_in)).squeeze(-1)
            step = kwargs.get('global_step', 0)
            warmup_steps = max(int(getattr(self, 'fuse_warmup', 2000)), 1)
            warm = min(1.0, float(step) / float(warmup_steps))
            w_base = torch.sigmoid(self.fuse_gate)
            w_eff = (1.0 - warm) * w_base + warm * w_dyn
            flg = float(getattr(self, 'fuse_len_gamma', 0.0))
            if flg > 0.0:
                boost = flg * (1.0 - len_norm).squeeze(-1)
                w_eff = w_eff + boost * (w_dyn - w_eff)
            floor_warm = min(1.0, float(step) / float(self.fuse_floor_warmup))
            floor = self.fuse_floor * (1.0 - floor_warm)
            w_eff = torch.clamp(w_eff, floor, 1.0 - floor)
            fuse_reg_weight = float(getattr(self, 'fuse_reg_weight', 1e-4))
            fuse_reg = fuse_reg_weight * (w_dyn - w_base).pow(2).mean()
            log2_feats = (1.0 - w_eff.view(-1,1,1)) * log_feats_lti + w_eff.view(-1,1,1) * log_feats_sti_mixed
        else:
            # Disable fuse gate: just add LTI and STI' directly
            fuse_reg = 0.0
            log2_feats = log_feats_lti + log_feats_sti_mixed
            
        pos_logits = (log2_feats * pos_embs).sum(dim=-1)
        neg_logits = (log2_feats * neg_embs).sum(dim=-1)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)
        indices = (pos != 0)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss + fuse_reg
        if self.beta is not None and self.beta > 0.0 and next_item_cat_logits is not None:
            last_pos = pos[:, -1]
            cat_targets = self._get_item_categories(last_pos)
            if cat_targets is not None:
                mask_last = (last_pos != 0)
                if mask_last.any():
                    cat_aux_loss = F.cross_entropy(next_item_cat_logits[mask_last], cat_targets[mask_last])
                    loss = loss + self.beta * cat_aux_loss    # 3) calculate negative sample loss L_ncl
        log_feats = log2_feats[:, -1, :]    # B,D(128) get the sequence representation of semantic and interaction
        
        nceloss = 0
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])#B,10,200->Bx10,200
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]    # (bs*sim_num, hidden_size)
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        cl_loss = contr_loss(log_feats, sim_log_feats)  

        sim_log_feats = torch.mean(sim_log_feats, dim=1)    # Average over neighbors, (bs, H,D)

        if self.user_sim_func == "cl":
            align_term = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_term = self.align(self.W_U(log_feats), sim_log_feats + log_feats)
        
        # if self.i_transfer:
        #     item_emb, item_intr = self.update_in_item(self.W_U(log2_feats), seq)
        #     align_i_loss = self.align(self.W_I(item_emb), item_intr + item_emb)  # L_i
        #     loss = loss + align_i_loss

        # if self.item_reg:
        #     unfold_item_id = torch.masked_select(seq, seq>0)
        #     llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
        #     id_item_emb = self.id_item_emb(unfold_item_id)
        #     reg_loss = self.reg(llm_item_emb, id_item_emb)
        #     loss += self.theta * reg_loss

        # Top-1 HR enhancement: add BPR/margin auxiliary loss
        w_hr = float(getattr(self, 'top1_margin_weight', 0.0))
        if w_hr > 0.0:
            s = (pos_logits[indices] - neg_logits[indices])
            m = float(getattr(self, 'top1_margin_m', 0.0))
            if m > 0.0:
                top1_loss = torch.nn.functional.softplus(m - s).mean()
            else:
                top1_loss = (-torch.log(torch.sigmoid(s) + 1e-8)).mean()
            if self.top1_margin_warmup > 0:
                step = kwargs.get('global_step', 0)
                warm = min(1.0, float(step) / float(max(self.top1_margin_warmup, 1)))
                w_hr = w_hr * warm
            loss = loss + w_hr * top1_loss
        loss = loss + self.lambda1 *  align_term + 1e-4 * (cl_loss + nceloss)

        try:
            alpha = getattr(self, 'alpha', 1e-4)
            tau = max(getattr(self, 'tau', 1.0), 1e-6)
            lengths = torch.sum(seq > 0, dim=1)
            tail_mask = (lengths <= getattr(self, 'ts_user', 10))
            if tail_mask.any():
                H2 = log_feats.shape[-1]
                H = H2 // 2
                Z_sem = F.normalize(log_feats[:, H:], dim=1)
                llm_feat = log2_feats[:, -1, H:]
                cluster_idx = self.assign_user_cluster(user_id=user_id, llm_feature=llm_feat, global_step=kwargs.get('global_step', 0))
                centers = self.get_cluster_centers(kwargs.get('global_step', 0))
                K = centers.shape[0]
                B = Z_sem.shape[0]
                one_hot = torch.zeros(B, K, device=self.device)
                one_hot.scatter_(1, cluster_idx.view(-1, 1), 1.0)
                # if self.use_sinkhorn_ema:
                #     sim_w = torch.softmax((F.normalize(llm_feat, dim=-1) @ centers.T) / max(self.tau, 1e-6), dim=-1)
                #     w_bal = self.sinkhorn(sim_w, self.sinkhorn_iters, self.sinkhorn_eps)
                #     cluster_sums = w_bal.transpose(0, 1) @ Z_sem
                #     counts_bal = w_bal.sum(dim=0).clamp(min=1e-6)
                # else:
                #     cluster_sums = one_hot.transpose(0, 1) @ Z_sem
                #     counts_bal = one_hot.sum(dim=0).clamp(min=1.0)

                cluster_sums = one_hot.transpose(0, 1) @ Z_sem
                counts_bal = one_hot.sum(dim=0).clamp(min=1.0)

                batch_means = (cluster_sums / counts_bal.unsqueeze(1)).detach()
                if not hasattr(self, 'proto_ema') or self.proto_ema is None or self.proto_ema.shape[0] != K:
                    self.register_buffer('proto_ema', torch.zeros(K, H, device=self.device))
                    self.register_buffer('proto_count', torch.zeros(K, dtype=torch.long, device=self.device))
                    self.register_buffer('proto_last_step', torch.zeros(K, dtype=torch.long, device=self.device))
                present = (one_hot.sum(dim=0) > 0)
                if present.any():
                    m = self.ema_m
                    self.proto_ema[present] = m * self.proto_ema[present] + (1.0 - m) * batch_means[present]
                    self.proto_count[present] = self.proto_count[present] + one_hot.sum(dim=0)[present].long()
                    self.proto_last_step[present] = kwargs.get('global_step', 0)
                if self.ema_noise_eps > 0.0 and self.ema_noise_interval > 0:
                    cur = kwargs.get('global_step', 0)
                    idle_mask = (~present) & ((cur - self.proto_last_step) >= self.ema_noise_interval)
                    if idle_mask.any():
                        idxs = torch.nonzero(idle_mask, as_tuple=False).view(-1)
                        for k in idxs.tolist():
                            self.proto_ema[k] = self.proto_ema[k] + self.ema_noise_eps * torch.randn(H, device=self.device)
                use_ema = (self.proto_count[cluster_idx] >= self.min_ema_count)
                centers_norm = F.normalize(centers, dim=1)
                pos_proto = torch.where(use_ema.view(-1, 1), self.proto_ema[cluster_idx], centers_norm[cluster_idx])
                pos_proto = F.normalize(pos_proto, dim=1)
                sim_pos = (Z_sem * pos_proto).sum(dim=1) / tau
                exp_pos = torch.exp(sim_pos)
                all_idx = torch.arange(K, device=self.device)
                neg_mask_centers = (one_hot.sum(dim=0) > 0).unsqueeze(0).expand(B, K) & (cluster_idx.view(-1, 1) != all_idx.view(1, -1))
                sim_neg_centers = (Z_sem @ centers_norm.T) / tau
                exp_neg_sum = torch.exp(sim_neg_centers.masked_fill(~neg_mask_centers, float('-inf'))).sum(dim=1)
                info_nce = -torch.log(exp_pos / (exp_pos + exp_neg_sum + 1e-8))
                present_count = int((one_hot.sum(dim=0) > 0).sum().item())
                valid_cluster_mask = (present_count >= 2)
                final_mask = tail_mask & valid_cluster_mask
                if final_mask.any():
                    tail_loss = info_nce[final_mask].mean()
                    step = kwargs.get('global_step', 0)
                    warmup_steps = max(int(getattr(self, 'tail_warmup', 2000)), 1)
                    tail_contrast_weight = alpha * min(1.0, float(step) / float(warmup_steps))
                    loss = loss + tail_contrast_weight * tail_loss
        except Exception:
            pass

        return loss

    # def sinkhorn(self, weights: torch.Tensor, iters: int = 5, eps: float = 1e-6):
    #     B, K = weights.shape
    #     w = weights + eps
    #     target = (B / max(K, 1))
    #     for _ in range(max(iters, 1)):
    #         w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    #         col = w.sum(dim=0, keepdim=True) + 1e-8
    #         w = w / col
    #         w = w * target
    #     w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    #     return w

    def predict(self,
                model,
                seq,
                item_indices,
                positions,
                user_id=None,
                **kwargs):
        """When predicting, fuse the final representations of long-term and short-term interests."""
        S = getattr(self, 'ts_user', 10)

        log_feats_lti = self.log2feats(seq, positions)  # (B, L, 2H)
        final_lti = log_feats_lti[:, -1, :]
        lengths = torch.sum(seq > 0, dim=1).float().unsqueeze(-1)

        seq_sti, _, _, positions_sti = self.build_sti_inputs(seq, None, None, positions, S)
        log_feats_sti = self.log2feats(seq_sti, positions_sti)  # (B, L, 2H)
        final_sti = self.apply_cluster_moe_to_final(
            log_feats_sti[:, -1, :],
            user_id,
            kwargs.get('global_step', 0),
            lengths=lengths,
            seq_len=seq.shape[1],
        )

        # next_item_cat_logits = self.predict_next_item_category(log_feats_sti, seq)
        # cat_weights_prior = torch.softmax(next_item_cat_logits / max(self.cat_temp, 1e-6), dim=-1)
        # hist_cat_weights = self.user_history_category_weights(seq)
        # cat_weights = self.combine_cat_weights(cat_weights_prior, hist_cat_weights)
        # final_lti = self.apply_category_moe_residual_to_final(final_lti, cat_weights)
        # final_sti = self.apply_category_moe_residual_to_final(final_sti, cat_weights)

        if hasattr(self, 'u_transfer') and self.u_transfer:
            final_lti = model.W_U(final_lti)
            final_sti = model.W_U(final_sti)
        H = self.hidden_size
        len_norm = (lengths / max(1.0, float(seq.shape[1])))
        final_lti_llm = final_lti[:, H:]
        final_sti_llm = final_sti[:, H:]
        centers = self.get_cluster_centers(kwargs.get('global_step', 0))
        cluster_idx = self.assign_user_cluster(user_id=user_id, llm_feature=final_sti_llm, global_step=kwargs.get('global_step', 0)) if user_id is not None else torch.zeros(seq.shape[0], device=self.device, dtype=torch.long)
        center_vec = centers[cluster_idx]
        sim_center = torch.sum(F.normalize(final_sti_llm, dim=-1) * F.normalize(center_vec, dim=-1), dim=-1)
        sim_center = (sim_center + 1.0) * 0.5
        
        if self.use_fuse_gate:
            gate_in = torch.cat([final_lti_llm, final_sti_llm, len_norm, sim_center.unsqueeze(-1)], dim=-1)
            gate_in = self.gate_ln(gate_in)
            w_dyn = torch.sigmoid(self.fuse_mlp(gate_in)).squeeze(-1)
            w_base = torch.sigmoid(self.fuse_gate)
            floor = self.fuse_floor * 0.5
            w_eff = torch.clamp(w_base * 0.5 + w_dyn * 0.5, floor, 1.0 - floor)
            final_feat = (1.0 - w_eff.view(-1,1)) * final_lti + w_eff.view(-1,1) * final_sti
        else:
            final_feat = final_lti + final_sti
            
        item_embs = self._get_embedding(item_indices)
        # if self.i_transfer:
        #     item_embs = model.W_I(item_embs)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

    def get_gate_info(self, seq, positions, user_id=None):
        S = getattr(self, 'ts_user', 10)
        log_feats_lti = self.log2feats(seq, positions)
        final_lti = log_feats_lti[:, -1, :]
        lengths = torch.sum(seq > 0, dim=1).float().unsqueeze(-1)
        seq_sti, _, _, positions_sti = self.build_sti_inputs(seq, None, None, positions, S)
        log_feats_sti = self.log2feats(seq_sti, positions_sti)
        final_sti = self.apply_cluster_moe_to_final(
            log_feats_sti[:, -1, :],
            user_id,
            0,
            lengths=lengths,
            seq_len=seq.shape[1],
        )
        H = self.hidden_size
        len_norm = (lengths / max(1.0, float(seq.shape[1])))
        final_lti_llm = final_lti[:, H:]
        final_sti_llm = final_sti[:, H:]
        centers = self.get_cluster_centers(0)
        cluster_idx = self.assign_user_cluster(user_id=user_id, llm_feature=final_sti_llm, global_step=0) if user_id is not None else torch.zeros(seq.shape[0], device=self.device, dtype=torch.long)
        center_vec = centers[cluster_idx]
        sim_center = torch.sum(F.normalize(final_sti_llm, dim=-1) * F.normalize(center_vec, dim=-1), dim=-1)
        sim_center = (sim_center + 1.0) * 0.5
        gate_in = torch.cat([final_lti_llm, final_sti_llm, len_norm, sim_center.unsqueeze(-1)], dim=-1)
        gate_in = self.gate_ln(gate_in)
        w_dyn = torch.sigmoid(self.fuse_mlp(gate_in)).squeeze(-1)
        w_base = torch.sigmoid(self.fuse_gate)
        return w_base, w_dyn, len_norm.view(-1), sim_center.view(-1)

    def build_sti_inputs(self, seq, pos, neg, positions, S):
        B, L = seq.shape
        non_zero_mask = (seq > 0).int()
        lengths = torch.sum(non_zero_mask, dim=1)
        idxs = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        keep_threshold = (lengths - S).clamp(min=0).unsqueeze(1)
        keep_mask = (idxs >= keep_threshold).int() * non_zero_mask
        seq_sti = seq * keep_mask
        positions_sti = positions * keep_mask
        pos_sti = pos * keep_mask if pos is not None else None
        neg_sti = neg * keep_mask if neg is not None else None
        return seq_sti, pos_sti, neg_sti, positions_sti

    def build_cluster_dyn_feat(self, llm_part, user_id: torch.Tensor = None, global_step: int = 0, id_part=None, lengths=None, seq_len=None):
        """Build the dynamic feature used by Cluster-MoE consistently in train/infer."""
        dyn_feat = None
        src = self.moe_dyn_source.lower()
        if src == 'user_emb' and user_id is not None:
            dyn_feat = self.gather_user_llm_emb(user_id, global_step)
        elif src == 'both' and user_id is not None:
            ue = self.gather_user_llm_emb(user_id, global_step)
            if llm_part.dim() == 3:
                se = llm_part[:, -1, :]
                cur_seq_len = seq_len if seq_len is not None else (id_part.shape[1] if id_part is not None else llm_part.shape[1])
                cur_lengths = lengths
                if cur_lengths is None:
                    cur_lengths = torch.sum((id_part != 0).any(dim=-1), dim=1).float().unsqueeze(-1) if id_part is not None \
                        else torch.full((llm_part.shape[0], 1), float(cur_seq_len), device=llm_part.device)
            else:
                se = llm_part
                cur_seq_len = seq_len if seq_len is not None else 1
                cur_lengths = lengths
                if cur_lengths is None:
                    cur_lengths = torch.full((llm_part.shape[0], 1), float(cur_seq_len), device=llm_part.device)
            len_norm = (cur_lengths / max(1.0, float(cur_seq_len)))
            mdl = float(getattr(self, 'moe_dyn_lambda', 0.5))
            if str(getattr(self, 'moe_dyn_lambda_mode', 'len')).lower() == 'len':
                mdl = (1.0 - len_norm).clamp(0.0, 1.0)
            dyn_feat = F.normalize(mdl * ue + (1.0 - mdl) * se, dim=-1)
        if dyn_feat is None:
            dyn_feat = llm_part[:, -1, :] if llm_part.dim() == 3 else llm_part
        return dyn_feat

    def apply_cluster_moe(self, log_feats_sti, user_id: torch.Tensor = None, global_step: int = 0):
        """Apply cluster MoE to STI features."""
        B, L, D2 = log_feats_sti.shape
        H = D2 // 2
        id_part = log_feats_sti[:, :, :H]
        llm_part = log_feats_sti[:, :, H:]
        if self.disable_cluster_moe:
            return log_feats_sti
        centers = self.get_cluster_centers(global_step)  # (K, H)
        dyn_feat = self.build_cluster_dyn_feat(
            llm_part=llm_part,
            user_id=user_id,
            global_step=global_step,
            id_part=id_part,
        )
            
        use_similar_users = hasattr(self, '_cluster_ids_raw') and self._cluster_ids_raw is not None and self._user_llm_emb_raw is not None and user_id is not None
        
        if use_similar_users:
            """Merge similar users."""
            ids = user_id.detach().cpu().long()
            if self._user_to_score_row is not None:
                idx = ids.clamp(min=0, max=self._user_to_score_row.shape[0] - 1)
                rows = self._user_to_score_row[idx]
                rows = rows.clamp(min=0, max=self._user_llm_emb_raw.shape[0] - 1)
            else:
                rows = ids.clamp(min=0, max=self._user_llm_emb_raw.shape[0] - 1)
            
            rows_dev = rows.to(self.device)
            c_ids = self._cluster_ids_raw[rows_dev] # (B,)
            
            with torch.no_grad():
                all_raw = self._user_llm_emb_raw.to(self.device)
                if all_raw.shape[-1] == H:
                    all_emb = all_raw
                else:
                    all_emb = self.adapter(all_raw)
                all_emb = F.normalize(all_emb, dim=-1).detach()
                
            dyn_feat_norm = F.normalize(dyn_feat, dim=-1)
            sim = dyn_feat_norm @ all_emb.T # (B, N)
            
            # Mask
            mask = (self._cluster_ids_raw.unsqueeze(0) == c_ids.unsqueeze(1))
            self_mask = (torch.arange(all_emb.shape[0], device=self.device).unsqueeze(0) == rows_dev.unsqueeze(1))
            valid_mask = mask & (~self_mask)
            
            sim = sim.masked_fill(~valid_mask, -1e9)
            logits_dyn = sim / max(self.moe_tau, 1e-6)
            p_dyn = torch.softmax(logits_dyn, dim=-1)
            
            if self.moe_gamma is not None and self.moe_gamma > 1.0:
                pd = p_dyn.pow(self.moe_gamma)
                p_dyn = pd / (pd.sum(dim=-1, keepdim=True) + 1e-8)
                
            weights = self.apply_topk(p_dyn, all_emb.shape[0])
            
            if self.moe_hard:
                idx_max = torch.argmax(weights, dim=-1)
                one_hot = torch.zeros_like(weights)
                one_hot.scatter_(1, idx_max.view(-1, 1), 1.0)
                weights = one_hot
                
            mix = self.maybe_detach_moe_weights(weights, global_step) @ all_emb
            K_ent = max(self.moe_topk if self.moe_topk is not None and self.moe_topk > 0 else all_emb.shape[0], 1)
        else:
            """Merge cluster centers."""
            logits_dyn = (F.normalize(dyn_feat, dim=-1) @ centers.T) / max(self.moe_tau, 1e-6)
            p_dyn = torch.softmax(logits_dyn, dim=-1)
            if self.moe_gamma is not None and self.moe_gamma > 1.0:
                pd = p_dyn.pow(self.moe_gamma)
                p_dyn = pd / (pd.sum(dim=-1, keepdim=True) + 1e-8)
            if user_id is not None and self.has_cluster_scores():
                scores = self.gather_user_scores(user_id)
                weights = self.fuse_cluster_weights(p_dyn=p_dyn, logits_dyn=logits_dyn, scores=scores, global_step=global_step)
            else:
                weights = p_dyn
            weights = self.apply_topk(weights, centers.shape[0])
            # if self.use_sinkhorn_ema:
            #     weights = self.sinkhorn(weights, self.sinkhorn_iters, self.sinkhorn_eps)
            if self.moe_hard:
                idx = torch.argmax(weights, dim=-1)
                one_hot = torch.zeros_like(weights)
                one_hot.scatter_(1, idx.view(-1, 1), 1.0)
                weights = one_hot
            mix = (self.maybe_detach_moe_weights(weights, global_step) @ centers)
            K_ent = int(weights.shape[-1])
            
        if self.cluster_moe_eta != 1.0:
            mix = mix * self.cluster_moe_eta

        # if self.moe_conf_mode.lower() == 'entropy':
        #     ent = -(p_dyn * torch.log(p_dyn.clamp(min=1e-8))).sum(dim=-1)
        #     max_ent = float(np.log(max(K_ent, 1)))
        #     if max_ent > 0.0:
        #         conf_vec = (1.0 - ent / max_ent).clamp(min=0.0, max=1.0)
        #         thresh = float(getattr(self, 'moe_conf_thresh', 0.0))
        #         enable = (conf_vec >= thresh).float().unsqueeze(-1)
        #         mix = mix * enable * conf_vec.unsqueeze(-1)

        ent = -(p_dyn * torch.log(p_dyn.clamp(min=1e-8))).sum(dim=-1)
        max_ent = float(np.log(max(K_ent, 1)))
        if max_ent > 0.0:
            conf_vec = (1.0 - ent / max_ent).clamp(min=0.0, max=1.0)
            thresh = float(getattr(self, 'moe_conf_thresh', 0.0))
            enable = (conf_vec >= thresh).float().unsqueeze(-1)
            mix = mix * enable * conf_vec.unsqueeze(-1)

        # apply_mode = self.moe_apply_mode.lower()
        # if apply_mode == 'last':
        #     min_len = int(getattr(self, 'moe_broadcast_min_len', 0))
        #     if min_len > 0:
        #         lengths = torch.sum((id_part != 0).any(dim=-1), dim=1)
        #         use_broadcast = (lengths <= min_len)
        #     else:
        #         use_broadcast = torch.zeros(B, dtype=torch.bool, device=self.device)
        #     llm_part_mixed = llm_part.clone()
        #     if use_broadcast.any():
        #         mix_broadcast = mix.unsqueeze(1).expand(B, L, H)
        #         llm_part_mixed[use_broadcast] = llm_part_mixed[use_broadcast] + mix_broadcast[use_broadcast]
        #     else:
        #         llm_part_mixed[:, -1, :] = llm_part_mixed[:, -1, :] + mix
        # else:
        #     mix_broadcast = mix.unsqueeze(1).expand(B, L, H)
        #     llm_part_mixed = llm_part + mix_broadcast

        min_len = int(getattr(self, 'moe_broadcast_min_len', 0))
        if min_len > 0:
            lengths = torch.sum((id_part != 0).any(dim=-1), dim=1)
            use_broadcast = (lengths <= min_len)
        else:
            use_broadcast = torch.zeros(B, dtype=torch.bool, device=self.device)
        llm_part_mixed = llm_part.clone()
        if use_broadcast.any():
            mix_broadcast = mix.unsqueeze(1).expand(B, L, H)
            llm_part_mixed[use_broadcast] = llm_part_mixed[use_broadcast] + mix_broadcast[use_broadcast]
        else:
            llm_part_mixed[:, -1, :] = llm_part_mixed[:, -1, :] + mix

        return torch.cat([id_part, llm_part_mixed], dim=-1)

    def predict_next_item_category(self, seq_feats, seq_ids):
        """Predict the category of the next item using the last step sequence features."""
        B, L, D = seq_feats.shape
        last_feat = seq_feats[:, -1, :]  # (B, 2H)
        logits = self.cat_gate(last_feat)  # (B, C)
        return logits

    # def user_history_category_weights(self, seq):
    #     cats = self._get_item_categories(seq)
    #     if cats is None:
    #         return torch.zeros((seq.shape[0], self.num_item_categories), device=self.device)
    #     B, L = seq.shape
    #     C = self.num_item_categories
    #     weights = torch.zeros((B, C), device=self.device)
    #     mask = (seq > 0)
    #     cats_masked = torch.where(mask, cats, torch.full_like(cats, fill_value=-1))
    #     for c in range(C):
    #         weights[:, c] = ((cats_masked == c) & mask).sum(dim=1)
    #     weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    #     return weights

    # def apply_category_moe_residual_to_feats(self, feats, cat_weights):
    #     B, L, D2 = feats.shape
    #     H = D2 // 2
    #     id_part = feats[:, :, :H]
    #     llm_part = feats[:, :, H:]
    #     experts_out = torch.stack([self.cat_experts[i](feats) for i in range(self.num_item_categories)], dim=2)  # (B, L, C, 2H)
    #     llm_experts = experts_out[:, :, :, H:]  # (B, L, C, H)
    #     K = min(getattr(self, 'cat_topk', 3), self.num_item_categories)
    #     if K > 0 and K < self.num_item_categories:
    #         topk_vals, topk_idx = torch.topk(cat_weights, k=K, dim=-1)
    #         sparse_w = torch.zeros_like(cat_weights)
    #         sparse_w.scatter_(1, topk_idx, topk_vals)
    #         w = (sparse_w / (sparse_w.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(1).unsqueeze(-1)
    #     else:
    #         w = cat_weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, C, 1)
    #     mix_llm = (llm_experts * w).sum(dim=2)  # (B, L, H)
    #     llm_part = llm_part + self.cat_moe_beta * mix_llm
    #     return torch.cat([id_part, llm_part], dim=-1)

    # def apply_category_moe_residual_to_final(self, final_vec, cat_weights):
    #     B, D2 = final_vec.shape
    #     H = D2 // 2
    #     id_part = final_vec[:, :H]
    #     llm_part = final_vec[:, H:]
    #     experts_out = torch.stack([self.cat_experts[i](final_vec) for i in range(self.num_item_categories)], dim=1)  # (B, C, 2H)
    #     llm_experts = experts_out[:, :, H:]  # (B, C, H)
    #     K = min(getattr(self, 'cat_topk', 3), self.num_item_categories)
    #     if K > 0 and K < self.num_item_categories:
    #         topk_vals, topk_idx = torch.topk(cat_weights, k=K, dim=-1)
    #         sparse_w = torch.zeros_like(cat_weights)
    #         sparse_w.scatter_(1, topk_idx, topk_vals)
    #         w = sparse_w / (sparse_w.sum(dim=-1, keepdim=True) + 1e-8)
    #         w = w.unsqueeze(-1)
    #     else:
    #         w = cat_weights.unsqueeze(-1)  # (B, C, 1)
    #     mix_llm = (llm_experts * w).sum(dim=1)  # (B, H)
    #     llm_part = llm_part + self.cat_moe_beta * mix_llm
    #     return torch.cat([id_part, llm_part], dim=-1)

    def assign_user_cluster(self, user_id: torch.Tensor = None, llm_feature: torch.Tensor = None, global_step: int = 0) -> torch.Tensor:
        centers = self.get_cluster_centers(global_step)  # (K, H)
        K = centers.shape[0]
        if llm_feature is not None:
            sims = F.normalize(llm_feature, dim=-1) @ centers.T  # (B, K)
            if user_id is not None and self.has_cluster_scores():
                scores = self.gather_user_scores(user_id)
                logits_dyn = sims / max(self.moe_tau, 1e-6)
                p_dyn = torch.softmax(logits_dyn, dim=-1)
                weights = self.fuse_cluster_weights(p_dyn=p_dyn, logits_dyn=logits_dyn, scores=scores, global_step=global_step)
                return torch.argmax(weights, dim=-1)
            return torch.argmax(sims, dim=-1)
        if user_id is not None and self.has_cluster_scores():
            weights = self.gather_user_scores(user_id)  # (B, K)
            return torch.argmax(weights, dim=-1)
        B = int(user_id.shape[0]) if user_id is not None else 1
        return torch.zeros((B,), dtype=torch.long, device=self.device)

    # def combine_cat_weights(self, prior, hist):
    #     weights = (1 - self.hist_lambda) * prior + self.hist_lambda * hist
    #     weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
    #     return weights

    def apply_cluster_moe_to_final(self, final_sti, user_id: torch.Tensor = None, global_step: int = 0, lengths=None, seq_len=None):
        """Apply cluster MoE to the final short-term interest vector."""
        B, D2 = final_sti.shape
        H = D2 // 2
        id_part = final_sti[:, :H]
        llm_part = final_sti[:, H:]
        if self.disable_cluster_moe:
            return final_sti
        centers = self.get_cluster_centers(global_step)  # (K, H)
        dyn_feat = self.build_cluster_dyn_feat(
            llm_part=llm_part,
            user_id=user_id,
            global_step=global_step,
            lengths=lengths,
            seq_len=seq_len,
        )
            
        """Merge cluster centers."""
        logits_dyn = (F.normalize(dyn_feat, dim=-1) @ centers.T) / max(self.moe_tau, 1e-6)
        p_dyn = torch.softmax(logits_dyn, dim=-1)
        if self.moe_gamma is not None and self.moe_gamma > 1.0:
            pd = p_dyn.pow(self.moe_gamma)
            p_dyn = pd / (pd.sum(dim=-1, keepdim=True) + 1e-8)
        if user_id is not None and self.has_cluster_scores():
            scores = self.gather_user_scores(user_id)
            weights = self.fuse_cluster_weights(p_dyn=p_dyn, logits_dyn=logits_dyn, scores=scores, global_step=global_step)
        else:
            weights = p_dyn
        weights = self.apply_topk(weights, centers.shape[0])
        # if self.use_sinkhorn_ema:
        #     weights = self.sinkhorn(weights, self.sinkhorn_iters, self.sinkhorn_eps)
        if self.moe_hard:
            idx = torch.argmax(weights, dim=-1)
            one_hot = torch.zeros_like(weights)
            one_hot.scatter_(1, idx.view(-1, 1), 1.0)
            weights = one_hot
        mix = (self.maybe_detach_moe_weights(weights, global_step) @ centers)
        K_ent = int(weights.shape[-1])
        
        if self.cluster_moe_eta != 1.0:
            mix = mix * self.cluster_moe_eta

        # if self.moe_conf_mode.lower() == 'entropy':
        #     ent = -(p_dyn * torch.log(p_dyn.clamp(min=1e-8))).sum(dim=-1)
        #     max_ent = float(np.log(max(K_ent, 1)))
        #     if max_ent > 0.0:
        #         conf_vec = (1.0 - ent / max_ent).clamp(min=0.0, max=1.0)
        #         thresh = float(getattr(self, 'moe_conf_thresh', 0.0))
        #         enable = (conf_vec >= thresh).float().unsqueeze(-1)
        #         mix = mix * enable * conf_vec.unsqueeze(-1)

        ent = -(p_dyn * torch.log(p_dyn.clamp(min=1e-8))).sum(dim=-1)
        max_ent = float(np.log(max(K_ent, 1)))
        if max_ent > 0.0:
            conf_vec = (1.0 - ent / max_ent).clamp(min=0.0, max=1.0)
            thresh = float(getattr(self, 'moe_conf_thresh', 0.0))
            enable = (conf_vec >= thresh).float().unsqueeze(-1)
            mix = mix * enable * conf_vec.unsqueeze(-1)

        llm_part = llm_part + mix
        return torch.cat([id_part, llm_part], dim=-1)

    

    def get_cluster_centers(self, global_step: int = 0):
        """Return cached cluster centers adapted to hidden dimension H.
        Fallback to zero centers if file is missing.
        """
        if self._cluster_loaded and hasattr(self, '_cluster_centers_raw') and (self._cluster_centers_raw is not None):
            if self._cluster_centers is None:
                return self.refresh_cluster_centers(global_step)
            if self.center_refresh_interval <= 0:
                return self._cluster_centers
            last_step = int(getattr(self, '_cluster_centers_step', -1))
            if (global_step - last_step) >= self.center_refresh_interval:
                return self.refresh_cluster_centers(global_step)
            return self._cluster_centers
        path = os.path.join("data", self.dataset, "handled", "user_label.pkl")
        centers = None
        try:
            with open(path, "rb") as f:
                label = pickle.load(f)
            raw_centers = label.get('centers', None)
            if raw_centers is not None:
                if isinstance(raw_centers, torch.Tensor):
                    centers_t = raw_centers.detach().clone().to(self.device).float()  # (K, 1536)
                elif isinstance(raw_centers, np.ndarray):
                    centers_t = torch.from_numpy(raw_centers).to(self.device).float()  # (K, 1536)
                else:
                    centers_t = torch.as_tensor(raw_centers, dtype=torch.float32, device=self.device)  # (K, 1536)
                self._cluster_centers_raw = centers_t
                self._cluster_loaded = True
                centers = self.refresh_cluster_centers(global_step)
                print(f"[MoE] Loaded {centers.shape[0]} cluster centers from '{path}', adapted to H={centers.shape[1]}")
            if 'scores' in label:
                scores_raw = label['scores']
                if isinstance(scores_raw, torch.Tensor):
                    scores = scores_raw.detach().clone().float().cpu()  # (N, K)
                elif isinstance(scores_raw, np.ndarray):
                    scores = torch.from_numpy(scores_raw).float()  # CPU tensor (N, K)
                else:
                    scores = torch.as_tensor(scores_raw, dtype=torch.float32)  # CPU tensor (N, K)
                self._cluster_scores = scores 
                self.maybe_build_user_score_row_map()
                
            if 'cluster_ids' in label:
                cids_raw = label['cluster_ids']
                if isinstance(cids_raw, torch.Tensor):
                    cids = cids_raw.detach().clone().long().to(self.device)
                elif isinstance(cids_raw, np.ndarray):
                    cids = torch.from_numpy(cids_raw).long().to(self.device)
                else:
                    cids = torch.as_tensor(cids_raw, dtype=torch.long, device=self.device)
                self._cluster_ids_raw = cids
                
            if 'scores' in label:
                try:
                    num_users, num_clusters = self._cluster_scores.shape
                    if hasattr(self, 'user_num') and num_users != self.user_num:
                        print(f"[MoE][Warn] user scores size mismatch: scores_users={num_users} vs model.user_num={self.user_num}")
                    if centers is not None and num_clusters != centers.shape[0]:
                        print(f"[MoE][Warn] user scores cluster K mismatch: scores_K={num_clusters} vs centers_K={centers.shape[0]}")
                    else:
                        print(f"[MoE] Loaded user scores with shape {self._cluster_scores.shape}")
                except Exception as _:
                    print("[MoE][Warn] Failed to verify user scores shape; proceeding anyway.")
            else:
                self._cluster_scores = None
                print(f"[MoE] No precomputed user scores in '{path}'; using dynamic similarity gating.")
        except Exception as e:
            centers = None
            print(f"[MoE][Warn] Failed to load cluster centers from '{path}': {e}. Falling back to zero centers.")
        if centers is None:
            H = self.id_item_emb.embedding_dim
            centers = torch.zeros((1, H), device=self.device)
            print(f"[MoE] Using zero centers with H={H}; gating works but may reduce performance.")
        self._cluster_centers = centers
        self._cluster_centers_step = int(global_step)
        self._cluster_loaded = True
        return self._cluster_centers

    def refresh_cluster_centers(self, global_step: int = 0):
        if not hasattr(self, '_cluster_centers_raw') or self._cluster_centers_raw is None:
            H = self.id_item_emb.embedding_dim
            self._cluster_centers = torch.zeros((1, H), device=self.device)
            self._cluster_centers_step = int(global_step)
            return self._cluster_centers
        with torch.no_grad():
            if self._cluster_centers_raw.shape[-1] == self.id_item_emb.embedding_dim:
                centers_adapt = self._cluster_centers_raw
            else:
                centers_adapt = self.adapter(self._cluster_centers_raw)
            centers = F.normalize(centers_adapt, dim=-1).detach()
        self._cluster_centers = centers
        self._cluster_centers_step = int(global_step)
        return self._cluster_centers

    def maybe_detach_moe_weights(self, weights: torch.Tensor, global_step: int = 0):
        if self.detach_moe_weights:
            return weights.detach()
        if self.moe_grad_warmup > 0 and int(global_step) < self.moe_grad_warmup:
            return weights.detach()
        return weights

    def fuse_cluster_weights(self, p_dyn: torch.Tensor, logits_dyn: torch.Tensor, scores: torch.Tensor, global_step: int = 0):
        lam = float(self.score_mix)
        if lam <= 0.0:
            return p_dyn
        if self.score_mix_warmup > 0:
            t = min(1.0, max(0.0, float(global_step) / float(max(self.score_mix_warmup, 1))))
            sched = str(getattr(self, 'score_mix_schedule', 'decay')).lower()
            if sched == "ramp":
                lam = lam * t
            elif sched == "decay":
                lam = lam * (1.0 - t)
        if lam <= 0.0:
            return p_dyn
        if self.score_mix_mode.lower() == "entropy":
            K = int(scores.shape[-1])
            s = torch.clamp(scores, min=1e-8)
            ent = -(s * torch.log(s)).sum(dim=-1)
            max_ent = float(np.log(max(K, 1)))
            if max_ent > 0.0:
                conf = 1.0 - (ent / max_ent)
                conf = conf.clamp(min=0.0, max=1.0)
                lam = lam * conf
        if self.score_fuse_mode.lower() == 'prob':
            score_tau = max(float(getattr(self, 'score_tau', 1.0)), 1e-6)
            s = torch.clamp(scores, min=1e-8)
            if score_tau != 1.0:
                s = s.pow(1.0 / score_tau)
                s = s / (s.sum(dim=-1, keepdim=True) + 1e-8)
            if isinstance(lam, torch.Tensor):
                w = lam.unsqueeze(-1) * s + (1.0 - lam.unsqueeze(-1)) * p_dyn
            else:
                w = lam * s + (1.0 - lam) * p_dyn
            w = torch.clamp(w, min=0.0)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
            return w
        score_tau = max(float(getattr(self, 'score_tau', 1.0)), 1e-6)
        s = torch.clamp(scores, min=1e-8)
        if isinstance(lam, torch.Tensor):
            logits = (1.0 - lam.unsqueeze(-1)) * logits_dyn + lam.unsqueeze(-1) * (torch.log(s) / score_tau)
        else:
            logits = (1.0 - lam) * logits_dyn + lam * (torch.log(s) / score_tau)
        return torch.softmax(logits, dim=-1)

    def has_cluster_scores(self):
        return hasattr(self, '_cluster_scores') and (self._cluster_scores is not None)

    def maybe_build_user_score_row_map(self):
        if not self.has_cluster_scores():
            return
        if self._user_to_score_row is not None:
            return
        base = os.path.join("data", self.dataset, "handled")
        candidates = [
            os.path.join(base, "inter.txt"),
            os.path.join(base, "inter_seq.txt"),
        ]
        inter_path = None
        for p in candidates:
            if os.path.exists(p):
                inter_path = p
                break
        if inter_path is None:
            return
        seen = set()
        order = []
        try:
            with open(inter_path, "r") as f:
                for line in f:
                    parts = line.rstrip().split()
                    if not parts:
                        continue
                    u = int(parts[0])
                    if u not in seen:
                        seen.add(u)
                        order.append(u)
        except Exception:
            return
        if len(order) == 0:
            return
        size = int(getattr(self, "user_num", max(order)))
        mapping = torch.full((size + 1,), -1, dtype=torch.long)
        for idx, u in enumerate(order):
            if 0 <= u <= size:
                mapping[u] = idx
        self._user_to_score_row = mapping

    def maybe_load_user_llm_emb(self):
        if self._user_llm_emb_raw is not None:
            return
        base = os.path.join("data", self.dataset, "handled")
        p = os.path.join(base, "usr_emb_np.pkl")
        if not os.path.exists(p):
            self._user_llm_emb_raw = None
            return
        try:
            raw = pickle.load(open(p, "rb"))
            if isinstance(raw, torch.Tensor):
                emb = raw.detach().clone().float().cpu()
            elif isinstance(raw, np.ndarray):
                emb = torch.from_numpy(raw).float()
            else:
                emb = torch.as_tensor(raw, dtype=torch.float32).cpu()
            self._user_llm_emb_raw = emb
        except Exception:
            self._user_llm_emb_raw = None

    def gather_user_llm_emb(self, user_id: torch.Tensor, global_step: int = 0):
        self.maybe_load_user_llm_emb()
        if self._user_llm_emb_raw is None:
            return None
        if self._user_to_score_row is None:
            self.maybe_build_user_score_row_map()
        ids = user_id.detach().cpu().long()
        if self._user_to_score_row is not None:
            idx = ids.clamp(min=0, max=self._user_to_score_row.shape[0] - 1)
            rows = self._user_to_score_row[idx]
            rows = rows.clamp(min=0, max=self._user_llm_emb_raw.shape[0] - 1)
        else:
            rows = ids.clamp(min=0, max=self._user_llm_emb_raw.shape[0] - 1)
        x = self._user_llm_emb_raw[rows].to(self.device)
        with torch.no_grad():
            if x.shape[-1] == self.id_item_emb.embedding_dim:
                out = x
            else:
                out = self.adapter(x)
            out = F.normalize(out, dim=-1).detach()
        return out

    def gather_user_scores(self, user_id: torch.Tensor):
        """Gather precomputed scores for user IDs in batch.
        Return tensor shape (B, K) on current device; clamp if user_id out of range.
        """

        ids = user_id.detach().cpu().long()
        if self._user_to_score_row is not None:
            idx = ids.clamp(min=0, max=self._user_to_score_row.shape[0] - 1)
            rows = self._user_to_score_row[idx]
            valid = rows >= 0
            rows = rows.clamp(min=0, max=self._cluster_scores.shape[0] - 1)
            weights = self._cluster_scores[rows]  # (B, K) CPU
            if (~valid).any():
                weights = weights.clone()
                weights[~valid] = 1.0
        else:
            ids.clamp_(min=0, max=self._cluster_scores.shape[0] - 1)
            weights = self._cluster_scores[ids]  # (B, K) CPU
        weights = weights.to(self.device)
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights

    def apply_topk(self, weights: torch.Tensor, K: int):
        """Apply optional Top-K sparsity and normalization to weights."""
        if self.moe_topk is not None and self.moe_topk > 0 and self.moe_topk < K:
            topk_vals, topk_idx = torch.topk(weights, k=self.moe_topk, dim=-1)
            new_weights = torch.zeros_like(weights)
            new_weights.scatter_(1, topk_idx, topk_vals)
            weights = new_weights
        # Normalize the weights
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights
    