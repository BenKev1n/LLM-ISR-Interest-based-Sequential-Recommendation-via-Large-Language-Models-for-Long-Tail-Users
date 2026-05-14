# here put the import lib
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from utils.utils import metric_report, metric_len_report, record_csv, metric_pop_report
from utils.utils import metric_len_5group, metric_pop_5group


class SeqTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch_idx, batch in enumerate(prog_iter):

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            inputs = self._prepare_train_inputs(batch)#seq:B,200;pos:B,200;neg:B,200;B,200;user_id:B;sim_seq:B,10,200;sim_positions:B,10,200
            # Pass global_step for model warmup scheduling
            inputs['global_step'] = self.global_step
            # Batch logging hook: controlled by batch_log args
            if getattr(self.args, 'batch_log', False):
                try:
                    batch_user_ids = inputs.get('user_id', None)
                    if batch_user_ids is not None:
                        # to CPU list for logging
                        batch_user_ids = batch_user_ids.detach().cpu().tolist()
                        sim_map = getattr(self.generator.train_dataset, 'sim_users', None)
                        msg_parts = []
                        # Print first two samples to avoid long log
                        for j, uid in enumerate(batch_user_ids[:2]):
                            neighbors = []
                            if sim_map is not None:
                                try:
                                    neighbors = sim_map[int(uid)][:3]
                                except Exception:
                                    neighbors = []
                            msg_parts.append(f"idx={j}, uid={int(uid)}, neigh={neighbors}")
                        if msg_parts:
                            self.logger.info(f"[Train][Batch {batch_idx}] " + "; ".join(msg_parts))
                except Exception:
                    # Silent failure, does not affect training
                    pass
            # Tail reweighting removed: use standard unweighted losses
            loss = self.model(**inputs)
            
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()
            # Increment global_step after successful optimizer step
            self.global_step += 1

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)
        gate_count = 0
        w_dyn_sum = 0.0
        w_dyn_sq_sum = 0.0
        len_sum = 0.0
        len_sq_sum = 0.0
        sim_sum = 0.0
        sim_sq_sum = 0.0
        wlen_sum = 0.0
        wsim_sum = 0.0
        w_min = 1.0
        w_max = 0.0
        w_extreme = 0
        w_base_val = None

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                inputs['global_step'] = getattr(self, 'global_step', 0)
                pred_logits = -self.model.predict(self.model, **inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

                if hasattr(self.model, 'get_gate_info'):
                    try:
                        wb, wd, ln, sc = self.model.get_gate_info(inputs["seq"], inputs["positions"], inputs.get("user_id", None))
                        if w_base_val is None:
                            w_base_val = wb.detach().item()
                        wd = wd.detach()
                        ln = ln.detach()
                        sc = sc.detach()
                        gate_count += wd.numel()
                        w_dyn_sum += wd.sum().item()
                        w_dyn_sq_sum += (wd.pow(2).sum().item())
                        len_sum += ln.sum().item()
                        len_sq_sum += (ln.pow(2).sum().item())
                        sim_sum += sc.sum().item()
                        sim_sq_sum += (sc.pow(2).sum().item())
                        wlen_sum += (wd * ln).sum().item()
                        wsim_sum += (wd * sc).sum().item()
                        w_min = min(w_min, float(wd.min().item()))
                        w_max = max(w_max, float(wd.max().item()))
                        w_extreme += int(((wd < 0.1) | (wd > 0.9)).sum().item())
                    except Exception:
                        pass

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        # Print user group performance for each epoch
        self.logger.info("User Group Performance:")
        for k, v in res_len_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        self.logger.info("Item Group Performance:")
        for k, v in res_pop_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict, **res_pop_dict}

        if gate_count > 0 and w_base_val is not None:
            mean_w = w_dyn_sum / gate_count
            var_w = max(1e-8, (w_dyn_sq_sum / gate_count) - mean_w * mean_w)
            std_w = var_w ** 0.5
            mean_len = len_sum / gate_count
            var_len = max(1e-8, (len_sq_sum / gate_count) - mean_len * mean_len)
            std_len = var_len ** 0.5
            mean_sim = sim_sum / gate_count
            var_sim = max(1e-8, (sim_sq_sum / gate_count) - mean_sim * mean_sim)
            std_sim = var_sim ** 0.5
            cov_w_len = (wlen_sum / gate_count) - mean_w * mean_len
            cov_w_sim = (wsim_sum / gate_count) - mean_w * mean_sim
            corr_len = cov_w_len / (std_w * std_len)
            corr_sim = cov_w_sim / (std_w * std_sim)
            extreme_ratio = w_extreme / gate_count
            self.logger.info("Gate Statistics:")
            self.logger.info('\t w_base: %.4f' % (w_base_val))
            self.logger.info('\t w_dyn mean: %.4f, std: %.4f, min: %.4f, max: %.4f' % (mean_w, std_w, w_min, w_max))
            self.logger.info('\t len_norm mean: %.4f, corr(w_dyn,len_norm): %.3f' % (mean_len, corr_len))
            self.logger.info('\t sim_center mean: %.4f, corr(w_dyn,sim_center): %.3f' % (mean_sim, corr_sim))
            self.logger.info('\t extreme ratio (<0.1 or >0.9): %.3f' % (extreme_ratio))

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict
    


    def save_user_emb(self):

        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        try:
            self.model.load_state_dict(model_state_dict['state_dict'])
        except:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        test_loader = self.test_loader

        self.model.eval()
        user_emb = torch.empty(0).to(self.device)
        desc = 'Running'

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            
            with torch.no_grad():

                per_user_emb = self.model.get_user_emb(**inputs)
                user_emb = torch.cat([user_emb, per_user_emb], dim=0)
        
        user_emb = user_emb.detach().cpu().numpy()
        import pickle
        pickle.dump(user_emb, open("./usr_emb_sasrec.pkl", "wb"))


    
    def test_group(self):

        print('')
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running Group test **********")
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)
        test_loader = self.test_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(self.model, **inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        # res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        # res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)
        hr_len, ndcg_len, count_len = metric_len_5group(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), [5, 10, 15, 20])
        hr_pop, ndcg_pop, count_pop = metric_pop_5group(pred_rank.detach().cpu().numpy(), self.item_pop,  target_items.detach().cpu().numpy(), [10, 30, 60, 100])

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("User Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_len, ndcg_len)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        self.logger.info("Item Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_pop, ndcg_pop)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        
        
        return res_dict
    


class CL4SRecTrainer(SeqTrainer):

    def __init__(self, args, logger, writer, device, generator):
        
        super().__init__(args, logger, writer, device, generator)


    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            seq, pos, neg, positions, aug1, aug2 = batch
            seq, pos, neg, positions, aug1, aug2 = seq.long(), pos.long(), neg.long(), positions.long(), aug1.long(), aug2.long()
            aug = (aug1, aug2)
            loss = self.model(seq, pos, neg, positions, aug)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            # train_time is optional in this trainer; keep structure minimal

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



class SSEPTTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            loss = self.model(seq_user, pos_user, neg_user, seq, pos, neg, positions)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            # train_time is optional in this trainer; keep structure minimal

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            try:
                self.model.load_state_dict(model_state_dict['state_dict'])
            except:
                self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            seq_len = torch.cat([seq_len, torch.sum(seq>0, dim=1)])

            with torch.no_grad():

                pred_logits = -self.model.predict(seq_user, seq, torch.cat([pos_user.unsqueeze(1), neg_user], dim=1), torch.cat([pos.unsqueeze(1), neg], dim=1), positions)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len)
        
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        for k, v in res_len_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict
