# here put the import lib
import os
import argparse
import torch

from generators.generator import Seq2SeqGeneratorAllUser
from generators.generator import GeneratorAllUser
from generators.bert_generator import BertGeneratorAllUser
from trainers.sequence_trainer import SeqTrainer
from utils.utils import set_seed
from utils.logger import Logger


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--model_name", 
                    default='llmisr_sasrec',
                    choices=[
                    "llmisr_sasrec", "gru4rec", "sasrec", "bert4rec", "melt_sasrec"
                    ],
                    type=str, 
                    required=False,
                    help="model name")
parser.add_argument("--dataset", 
                    default="yelp", 
                    choices=["yelp", "fashion", "beauty", "appliances", "video_games", "LastFM", "movielens"],  # preprocess by myself
                    help="Choose the dataset")
parser.add_argument("--inter_file",
                    default="inter",
                    type=str,
                    help="the name of interaction file")
parser.add_argument("--demo", 
                    default=False, 
                    action='store_true', 
                    help='whether run demo')
parser.add_argument("--pretrain_dir",
                    type=str,
                    default="sasrec_seq",
                    help="the path that pretrained model saved in")
parser.add_argument("--output_dir",
                    default='./saved/',
                    type=str,
                    required=False,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--check_path",
                    default='',
                    type=str,
                    help="the save path of checkpoints for different running")
parser.add_argument("--do_test",
                    default=False,
                    action="store_true",
                    help="whehther run the test on the well-trained model")
parser.add_argument("--do_emb",
                    default=False,
                    action="store_true",
                    help="save the user embedding derived from the SRS model")
parser.add_argument("--do_group",
                    default=False,
                    action="store_true",
                    help="conduct the group test")
parser.add_argument("--keepon",
                    default=False,
                    action="store_true",
                    help="whether keep on training based on a trained model")
parser.add_argument("--keepon_path",
                    type=str,
                    default="normal",
                    help="the path of trained model for keep on training")
parser.add_argument("--clip_path",
                    type=str,
                    default="",
                    help="the path to save the CLIP-pretrained embedding and adapter")
parser.add_argument("--ts_user",
                    type=int,
                    default=12,
                    help="the threshold to split the short and long seq")
parser.add_argument("--ts_item",
                    type=int,
                    default=13,
                    help="the threshold to split the long-tail and popular items")

parser.add_argument("--lamb_u", type=float, default=1.0, help="Weight for user branch in MELT")
parser.add_argument("--lamb_i", type=float, default=1.0, help="Weight for item branch in MELT")

# Model parameters
parser.add_argument("--hidden_size",
                    default=64,
                    type=int,
                    help="the hidden size of embedding")
parser.add_argument("--trm_num",
                    default=2,
                    type=int,
                    help="the number of transformer layer")
parser.add_argument("--num_heads",
                    default=1,
                    type=int,
                    help="the number of heads in Trm layer")
parser.add_argument("--num_layers",
                    default=1,
                    type=int,
                    help="the number of GRU layers")
parser.add_argument("--cl_scale",
                    type=float,
                    default=0.1,
                    help="the scale for contastive loss")
parser.add_argument("--mask_crop_ratio",
                    type=float,
                    default=0.3,
                    help="the mask/crop ratio for CL4SRec")
parser.add_argument("--tau",
                    default=0.75,
                    type=float,
                    help="the temperature for contrastive loss")
parser.add_argument("--sse_ratio",
                    default=0.4,
                    type=float,
                    help="the sse ratio for SSE-PT model")
parser.add_argument("--dropout_rate",
                    default=0.5,
                    type=float,
                    help="the dropout rate")
parser.add_argument("--max_len",
                    default=200,
                    type=int,
                    help="the max length of input sequence")
parser.add_argument("--mask_prob",
                    type=float,
                    default=0.4,
                    help="the mask probability for training Bert model")
parser.add_argument("--aug",
                    default=False,
                    action="store_true",
                    help="whether augment the sequence data")
parser.add_argument("--aug_seq",
                    default=False,
                    action="store_true",
                    help="whether use the augmented data")
parser.add_argument("--aug_seq_len",
                    default=0,
                    type=int,
                    help="the augmented length for each sequence")
parser.add_argument("--aug_file",
                    default="inter",
                    type=str,
                    help="the augmentation file name")
parser.add_argument("--train_neg",
                    default=1,
                    type=int,
                    help="the number of negative samples for training")
parser.add_argument("--test_neg",
                    default=100,
                    type=int,
                    help="the number of negative samples for test")
parser.add_argument("--suffix_num",
                    default=5,
                    type=int,
                    help="the suffix number for augmented sequence")
parser.add_argument("--prompt_num",
                    default=2,
                    type=int,
                    help="the number of prompts")
parser.add_argument("--freeze",
                    default=False,
                    action="store_true",
                    help="whether freeze the pretrained architecture when finetuning")
parser.add_argument("--pg",
                    default="length",
                    choices=['length', 'attention'],
                    type=str,
                    help="choose the prompt generator")
parser.add_argument("--use_cross_att",
                    default=False,
                    action="store_true",
                    help="whether add a cross-attention to interact the dual-view")
parser.add_argument("--lambda1",
                    default=0.1,
                    type=float,
                    help="the weight of auxiliary loss")
parser.add_argument("--user_sim_func",
                    default="kd",
                    type=str,
                    help="the type of user similarity function to derive the loss")
parser.add_argument("--item_reg",
                    default=False,
                    action="store_true",
                    help="whether regularize the item embedding by CL")
parser.add_argument("--theta",
                    default=0.1,
                    type=float,
                    help="the weight of regulation loss")
parser.add_argument("--sim_user_num",
                    default=10,
                    type=int,
                    help="the number of similar users for enhancement--10")
parser.add_argument("--split_backbone",
                    default=False,
                    action="store_true",
                    help="whether use a split backbone")
parser.add_argument("--co_view",
                    default=False,
                    action="store_true",
                    help="only use the collaborative view")
parser.add_argument("--se_view",
                    default=False,
                    action="store_true",
                    help="only use the semantic view")


# Other parameters
parser.add_argument("--train_batch_size",
                    default=128,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--lr",
                    default=0.001,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2",
                    default=0,
                    type=float,
                    help='The L2 regularization')
parser.add_argument("--num_train_epochs",
                    default=200,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--lr_dc_step",
                    default=1000,
                    type=int,
                    help='every n step, decrease the lr')
parser.add_argument("--lr_dc",
                    default=0,
                    type=float,
                    help='how many learning rate to decrease')
parser.add_argument("--patience",
                    type=int,
                    default=20,
                    help='How many steps to tolerate the performance decrease while training')
parser.add_argument("--watch_metric",
                    type=str,
                    default='NDCG@10',
                    help="which metric is used to select model.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for different data split")
parser.add_argument("--no_cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument('--gpu_id',
                    default=0,
                    type=int,
                    help='The device id.')
parser.add_argument('--num_workers',
                    default=8,
                    type=int,
                    help='The number of workers in dataloader')
parser.add_argument("--log", 
                    default=False,
                    action="store_true",
                    help="whether create a new log file")

# Toggle for per-batch logging of user_id and neighbors
parser.add_argument("--batch_log",
                    default=False,
                    action="store_true",
                    help="enable per-batch logging: print (index, user_id) and top-3 neighbors")

parser.add_argument("--u_transfer",
                    default=True,
                    action="store_true",
                    help="whehther run the W_U liner layer for user")
parser.add_argument("--i_transfer",
                    default=False,
                    action="store_true",
                    help="whehther run the W_I liner layer for item")

# params for the diffusion reverse model
parser.add_argument('--time_type', type=str, default='cat', help='(time_emb,x_t,condition)cat or add')
parser.add_argument('--dims', type=str, default='[100]', help='the dims for the DNN')  # 1000
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size') #  10
parser.add_argument('--diffuser_type', type=str, default='mlp2', help='type of diffuser.')
# 无条件扩散引导
parser.add_argument('--p', type=float, default=0.1, help='dropout ')
parser.add_argument('--w', type=float, default=1.0, help='dropout ')

# params for the forward process
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', action='store_true', default=True, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=5, help='steps of the forward process during inference')
parser.add_argument('--reweight', action='store_true', default=True, help='assign different weight to different timestep or not')

# 控制长短期兴趣融合比重的可学习参数
parser.add_argument("--fuse_init",
                    default=0.5,
                    type=float,
                    help="the weight of fuse gate")
parser.add_argument("--use_fuse_gate",
                    action="store_true",
                    default=True,
                    help="whether to use fuse gate for LTI and STI. If False, just add them.")
parser.add_argument("--disable_fuse_gate",
                    action="store_false",
                    dest="use_fuse_gate",
                    help="disable fuse gate for LTI and STI, use addition instead.")
parser.add_argument("--fuse_warmup",
                    default=2000,
                    type=int,
                    help="warmup steps for dynamic fuse gate")
parser.add_argument("--fuse_floor",
                    default=0.1,
                    type=float,
                    help="early floor constraint for gate extremes")
parser.add_argument("--fuse_floor_warmup",
                    default=4000,
                    type=int,
                    help="warmup steps for floor decay")

parser.add_argument("--ema_center_eta",
                    default=0.3,
                    type=float,
                    help="mixing weight between cluster center and EMA/batch proto")
parser.add_argument("--use_sinkhorn_ema",
                    default=False,
                    action="store_true",
                    help="use Sinkhorn-Knopp to balance cluster weights for proto update")
parser.add_argument("--sinkhorn_iters",
                    default=3,
                    type=int,
                    help="iterations for Sinkhorn balancing")
parser.add_argument("--sinkhorn_eps",
                    default=1e-6,
                    type=float,
                    help="epsilon for Sinkhorn numerical stability")
parser.add_argument("--ema_m",
                    default=0.9,
                    type=float,
                    help="EMA momentum for prototypes")
parser.add_argument("--ema_warmup",
                    default=2000,
                    type=int,
                    help="warmup for using EMA prototypes vs batch prototypes")
parser.add_argument("--min_ema_count",
                    default=5,
                    type=int,
                    help="minimum EMA count to trust EMA prototype")
parser.add_argument("--ema_noise_eps",
                    default=1e-4,
                    type=float,
                    help="random noise epsilon for idle cluster prototypes")
parser.add_argument("--ema_noise_interval",
                    default=2000,
                    type=int,
                    help="steps interval to add perturbation to idle clusters")
parser.add_argument("--alpha",
                    default=1e-4,
                    type=float,
                    help="weight for tail InfoNCE loss with warmup")
parser.add_argument("--tail_warmup",
                    default=2000,
                    type=int,
                    help="warmup steps for tail InfoNCE weight")

parser.add_argument("--moe_topk",
                    default=1,
                    type=int,
                    help="top-k for cluster MoE weights")
parser.add_argument("--moe_tau",
                    default=0.5,
                    type=float,
                    help="temperature for cluster MoE gating; <=0 uses --tau")
parser.add_argument("--cluster_moe_eta",
                    default=1.0,
                    type=float,
                    help="residual strength for adding cluster center mixture")
parser.add_argument("--moe_apply_mode",
                    default="last",
                    choices=["broadcast", "last"],
                    type=str,
                    help="apply cluster MoE on all steps or last step only")
parser.add_argument("--moe_dyn_source",
                    default="seq_last",
                    choices=["seq_last", "user_emb", "both"],
                    type=str,
                    help="feature source for dynamic gating")
parser.add_argument("--moe_dyn_lambda",
                    default=0.5,
                    type=float,
                    help="mix ratio when moe_dyn_source=both (user_emb weight)")
parser.add_argument("--moe_dyn_lambda_mode",
                    default="len",
                    choices=["const", "len"],
                    type=str,
                    help="alpha schedule when moe_dyn_source=both")
parser.add_argument("--moe_gamma",
                    default=1.0,
                    type=float,
                    help="sharpness for dynamic gating distribution")
parser.add_argument("--moe_conf_mode",
                    default="entropy",
                    choices=["none", "entropy"],
                    type=str,
                    help="confidence to scale cluster residual")
parser.add_argument("--moe_conf_thresh",
                    default=0.0,
                    type=float,
                    help="threshold on confidence to enable cluster residual")
parser.add_argument("--moe_broadcast_min_len",
                    default=0,
                    type=int,
                    help="apply broadcast mixing when sequence length <= this value")
parser.add_argument("--fuse_len_gamma",
                    default=0.3,
                    type=float,
                    help="boost short-seq fusion towards short-term for HR improvement")
parser.add_argument("--moe_hard",
                    default=True,
                    action="store_true",
                    help="use hard top-1 expert selection when mixing centers")
parser.add_argument("--top1_margin_weight",
                    default=0.1,
                    type=float,
                    help="weight for auxiliary BPR/margin loss to boost HR")
parser.add_argument("--top1_margin_warmup",
                    default=2000,
                    type=int,
                    help="warmup steps for top1 margin loss")
parser.add_argument("--top1_margin_m",
                    default=0.0,
                    type=float,
                    help="margin m for hinge/softplus in top1 loss")
parser.add_argument("--disable_cluster_moe",
                    default=False,
                    action="store_true",
                    help="disable cluster MoE module")
parser.add_argument("--score_mix",
                    default=0.0,
                    type=float,
                    help="mix ratio of precomputed cluster scores")
parser.add_argument("--score_mix_schedule",
                    default="decay",
                    choices=["const", "decay", "ramp"],
                    type=str,
                    help="schedule for score_mix when score_mix_warmup > 0")
parser.add_argument("--score_tau",
                    default=1.0,
                    type=float,
                    help="temperature for precomputed scores; <1 sharpens, >1 smooths")
parser.add_argument("--score_mix_warmup",
                    default=0,
                    type=int,
                    help="warmup steps for score_mix schedule")
parser.add_argument("--score_fuse_mode",
                    default="logit",
                    choices=["logit", "prob"],
                    type=str,
                    help="how to fuse precomputed scores and dynamic weights")
parser.add_argument("--score_mix_mode",
                    default="const",
                    choices=["const", "entropy"],
                    type=str,
                    help="how to scale score_mix across users")
parser.add_argument("--detach_moe_weights",
                    default=False,
                    action="store_true",
                    help="detach MoE weights when mixing cluster centers")
parser.add_argument("--moe_grad_warmup",
                    default=0,
                    type=int,
                    help="warmup steps to detach MoE weights before enabling gradients")
parser.add_argument("--center_refresh_interval",
                    default=1,
                    type=int,
                    help="steps interval to refresh adapted cluster centers")

parser.add_argument("--cat_moe_beta",
                    default=0.0,
                    type=float,
                    help="beta for category MoE gate")
parser.add_argument("--cat_temp",
                    default=1.0,
                    type=float,
                    help="temperature for category MoE gate")
parser.add_argument("--hist_lambda",
                    default=0.3,
                    type=float,
                    help="lambda for history MoE gate")
parser.add_argument("--beta",
                    default=0.0,
                    type=float,
                    help="weight for category auxiliary loss")
parser.add_argument("--cat_topk",
                    default=5,
                    type=int,
                    help="top-k for category auxiliary loss")


torch.autograd.set_detect_anomaly(True)

args = parser.parse_args()
set_seed(args.seed) # fix the random seed
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.pretrain_dir = os.path.join(args.output_dir, args.pretrain_dir)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.keepon_path = os.path.join(args.output_dir, args.keepon_path)
args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    if args.model_name in ['gru4rec']:
        generator = GeneratorAllUser(args, logger, device)
    elif args.model_name in ["bert4rec"]:
        generator = BertGeneratorAllUser(args, logger, device)
    elif args.model_name in ["llmisr_sasrec", "sasrec", "melt_sasrec"]:
        generator = Seq2SeqGeneratorAllUser(args, logger, device)
    else:
        raise ValueError

    # model training
    if args.model_name == "melt_sasrec":
        from trainers.melt_trainer import MELTTrainer
        trainer = MELTTrainer(args, logger, writer, device, generator)
    else:
        trainer = SeqTrainer(args, logger, writer, device, generator)

    if args.do_test:
        trainer.test()
    elif args.do_emb:
        trainer.save_user_emb()
    elif args.do_group:
        trainer.test_group()
    else:
        trainer.train()

    log_manager.end_log()   # delete the logger threads


if __name__ == "__main__":
    main()



