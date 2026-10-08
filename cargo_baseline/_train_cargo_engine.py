# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer_DCL
from clustercontrast.trainers_no_ema_loss import (
    ClusterContrastTrainer_PCLMP_NoEMALoss,
)
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance,compute_modal_invariant_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
import os
import torch.utils.data as data
from torch.autograd import Variable
import math
from collections import Counter
from scipy.optimize import linear_sum_assignment
from cargo_evaluation import evaluate_cargo_protocols
start_epoch = best_mAP = 0

def get_data(name, data_dir,trial=0):
    # Both CARGO view-domain loaders read the same official dataset root.
    dataset = datasets.create(name, data_dir, trial=trial)
    return dataset


def apply_smoke_subset(dataset, max_ids):
    """Deterministically restrict training data for plumbing-only tests."""
    if max_ids <= 0:
        return
    selected_ids = sorted({pid for _, pid, _ in dataset.train})[:max_ids]
    selected_ids = set(selected_ids)
    before = len(dataset.train)
    dataset.train = [sample for sample in dataset.train
                     if sample[1] in selected_ids]
    print("[SMOKE SUBSET] {} -> {} images, {} parsed identities".format(
        before, len(dataset.train), len(selected_ids)))
def check_feature_rows_finite(name, tensor, samples, epoch, enabled):
    """Stop before clustering/Memory and report the exact bad feature row."""
    if not enabled:
        return
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite_rows = finite_mask.all(dim=1)
    if bool(finite_rows.all()):
        return

    bad_indexes = torch.nonzero(~finite_rows, as_tuple=False).flatten().tolist()
    details = []
    for index in bad_indexes[:10]:
        row = detached[index]
        row_finite = torch.isfinite(row)
        finite = row[row_finite]
        if finite.numel() > 0:
            value_range = 'min={:.6f}, max={:.6f}'.format(
                finite.min().item(), finite.max().item())
        else:
            value_range = 'no finite values'
        identifier = 'cluster_index={}'.format(index)
        if samples is not None and index < len(samples):
            identifier = 'sample_index={}, path={}'.format(index, samples[index][0])
        details.append(
            '{}; {}; nan={}; posinf={}; neginf={}'.format(
                identifier, value_range,
                int(torch.isnan(row).sum().item()),
                int(torch.isposinf(row).sum().item()),
                int(torch.isneginf(row).sum().item())))
    raise RuntimeError(
        '[NonFinite FeatureRows] {}: epoch={}, shape={}, bad_rows={}; {}'.format(
            name, epoch, tuple(detached.shape), len(bad_indexes),
            ' | '.join(details)))


def summarize_clustering(name, labels, epoch, eps):
    """Print label-free DBSCAN health statistics and return them."""
    labels = np.asarray(labels, dtype=np.int64)
    total = int(labels.size)
    valid = labels[labels >= 0]
    noise = total - int(valid.size)
    if valid.size:
        _, counts = np.unique(valid, return_counts=True)
        counts = counts.astype(np.int64, copy=False)
    else:
        counts = np.empty(0, dtype=np.int64)

    stats = {
        'name': name,
        'epoch': int(epoch),
        'eps': float(eps),
        'total': total,
        'clustered': int(valid.size),
        'noise': noise,
        'noise_ratio': float(noise / total) if total else 0.0,
        'clusters': int(counts.size),
        'size_min': int(counts.min()) if counts.size else 0,
        'size_median': float(np.median(counts)) if counts.size else 0.0,
        'size_mean': float(counts.mean()) if counts.size else 0.0,
        'size_p90': float(np.percentile(counts, 90)) if counts.size else 0.0,
        'size_max': int(counts.max()) if counts.size else 0,
        'largest_share': (
            float(counts.max() / valid.size) if counts.size else 0.0),
    }
    print(
        '[ClusterHealth] {name} epoch={epoch} eps={eps:.3f} '
        'total={total} clustered={clustered} noise={noise} '
        'noise_ratio={noise_ratio:.2%} clusters={clusters} '
        'size[min/median/mean/p90/max]='
        '{size_min}/{size_median:.1f}/{size_mean:.1f}/{size_p90:.1f}/{size_max} '
        'largest_share={largest_share:.2%}'.format(**stats))
    return stats


def check_cluster_collapse(name, stats, initial_clusters, ratio):
    """Fail early when cluster count falls far below its epoch-0 value."""
    if ratio <= 0 or initial_clusters <= 0:
        return
    minimum = max(1, int(math.ceil(initial_clusters * ratio)))
    if stats['clusters'] < minimum:
        raise RuntimeError(
            '[ClusterCollapse] {}: epoch={}, clusters={}, epoch0_clusters={}, '
            'required_ratio={:.3f}, minimum={}. Lowering this guard does not '
            'repair clustering; inspect eps and ClusterHealth first.'.format(
                name, stats['epoch'], stats['clusters'], initial_clusters,
                ratio, minimum))

def map_to_agva_path(fname, original_root, agva_root):
    """
    将原始空中图片绝对路径映射到 AGVA 数据根目录。

    Example: map ``<original_root>/relative/image.jpg`` to
    ``<agva_root>/relative/image.jpg``. Offline AGVA is disabled in this
    CARGO baseline release.
    """
    relative_path = osp.relpath(fname, original_root)
    candidate_path = osp.join(agva_root, relative_path)

    if osp.isfile(candidate_path):
        return candidate_path, True

    return fname, False


def maybe_use_agva_path(fname, args, rng=None):
    """
    只针对空中训练图片，根据 agva_ir_prob 决定读取原图或 AGVA 图。

    参数:
        rng: 独立的随机数生成器（推荐），默认为 random 模块全局状态

    返回：
        (selected_path, used_agva, fallback_missing)
    """
    if not args.use_agva_ir:
        return fname, False, False

    # 只允许空中训练图进入 AGVA 路径映射
    if 'aerial_modify' not in fname:
        return fname, False, False

    if rng is None:
        rng = random

    if rng.random() >= args.agva_ir_prob:
        return fname, False, False

    selected_path, found = map_to_agva_path(
        fname,
        args.agva_orig_root,
        args.agva_ir_root,
    )

    if found:
        return selected_path, True, False

    # AGVA 文件不存在时安全回退原图，不能让训练中断
    return fname, False, True





class channel_select(object):
    def __init__(self,channel=0):
        self.channel = channel

    def __call__(self, img):
        if self.channel == 3:
            img_gray = img.convert('L')
            np_img = np.array(img_gray, dtype=np.uint8)
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL=Image.fromarray(img_aug, 'RGB')
        else:
            np_img = np.array(img, dtype=np.uint8)
            np_img = np_img[:,:,self.channel]
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL=Image.fromarray(img_aug, 'RGB')
        return img_PIL



def build_ir_train_transform(args, height, width, normalizer):
    """Ordinary RGB augmentation for CARGO aerial images.

    ChannelAug.py is intentionally not imported or used in this baseline.
    """
    return T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5),
    ])

def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):


    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(
            train_set, root=dataset.images_dir, transform=train_transformer,
            image_process=None),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):



    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    model_ema = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model_ema.cuda()
    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)
    return model, model_ema




class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
def extract_gall_feat(model,gall_loader,ngall):
    pool_dim=2048
    net = model
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input,input, 2)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 2)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc
    
def extract_query_feat(model,query_loader,nquery):
    pool_dim=2048
    net = model
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input, input,1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc


def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    if modal=='visible':
        input_data_path = osp.join(img_dir, 'idx/test_visible_{}.txt'.format(trial))
    elif modal=='thermal':
        input_data_path = osp.join(img_dir, 'idx/test_thermal_{}.txt'.format(trial))
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        # idx files contain relative paths: bounding_box_test_ground/xxx.jpg label
        file_image = [osp.join(img_dir, s.split(' ')[0]) for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)
def eval_regdb(distmat, q_pids, g_pids, max_rank = 20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    
    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP

def associated_analysis_for_all(all_origin, num_ground_samples, log_dir):
    """Report which ALL clusters contain both CARGO view domains.

    ``all_origin`` follows the same ordering as ``features_all``: all ground
    samples first, followed by all aerial samples.  Using that explicit order
    avoids inheriting the original LAG/AG-ReID implementation's assumptions
    about directory names such as ``ground_modify`` and ``aerial_modify``.
    """
    del log_dir  # Kept in the signature for compatibility with the caller.
    all_label_set = sorted(label for label in set(all_origin) if label != -1)
    associate = 0
    flag_ir_list = collections.defaultdict(int)
    flag_rgb_list = collections.defaultdict(int)

    for label in all_label_set:
        indexes = np.flatnonzero(all_origin == label)
        has_ground = bool(np.any(indexes < num_ground_samples))
        has_aerial = bool(np.any(indexes >= num_ground_samples))
        flag_rgb_list[int(label)] = int(has_ground)
        flag_ir_list[int(label)] = int(has_aerial)
        if has_ground and has_aerial:
            associate += 1

    associate_rate = associate / len(all_label_set) if all_label_set else 0.0
    print('associate rate', associate_rate)
    return flag_ir_list, flag_rgb_list

def get_experiment_output_directory(logs_root, log_name, trial):
    name = str(log_name).strip()
    if not name:
        raise ValueError('experiment log name cannot be empty')
    return osp.normpath(osp.join(logs_root, name, str(trial)))


def check_experiment_output_directory(output_dir, allow_overwrite=False):
    if allow_overwrite or not osp.isdir(output_dir):
        return []
    conflicts = []
    for filename in sorted(os.listdir(output_dir)):
        path = osp.join(output_dir, filename)
        if not osp.isfile(path):
            continue
        lower = filename.lower()
        if (lower in ('checkpoint.pth.tar', 'model_best.pth.tar') or
                lower.endswith('.log') or lower.endswith('log.txt')):
            conflicts.append(filename)
    if conflicts:
        raise FileExistsError(
            'Output directory already contains experiment results: {}; '
            'conflicts: {}. Use --allow-log-overwrite only when intentional.'
            .format(output_dir, ', '.join(conflicts)))
    return conflicts


def main():
    args = parser.parse_args()
    # Keep --eps as a backwards-compatible common default while allowing the
    # two CARGO view domains to use independently diagnosed thresholds.
    if args.aerial_eps is None:
        args.aerial_eps = args.eps
    if args.ground_eps is None:
        args.ground_eps = args.eps
    if not 0.0 <= args.cluster_collapse_ratio <= 1.0:
        parser.error('--cluster-collapse-ratio must be in [0, 1]')
    for name, value in (
            ('aerial', args.aerial_eps), ('ground', args.ground_eps),
            ('ALL', args.all_eps)):
        if value <= 0:
            parser.error('{} DBSCAN eps must be positive'.format(name))
    if not args.skip_evaluation and 'ag' not in args.eval_protocols:
        parser.error('--eval-protocols must include ag because official A<->G '
                     'is the primary CARGO protocol')
    print("========== CARGO PCLHD Baseline ==========")
    print("Dataset root:", osp.abspath(osp.expanduser(args.data_dir)))
    print("Stage 1 memory: CM")
    print("Stage 2 memory: CMhard")
    print("Backbone:", args.arch)
    print("Stable GeM:", args.arch == 'agw_cargo_stable')
    print("DBSCAN eps (aerial/ground/ALL): {:.3f}/{:.3f}/{:.3f}".format(
        args.aerial_eps, args.ground_eps, args.all_eps))
    print("Cluster collapse guard ratio:", args.cluster_collapse_ratio)
    print("Stage 2 hard loss: True")
    print("ChannelAug.py imported: False")
    print("ChannelAdapGray: False")
    print("ChannelExchange: False")
    print("Dynamic AGVA: False")
    print("Three-domain matching: False")
    print("EMA loss included in backward: False")
    print("EMA encoder update enabled: True")
    print("ALL-memory second optimization: True")
    print("Checkpoint policy: fixed-last")
    print("Evaluation protocols:", ', '.join(args.eval_protocols))
    print("Skip evaluation:", args.skip_evaluation)
    print("Smoke max IDs:", args.smoke_max_ids)
    print("==========================================")
    args.logs_root = osp.abspath(osp.expanduser(args.logs_dir))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    log_s1_name = args.stage1_log_name
    log_s2_name = args.stage2_log_name
    stage1_dir = get_experiment_output_directory(
        args.logs_root, log_s1_name, args.trial)
    stage2_dir = get_experiment_output_directory(
        args.logs_root, log_s2_name, args.trial)

    if args.stage2_only and not args.stage1_checkpoint:
        parser.error('--stage2-only requires --stage1-checkpoint')

    if args.stage1_only:
        check_experiment_output_directory(
            stage1_dir, args.allow_log_overwrite)
        print('Run mode: Stage 1 only')
        main_worker_stage1(args, log_s1_name)
        return

    if args.stage2_only:
        check_experiment_output_directory(
            stage2_dir, args.allow_log_overwrite)
        print('Run mode: Stage 2 only')
        print('Stage 1 checkpoint: {}'.format(args.stage1_checkpoint))
        main_worker_stage2(
            args, log_s1_name, log_s2_name,
            stage1_checkpoint=args.stage1_checkpoint)
        return

    check_experiment_output_directory(stage1_dir, args.allow_log_overwrite)
    check_experiment_output_directory(stage2_dir, args.allow_log_overwrite)
    print('Run mode: full Stage 1 -> Stage 2')
    main_worker_stage1(args, log_s1_name)
    if isinstance(sys.stdout, Logger):
        stage1_logger = sys.stdout
        sys.stdout = stage1_logger.console
        stage1_logger.close()
    main_worker_stage2(args, log_s1_name, log_s2_name)


def main_worker_stage1(args,log_s1_name):
    logs_dir_root = osp.join(args.logs_root, log_s1_name)
    data_dir = args.data_dir
    trial = args.trial
    # global start_epoch, best_mAP
    # Stage 1 remains the ordinary CM initialization.  CMhard is enabled
    # only in Stage 2 for the requested CARGO baseline.
    args.memorybank = 'CM'
    start_epoch =0
    best_mAP =0
    best_R1 =0
    args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    # Reuse the two AGW input branches as aerial and ground RGB branches.
    dataset_ir = get_data('cargo_aerial', args.data_dir, trial=trial)
    dataset_rgb = get_data('cargo_ground', args.data_dir, trial=trial)
    apply_smoke_subset(dataset_ir, args.smoke_max_ids)
    apply_smoke_subset(dataset_rgb, args.smoke_max_ids)


    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
    # Create model
    model, _ = create_model(args)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = ClusterContrastTrainer_DCL(
        model, debug_nonfinite=args.debug_nonfinite)

    initial_num_cluster_ir = None
    initial_num_cluster_rgb = None
    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = args.aerial_eps
                print('Aerial clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = args.ground_eps
                print('Ground clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled ground data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             args.test_batch, args.workers,
                                             testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            check_feature_rows_finite(
                'raw_cluster_features_rgb', features_rgb,
                sorted(dataset_rgb.train), epoch,
                getattr(args, 'debug_nonfinite', False))

            
            print('==> Create pseudo labels for unlabeled aerial data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             args.test_batch, args.workers,
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            check_feature_rows_finite(
                'raw_cluster_features_ir', features_ir,
                sorted(dataset_ir.train), epoch,
                getattr(args, 'debug_nonfinite', False))


            # Adaptive clustering for IR: if no clusters found, increase eps gradually
            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,search_option=3)
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
                num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
                if num_cluster_ir > 0:
                    break
                cluster_ir.eps += 0.05
                print('IR: no clusters found, increasing eps to {:.3f} and re-clustering...'.format(cluster_ir.eps))
                attempt += 1
            if num_cluster_ir == 0:
                print('WARNING: IR clustering failed after {} attempts, assigning all samples to cluster 0'.format(max_attempts))
                pseudo_labels_ir = np.zeros(len(features_ir), dtype=np.int32)
                num_cluster_ir = 1

            # Adaptive clustering for RGB
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)
            cluster_rgb.eps = getattr(cluster_rgb, 'eps', 0.3)
            attempt = 0
            while attempt < max_attempts:
                pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
                num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
                if num_cluster_rgb > 0:
                    break
                cluster_rgb.eps += 0.05
                print('RGB: no clusters found, increasing eps to {:.3f} and re-clustering...'.format(cluster_rgb.eps))
                attempt += 1
            if num_cluster_rgb == 0:
                print('WARNING: RGB clustering failed after {} attempts, assigning all samples to cluster 0'.format(max_attempts))
                pseudo_labels_rgb = np.zeros(len(features_rgb), dtype=np.int32)
                num_cluster_rgb = 1

            stats_ir = summarize_clustering(
                'aerial', pseudo_labels_ir, epoch, cluster_ir.eps)
            stats_rgb = summarize_clustering(
                'ground', pseudo_labels_rgb, epoch, cluster_rgb.eps)
            if epoch == 0:
                initial_num_cluster_ir = stats_ir['clusters']
                initial_num_cluster_rgb = stats_rgb['clusters']
            else:
                check_cluster_collapse(
                    'aerial', stats_ir, initial_num_cluster_ir,
                    args.cluster_collapse_ratio)
                check_cluster_collapse(
                    'ground', stats_rgb, initial_num_cluster_rgb,
                    args.cluster_collapse_ratio)
            if args.num_instances > 0:
                min_clusters = max(
                    1, (args.batch_size + args.num_instances - 1)
                    // args.num_instances)
            else:
                min_clusters = 1
            if num_cluster_ir < min_clusters:
                raise RuntimeError(
                    'IR clustering collapsed: clusters={}, required={}'
                    .format(num_cluster_ir, min_clusters))
            if num_cluster_rgb < min_clusters:
                raise RuntimeError(
                    'RGB clustering collapsed: clusters={}, required={}'
                    .format(num_cluster_rgb, min_clusters))

            del rerank_dist_rgb
            del rerank_dist_ir

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            if len(centers) == 0:
                return torch.empty(0, features.size(1), device=features.device)

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        check_feature_rows_finite(
            'cluster_centers_ir', cluster_features_ir, None, epoch,
            getattr(args, 'debug_nonfinite', False))
        check_feature_rows_finite(
            'cluster_centers_rgb', cluster_features_rgb, None, epoch,
            getattr(args, 'debug_nonfinite', False))
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                                  momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                  num_instances=args.num_instances).cuda()
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                                   momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                   num_instances=args.num_instances).cuda()
        if args.memorybank == 'CM':
            memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        elif args.memorybank == 'CMhybrid':
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb

        

        # ========== Stage1 AGVA: 空中 loss 训练数据集（仅 loss-loader 使用 AGVA 图片） ==========
        pseudo_labeled_dataset_ir = []
        ir_label=[]

        stage1_agva_count = 0
        stage1_original_count = 0
        stage1_fallback_count = 0
        agva_rng = random.Random(args.seed + epoch)  # 独立 AGVA RNG，不影响 baseline sampler
        _agva_map_printed = 0  # 每个 epoch 最多打印 3 个路径映射示例

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                train_fname, used_agva, fallback_missing = maybe_use_agva_path(
                    fname, args, agva_rng
                )

                if used_agva and _agva_map_printed < 3:
                    print("[AGVA MAP]")
                    print("original: {}".format(fname))
                    print("selected: {}".format(train_fname))
                    _agva_map_printed += 1

                pseudo_labeled_dataset_ir.append(
                    (train_fname, label.item(), cid)
                )
                ir_label.append(label.item())

                if used_agva:
                    stage1_agva_count += 1
                else:
                    stage1_original_count += 1

                if fallback_missing:
                    stage1_fallback_count += 1

        print(
            "[Stage1 AGVA] augmented={}, original={}, missing_fallback={}".format(
                stage1_agva_count,
                stage1_original_count,
                stage1_fallback_count,
            )
        )
        print('==> Statistics for aerial epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())

        print('==> Statistics for ground epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        ########################
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5)
        ])
        
        train_transformer_rgb1 = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5)
        ])

        transform_thermal = build_ir_train_transform(args, height, width, normalizer)

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        args.batch_size//2, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir))

        should_evaluate = (not args.skip_evaluation and
                           ((epoch + 1) % args.eval_step == 0
                            or (epoch + 1) == args.epochs))
        if should_evaluate:
            results = evaluate_cargo_protocols(
                model, data_dir, height=args.height, width=args.width,
                batch_size=args.test_batch, workers=args.workers,
                protocols=args.eval_protocols)
            primary = results['ag']
            cmc, mAP, mINP = (
                primary['cmc'], primary['mAP'], primary['mINP'])
            print('\n * Finished Stage 1 epoch {:3d}   official A<->G '
                  'R1: {:5.1%}  mAP: {:5.1%}\n'.format(
                      epoch, cmc[0], mAP))

        # Always overwrite checkpoint.pth.tar with the current epoch.  This
        # fixed-last policy prevents formal results from selecting an epoch
        # using CARGO test labels.
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'checkpoint_policy': 'fixed-last',
            'backbone': args.arch,
            'aerial_eps': args.aerial_eps,
            'ground_eps': args.ground_eps,
            'all_eps': args.all_eps,
            'stage2_memorybank': 'CMhard',
            'channel_augmentation': False,
            'dynamic_agva': False,
            'three_domain': False,
        }, False, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


def main_worker_stage2(args,log_s1_name,log_s2_name,stage1_checkpoint=None):
    logs_dir_root = osp.join(args.logs_root, log_s2_name)
    trial = args.trial
    start_epoch =0
    best_mAP =0
    best_R1 = 0
    args.memorybank = 'CMhard'
    data_dir = args.data_dir
    args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('cargo_aerial', args.data_dir, trial=trial)
    dataset_rgb = get_data('cargo_ground', args.data_dir, trial=trial)
    apply_smoke_subset(dataset_ir, args.smoke_max_ids)
    apply_smoke_subset(dataset_rgb, args.smoke_max_ids)

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
    # Create model
    model, model_ema = create_model(args)
    if stage1_checkpoint:
        checkpoint_path = osp.abspath(osp.expanduser(stage1_checkpoint))
    else:
        checkpoint_path = osp.join(
            args.logs_root, log_s1_name, str(trial), 'checkpoint.pth.tar')
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(
            'Stage 1 checkpoint does not exist: {}'.format(checkpoint_path))
    checkpoint = load_checkpoint(checkpoint_path)
    print('Stage 1 checkpoint loaded from: {}'.format(checkpoint_path))

    model.load_state_dict(checkpoint['state_dict'])
    model_ema.load_state_dict(checkpoint['state_dict'])
    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # Trainer
    trainer = ClusterContrastTrainer_PCLMP_NoEMALoss(model, model_ema)

    initial_num_cluster_ir = None
    initial_num_cluster_rgb = None
    initial_num_cluster_all = None
    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = args.aerial_eps
                print('Aerial clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = args.ground_eps
                print('Ground clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                all_eps = args.all_eps
                print('All Clustering criterion: eps: {:.3f}'.format(all_eps))
                cluster_all = DBSCAN(eps=all_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled ground data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             args.test_batch, args.workers,
                                             testset=sorted(dataset_rgb.train))
            features_rgb_ema, _ = extract_features(model_ema, cluster_loader_rgb, print_freq=50, mode=1)
            features_rgb_ema = torch.cat([features_rgb_ema[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            check_feature_rows_finite(
                'stage2_raw_features_rgb', features_rgb,
                sorted(dataset_rgb.train), epoch, args.debug_nonfinite)
            check_feature_rows_finite(
                'stage2_raw_features_rgb_ema', features_rgb_ema,
                sorted(dataset_rgb.train), epoch, args.debug_nonfinite)

            
            print('==> Create pseudo labels for unlabeled aerial data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             args.test_batch, args.workers,
                                             testset=sorted(dataset_ir.train))
            features_ir_ema, _ = extract_features(model_ema, cluster_loader_ir, print_freq=50, mode=2)
            features_ir_ema = torch.cat([features_ir_ema[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            check_feature_rows_finite(
                'stage2_raw_features_ir', features_ir,
                sorted(dataset_ir.train), epoch, args.debug_nonfinite)
            check_feature_rows_finite(
                'stage2_raw_features_ir_ema', features_ir_ema,
                sorted(dataset_ir.train), epoch, args.debug_nonfinite)

            print('==> Create pseudo labels for unlabeled ALL data')
            features_all = torch.cat([features_rgb, features_ir], dim=0)
            
            # Adaptive clustering for IR
            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,search_option=3)
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
                num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
                if num_cluster_ir > 0:
                    break
                cluster_ir.eps += 0.05
                print('Stage2 IR: no clusters found, increasing eps to {:.3f} and re-clustering...'.format(cluster_ir.eps))
                attempt += 1
            if num_cluster_ir == 0:
                print('WARNING: Stage2 IR clustering failed, assigning all samples to cluster 0')
                pseudo_labels_ir = np.zeros(len(features_ir), dtype=np.int32)
                num_cluster_ir = 1

            # Adaptive clustering for RGB
            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)
            attempt = 0
            while attempt < max_attempts:
                pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
                num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
                if num_cluster_rgb > 0:
                    break
                cluster_rgb.eps += 0.05
                print('Stage2 RGB: no clusters found, increasing eps to {:.3f} and re-clustering...'.format(cluster_rgb.eps))
                attempt += 1
            if num_cluster_rgb == 0:
                print('WARNING: Stage2 RGB clustering failed, assigning all samples to cluster 0')
                pseudo_labels_rgb = np.zeros(len(features_rgb), dtype=np.int32)
                num_cluster_rgb = 1

            # Adaptive clustering for ALL
            rerank_dist_all = compute_modal_invariant_jaccard_distance(features_all, k1=40, k2=32,
                                                                       file=sorted(dataset_rgb.train) + sorted(
                                                                           dataset_ir.train), search_option=3)
            attempt = 0
            while attempt < max_attempts:
                pseudo_labels_all = cluster_all.fit_predict(rerank_dist_all)
                num_cluster_all = len(set(pseudo_labels_all)) - (1 if -1 in pseudo_labels_all else 0)
                if num_cluster_all > 0:
                    break
                cluster_all.eps += 0.05
                print('Stage2 ALL: no clusters found, increasing eps to {:.3f} and re-clustering...'.format(cluster_all.eps))
                attempt += 1
            if num_cluster_all == 0:
                print('WARNING: Stage2 ALL clustering failed, assigning all samples to cluster 0')
                pseudo_labels_all = np.zeros(len(features_all), dtype=np.int32)
                num_cluster_all = 1

            stats_ir = summarize_clustering(
                'aerial', pseudo_labels_ir, epoch, cluster_ir.eps)
            stats_rgb = summarize_clustering(
                'ground', pseudo_labels_rgb, epoch, cluster_rgb.eps)
            stats_all = summarize_clustering(
                'ALL', pseudo_labels_all, epoch, cluster_all.eps)
            if epoch == 0:
                initial_num_cluster_ir = stats_ir['clusters']
                initial_num_cluster_rgb = stats_rgb['clusters']
                initial_num_cluster_all = stats_all['clusters']
            else:
                check_cluster_collapse(
                    'aerial', stats_ir, initial_num_cluster_ir,
                    args.cluster_collapse_ratio)
                check_cluster_collapse(
                    'ground', stats_rgb, initial_num_cluster_rgb,
                    args.cluster_collapse_ratio)
                check_cluster_collapse(
                    'ALL', stats_all, initial_num_cluster_all,
                    args.cluster_collapse_ratio)

            del rerank_dist_rgb
            del rerank_dist_ir
            del rerank_dist_all

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            if len(centers) == 0:
                return torch.empty(0, features.size(1), device=features.device)

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers
        
        # generate new dataset and calculate all cluster centers
        @torch.no_grad()
        def generate_modal_invariant_cluster_features(
                labels, num_cluster_all, features, num_ground_samples):
            centers_IR = collections.defaultdict(list)
            centers_RBG = collections.defaultdict(list)
            centers_IR_mean = collections.defaultdict(list)
            centers_RBG_mean = collections.defaultdict(list)
            # centers_all = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                if i < num_ground_samples:
                    centers_RBG[labels[i]].append(features[i])
                else:
                    centers_IR[labels[i]].append(features[i])
            for i in range(num_cluster_all):
                if centers_RBG[i] != []:
                    centers_RBG_mean[i] = torch.stack(centers_RBG[i], dim=0).mean(0)
                if centers_IR[i] != []:
                    centers_IR_mean[i] = torch.stack(centers_IR[i], dim=0).mean(0)
            centers_all = []
            for i in range(num_cluster_all):
                if centers_RBG_mean[i] == []:
                    centers_all.append(centers_IR_mean[i])
                elif centers_IR_mean[i] == []:
                    centers_all.append(centers_RBG_mean[i])
                else:
                    centers_all.append(torch.mean(torch.stack([centers_RBG_mean[i], centers_IR_mean[i]], dim=0), dim=0))
            centers_all = torch.stack(centers_all, dim=0)

            return centers_all

        # generate instances features
        def generate_random_features(labels, features, num_cluster, num_instances):
            indexes = np.zeros(num_cluster * num_instances)
            for i in range(num_cluster):
                index = [i + k * num_cluster for k in range(num_instances)]
                samples = np.random.choice(np.where(labels == i)[0], num_instances, True)
                indexes[index] = samples
            memory_features = features[indexes]
            return memory_features        

        memory_features_ir = generate_random_features(pseudo_labels_ir, features_ir_ema, num_cluster_ir, args.num_instances)
        memory_features_rgb = generate_random_features(pseudo_labels_rgb, features_rgb_ema, num_cluster_rgb, args.num_instances)
        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        num_ground_samples = len(dataset_rgb.train)
        cluster_features_all = generate_modal_invariant_cluster_features(
            pseudo_labels_all, num_cluster_all, features_all,
            num_ground_samples)
        check_feature_rows_finite(
            'stage2_cluster_centers_ir', cluster_features_ir, None, epoch,
            args.debug_nonfinite)
        check_feature_rows_finite(
            'stage2_cluster_centers_rgb', cluster_features_rgb, None, epoch,
            args.debug_nonfinite)
        check_feature_rows_finite(
            'stage2_cluster_centers_all', cluster_features_all, None, epoch,
            args.debug_nonfinite)
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp,
                                  momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                  num_instances=args.num_instances).cuda()
        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
                                   momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                   num_instances=args.num_instances).cuda()
        memory_all = ClusterMemory(model.module.num_features, num_cluster_all, temp=args.temp,
                                    momentum=args.momentum, mode='CMhybrid', smooth=args.smooth,
                                   num_instances=args.num_instances).cuda()
        if args.memorybank == 'CM':
            memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
            memory_all.features = F.normalize(cluster_features_all, dim=1).cuda()
        elif args.memorybank == 'CMhybrid':
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()
            memory_all.features = F.normalize(cluster_features_all.repeat(2, 1), dim=1).cuda()
        elif args.memorybank == 'CMhard':
            # Cluster proxies
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()
            memory_all.features = F.normalize(cluster_features_all.repeat(2,1), dim=1).cuda()
            # Instance proxies
            memory_ir.features_ema = F.normalize(memory_features_ir, dim=1).cuda()
            memory_rgb.features_ema = F.normalize(memory_features_rgb, dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb
        trainer.memory_all = memory_all

        # ========== Stage2 AGVA: 空中 loss 训练数据集（仅 loss-loader 使用 AGVA 图片） ==========
        pseudo_labeled_dataset_ir = []
        ir_label=[]
        stage2_agva_count = 0
        stage2_original_count = 0
        stage2_fallback_count = 0
        stage2_agva_rng = random.Random(args.seed + epoch)
        stage2_agva_map_printed = 0
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                train_fname, used_agva, fallback_missing = maybe_use_agva_path(
                    fname, args, stage2_agva_rng
                )
                if used_agva and stage2_agva_map_printed < 3:
                    print("[AGVA MAP]")
                    print("original: {}".format(fname))
                    print("selected: {}".format(train_fname))
                    stage2_agva_map_printed += 1
                pseudo_labeled_dataset_ir.append((train_fname, label.item(), cid))
                ir_label.append(label.item())
                if used_agva:
                    stage2_agva_count += 1
                else:
                    stage2_original_count += 1
                if fallback_missing:
                    stage2_fallback_count += 1
        print(
            "[Stage2 IR AGVA] augmented={}, original={}, missing_fallback={}".format(
                stage2_agva_count,
                stage2_original_count,
                stage2_fallback_count,
            )
        )
        print('==> Statistics for aerial epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label=[]
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())

        print('==> Statistics for ground epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        all_label = []
        all_file_name = []
        for i, ((fname, _, cid), label) in enumerate(
                zip(sorted(dataset_rgb.train) + sorted(dataset_ir.train), pseudo_labels_all)):
            if label != -1:
                all_file_name.append(fname)
                all_label.append(label.item())

        flag_ir_list, flag_rgb_list = associated_analysis_for_all(
            pseudo_labels_all, num_ground_samples, args.logs_dir)
        print('==> Statistics for ALL epoch {}: {} clusters'.format(epoch, num_cluster_all))

        all_label = []
        pseudo_labeled_dataset_all_ir = []
        pseudo_labeled_dataset_all_rgb = []
        stage2_all_ir_agva_count = 0
        stage2_all_ir_original_count = 0
        stage2_all_ir_fallback_count = 0
        stage2_all_ir_agva_rng = random.Random(args.seed + epoch)
        stage2_all_ir_agva_map_printed = 0
        for i, ((fname, _, cid), label) in enumerate(
                zip(sorted(dataset_rgb.train) + sorted(dataset_ir.train), pseudo_labels_all)):
            if label != -1:
                all_file_name.append(fname)
                all_label.append(label.item())
            is_ground = i < num_ground_samples
            if (not is_ground and flag_ir_list[label] == 1 and
                    flag_rgb_list[label] == 1):
                train_fname, used_agva, fallback_missing = maybe_use_agva_path(
                    fname, args, stage2_all_ir_agva_rng
                )
                if used_agva and stage2_all_ir_agva_map_printed < 3:
                    print("[AGVA MAP]")
                    print("original: {}".format(fname))
                    print("selected: {}".format(train_fname))
                    stage2_all_ir_agva_map_printed += 1
                pseudo_labeled_dataset_all_ir.append((train_fname, label.item(), cid))
                if used_agva:
                    stage2_all_ir_agva_count += 1
                else:
                    stage2_all_ir_original_count += 1
                if fallback_missing:
                    stage2_all_ir_fallback_count += 1
            elif (is_ground and flag_ir_list[label] == 1 and
                  flag_rgb_list[label] == 1):
                pseudo_labeled_dataset_all_rgb.append((fname, label.item(), cid))
        print(
            "[Stage2 ALL-IR AGVA] augmented={}, original={}, missing_fallback={}".format(
                stage2_all_ir_agva_count,
                stage2_all_ir_original_count,
                stage2_all_ir_fallback_count,
            )
        )

        ######################## PGM
        print("Start Bipartite Graph Matching")
        i2r = {}
        r2i = {}
        R = []
        bgm = False
        if num_cluster_rgb >= num_cluster_ir:
            # clusternorm
            cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)
            cluster_features_ir = F.normalize(cluster_features_ir, dim=1)
            # [-1, 1] torch.mm(cluster_features_rgb, cluster_features_ir.T) #CostMatrix
            similarity = ((torch.mm(cluster_features_rgb, cluster_features_ir.T)) / 1).exp().cpu()  # .exp().cpu()
            dis_similarity = (1 / (similarity))
            cost = dis_similarity / 1
            tmp = torch.zeros(dis_similarity.shape[0], dis_similarity.shape[0] - dis_similarity.shape[1])
            cost = (torch.cat((cost, tmp), 1))
            unmatched_row = []
            row_ind, col_ind = linear_sum_assignment(cost)
            for idx, item in enumerate(row_ind):
                if col_ind[idx] < similarity.shape[1]:
                    R.append((row_ind[idx], col_ind[idx]))
                    r2i[row_ind[idx]] = col_ind[idx]
                    i2r[col_ind[idx]] = row_ind[idx]
                else:
                    unmatched_row.append(row_ind[idx])
            if bgm is False:
                unmatched_cost = cost[unmatched_row][:, :dis_similarity.shape[1]]
                unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
                for idx, item in enumerate(unmatched_row_ind):
                    R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                    r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
            del cluster_features_ir, cluster_features_rgb

        print("Finish Bipartite Graph Matching")
        ####################################
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5)
        ])
        
        train_transformer_rgb1 = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5)
        ])

        transform_thermal = build_ir_train_transform(args, height, width, normalizer)

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
        # A small smoke subset can yield fewer than batch_size/num_instances
        # cross-domain ALL pseudo identities.  With drop_last=True this would
        # otherwise create an empty DataLoader.  Keep the configured batch
        # size whenever enough associated identities exist, and only shrink
        # the ALL-memory batch to the available PK samples when necessary.
        all_identity_count = len({item[1] for item in pseudo_labeled_dataset_all_ir})
        if all_identity_count == 0:
            raise RuntimeError(
                'Stage2 ALL clustering produced no cross-domain pseudo '
                'identity; cannot run the ALL-memory optimization.')
        all_batch_size = min(
            args.batch_size, all_identity_count * args.num_instances)
        print('[Stage2 ALL loader] cross-domain identities={}, batch_size={} '
              '(configured={})'.format(
                  all_identity_count, all_batch_size, args.batch_size))

        train_loader_all_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                                  all_batch_size, args.workers, args.num_instances, iters,
                                                  trainset=pseudo_labeled_dataset_all_ir, no_cam=args.no_cam,
                                                  train_transformer=transform_thermal)

        train_loader_all_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                                      all_batch_size, args.workers, args.num_instances, iters,
                                                      trainset=pseudo_labeled_dataset_all_rgb, no_cam=args.no_cam,
                                                      train_transformer=train_transformer_rgb,
                                                      train_transformer1=train_transformer_rgb1)        

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()
        train_loader_all_ir.new_epoch()
        train_loader_all_rgb.new_epoch()        


        trainer.train(epoch, train_loader_ir, train_loader_rgb, train_loader_all_ir, train_loader_all_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r, r2i=r2i)

        should_evaluate = (not args.skip_evaluation and
                           ((epoch + 1) % args.eval_step == 0
                            or (epoch + 1) == args.epochs))
        if should_evaluate:
            results = evaluate_cargo_protocols(
                model_ema, data_dir, height=args.height, width=args.width,
                batch_size=args.test_batch, workers=args.workers,
                protocols=args.eval_protocols)
            primary = results['ag']
            cmc, mAP, mINP = (
                primary['cmc'], primary['mAP'], primary['mINP'])
            print('\n * Finished Stage 2 epoch {:3d}   official A<->G '
                  'R1: {:5.1%}  mAP: {:5.1%}\n'.format(
                      epoch, cmc[0], mAP))

        save_checkpoint({
            'state_dict': model_ema.state_dict(),
            'epoch': epoch + 1,
            'checkpoint_policy': 'fixed-last',
            'backbone': args.arch,
            'aerial_eps': args.aerial_eps,
            'ground_eps': args.ground_eps,
            'all_eps': args.all_eps,
            'stage2_memorybank': 'CMhard',
            'channel_augmentation': False,
            'dynamic_agva': False,
            'three_domain': False,
            'ema_loss_in_backward': False,
            'ema_encoder_update': True,
            'all_memory_second_step': True,
        }, False, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="CARGO PCLHD-CMhard channel-off baseline engine")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cargo_aerial',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--test-batch', type=int, default=64, help="test batch size for feature extraction")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=288, help="input height")
    parser.add_argument('--width', type=int, default=144, help="input width")
    parser.add_argument('--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.5,
                        help="common DBSCAN threshold used when a domain-specific threshold is omitted")
    parser.add_argument('--aerial-eps', type=float, default=None,
                        help='CARGO aerial-domain DBSCAN threshold; defaults to --eps')
    parser.add_argument('--ground-eps', type=float, default=None,
                        help='CARGO ground-domain DBSCAN threshold; defaults to --eps')
    parser.add_argument('--all-eps', type=float, default=0.5,
                        help='DBSCAN threshold for Stage 2 ALL clustering')
    parser.add_argument(
        '--cluster-collapse-ratio', type=float, default=0.0,
        help='fail if a later cluster count falls below this fraction of its epoch-0 count; 0 disables the guard')
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='agw_cargo_stable',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('-mb', '--memorybank', type=str, default='CM',
                    choices=['CM', 'CMhard', 'CMhybrid'])
    parser.add_argument('--smooth', type=float, default=0, help="label smoothing")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument(
        '--debug-nonfinite', action='store_true',
        help='stop at the first non-finite Stage 1 input, feature, memory, loss, gradient, or parameter')
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument(
        '--eval-protocols', nargs='+',
        choices=['all', 'aa', 'gg', 'ag', 'g2ag'],
        default=['all', 'aa', 'gg', 'ag', 'g2ag'],
        help='CARGO protocols evaluated at --eval-step intervals')
    parser.add_argument(
        '--skip-evaluation', action='store_true',
        help='skip query/gallery evaluation (intended only for smoke tests)')
    parser.add_argument(
        '--smoke-max-ids', type=int, default=0,
        help='keep only the first N parsed training IDs; 0 uses full data')
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='./datasets/CARGO')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--stage1-log-name', type=str,
                        default='cargo_pclhd_channel_off_s1',
                        help='Stage 1 log directory name under --logs-dir')
    parser.add_argument('--stage2-log-name', type=str,
                        default='cargo_pclhd_cmhard_channel_off_s2',
                        help='Stage 2 log directory name under --logs-dir')
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        '--stage1-only', action='store_true', help='run only Stage 1')
    stage_group.add_argument(
        '--stage2-only', action='store_true',
        help='run only Stage 2 from --stage1-checkpoint')
    parser.add_argument(
        '--stage1-checkpoint', type=str, default=None,
        help='explicit Stage 1 checkpoint.pth.tar for Stage 2-only mode')
    parser.add_argument(
        '--allow-log-overwrite', action='store_true')
    # Offline AGVA is intentionally outside this reproducibility release.
    parser.set_defaults(
        use_agva_ir=False,
        agva_orig_root='',
        agva_ir_root='',
        agva_ir_prob=0.0)

    main()
