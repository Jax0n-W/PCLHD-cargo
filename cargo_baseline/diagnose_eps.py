"""Label-free DBSCAN threshold diagnosis for CARGO.

Features and Jaccard distances are computed once per domain, then reused for
all requested epsilon values.  Training identities are never read by this
tool; the report contains only cluster-size and noise statistics.
"""

from __future__ import absolute_import, print_function

import argparse
import gc
import json
import os
import os.path as osp
import sys

import numpy as np
from sklearn.cluster import DBSCAN
import torch
from torch import nn


HERE = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clustercontrast import datasets, models
from clustercontrast.evaluators import extract_features
from clustercontrast.utils.faiss_rerank import (
    compute_jaccard_distance,
    compute_modal_invariant_jaccard_distance,
)
from clustercontrast.utils.serialization import load_checkpoint
from _train_cargo_engine import get_test_loader, summarize_clustering


def _extract_domain(model, dataset, modal, args, name):
    samples = sorted(dataset.train)
    loader = get_test_loader(
        dataset, args.height, args.width, args.test_batch, args.workers,
        testset=samples)
    feature_dict, _ = extract_features(
        model, loader, print_freq=args.print_freq, mode=modal)
    features = torch.cat(
        [feature_dict[path].unsqueeze(0) for path, _, _ in samples], dim=0)
    if not bool(torch.isfinite(features).all()):
        bad = torch.nonzero(
            ~torch.isfinite(features).all(dim=1), as_tuple=False
        ).flatten().tolist()
        paths = [samples[index][0] for index in bad[:10]]
        raise RuntimeError(
            'Non-finite {} features: rows={}, sample_paths={}'.format(
                name, len(bad), paths))
    return features, samples


def _scan_distance(name, distance, eps_values, min_samples):
    reports = []
    for eps in eps_values:
        labels = DBSCAN(
            eps=eps, min_samples=min_samples, metric='precomputed',
            n_jobs=-1).fit_predict(distance)
        reports.append(summarize_clustering(name, labels, 0, eps))
    return reports


def _load_model(args):
    model = models.create(
        args.arch, num_features=0, norm=True, dropout=0,
        num_classes=0, pooling_type='gem')
    model = nn.DataParallel(model.cuda())
    if args.checkpoint:
        checkpoint = load_checkpoint(osp.abspath(osp.expanduser(args.checkpoint)))
        model.load_state_dict(checkpoint['state_dict'])
        print('Feature checkpoint: {}'.format(args.checkpoint))
        print('Checkpoint epoch: {}'.format(checkpoint.get('epoch', 'unknown')))
    else:
        print('WARNING: no checkpoint supplied; scanning the model initialization only.')
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Scan CARGO DBSCAN eps without retraining')
    parser.add_argument(
        '--data-dir', default='./datasets/CARGO')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--arch', default='agw_cargo_stable',
                        choices=models.names())
    parser.add_argument('--eps-values', type=float, nargs='+',
                        default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument('--all-eps-values', type=float, nargs='+',
                        default=[0.30, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument('--include-all', action='store_true',
                        help='also build the larger modal-invariant ALL distance matrix')
    parser.add_argument('--min-samples', type=int, default=4)
    parser.add_argument('--k1', type=int, default=30)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--all-k1', type=int, default=40)
    parser.add_argument('--all-k2', type=int, default=32)
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--width', type=int, default=144)
    parser.add_argument('--test-batch', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the current AGW feature pipeline')

    model = _load_model(args)
    aerial = datasets.create('cargo_aerial', args.data_dir, trial=1)
    ground = datasets.create('cargo_ground', args.data_dir, trial=1)

    print('==> Extract aerial features once')
    aerial_features, aerial_samples = _extract_domain(
        model, aerial, 2, args, 'aerial')
    print('==> Compute aerial Jaccard distance once')
    aerial_distance = compute_jaccard_distance(
        aerial_features, k1=args.k1, k2=args.k2, search_option=3)
    aerial_report = _scan_distance(
        'aerial', aerial_distance, args.eps_values, args.min_samples)
    del aerial_distance
    gc.collect()

    print('==> Extract ground features once')
    ground_features, ground_samples = _extract_domain(
        model, ground, 1, args, 'ground')
    print('==> Compute ground Jaccard distance once')
    ground_distance = compute_jaccard_distance(
        ground_features, k1=args.k1, k2=args.k2, search_option=3)
    ground_report = _scan_distance(
        'ground', ground_distance, args.eps_values, args.min_samples)
    del ground_distance
    gc.collect()

    all_report = []
    if args.include_all:
        print('==> Compute modal-invariant ALL Jaccard distance once')
        all_features = torch.cat([ground_features, aerial_features], dim=0)
        all_samples = ground_samples + aerial_samples
        all_distance = compute_modal_invariant_jaccard_distance(
            all_features, k1=args.all_k1, k2=args.all_k2,
            file=all_samples, search_option=3)
        all_report = _scan_distance(
            'ALL', all_distance, args.all_eps_values, args.min_samples)
        del all_distance, all_features
        gc.collect()

    report = {
        'data_dir': osp.abspath(osp.expanduser(args.data_dir)),
        'checkpoint': (
            osp.abspath(osp.expanduser(args.checkpoint))
            if args.checkpoint else None),
        'arch': args.arch,
        'identity_labels_used': False,
        'k1': args.k1,
        'k2': args.k2,
        'all_k1': args.all_k1,
        'all_k2': args.all_k2,
        'min_samples': args.min_samples,
        'aerial': aerial_report,
        'ground': ground_report,
        'ALL': all_report,
    }
    output = osp.abspath(osp.expanduser(args.output))
    os.makedirs(osp.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print('Diagnosis report written to: {}'.format(output))


if __name__ == '__main__':
    main()
