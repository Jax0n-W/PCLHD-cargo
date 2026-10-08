from __future__ import absolute_import, print_function

import glob
import os.path as osp
import re

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from clustercontrast.utils.data import transforms as T


_PATTERN = re.compile(
    r"^Cam(?P<cam>\d+)_(?P<time>day|night)_(?P<pid>\d+)_(?P<index>\d+)\.jpg$",
    re.IGNORECASE,
)
PROTOCOLS = ("all", "aa", "gg", "ag", "g2ag")


def _read_split(data_root, split):
    samples = []
    for cam in range(1, 14):
        for path in sorted(glob.glob(osp.join(
                data_root, split, "Cam{}".format(cam), "*.jpg"))):
            match = _PATTERN.match(osp.basename(path))
            if match is None:
                raise ValueError("Invalid CARGO image name: {}".format(path))
            parsed_cam = int(match.group("cam"))
            pid = int(match.group("pid"))
            domain = "aerial" if parsed_cam <= 5 else "ground"
            samples.append({
                "path": path,
                "pid": pid,
                "camid": parsed_cam - 1,
                "domain": domain,
            })
    return samples


class _ImageDataset(Dataset):
    def __init__(self, indexed_samples, transform):
        self.indexed_samples = indexed_samples
        self.transform = transform

    def __len__(self):
        return len(self.indexed_samples)

    def __getitem__(self, index):
        output_index, sample = self.indexed_samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        return self.transform(image), output_index, sample["path"]


def _flip_lr(images):
    return torch.flip(images, dims=(3,))


def _check_finite_features(tensor, stage, domain, paths):
    finite_rows = torch.isfinite(tensor).all(dim=1)
    if bool(finite_rows.all()):
        return
    bad = torch.nonzero(~finite_rows, as_tuple=False).flatten().tolist()
    details = []
    for index in bad[:8]:
        row = tensor[index]
        finite = row[torch.isfinite(row)]
        max_abs = float(finite.abs().max().item()) if finite.numel() else float('nan')
        details.append(
            'path={}; nan={}; posinf={}; neginf={}; finite_max_abs={:.6g}'.format(
                paths[index], int(torch.isnan(row).sum().item()),
                int(torch.isposinf(row).sum().item()),
                int(torch.isneginf(row).sum().item()), max_abs))
    raise RuntimeError(
        'Non-finite CARGO evaluation features: stage={}, domain={}, '
        'bad_rows={}; {}'.format(stage, domain, len(bad), ' | '.join(details)))


@torch.no_grad()
def _extract_split(model, samples, height, width, batch_size, workers):
    transform = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    features = torch.empty((len(samples), 2048), dtype=torch.float32)
    model.eval()
    for domain, modal in (("aerial", 2), ("ground", 1)):
        indexed = [
            (index, sample) for index, sample in enumerate(samples)
            if sample["domain"] == domain
        ]
        loader = DataLoader(
            _ImageDataset(indexed, transform), batch_size=batch_size,
            shuffle=False, num_workers=workers, pin_memory=True)
        for images, indexes, paths in loader:
            images = images.cuda(non_blocking=True)
            output = model(images, images, modal)
            flipped = _flip_lr(images)
            output_flip = model(flipped, flipped, modal)
            _check_finite_features(output, 'original', domain, paths)
            _check_finite_features(output_flip, 'horizontal_flip', domain, paths)
            fused = (output + output_flip) / 2.0
            _check_finite_features(fused, 'pre_normalization_fusion', domain, paths)
            output = F.normalize(fused, dim=1)
            _check_finite_features(output, 'normalized_fusion', domain, paths)
            features[indexes.long()] = output.cpu()
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("Non-finite CARGO evaluation features detected")
    return features.numpy()


def _protocol_samples(query, gallery, protocol):
    if protocol == "all":
        return query, gallery, False
    if protocol == "aa":
        return ([x for x in query if x["domain"] == "aerial"],
                [x for x in gallery if x["domain"] == "aerial"], False)
    if protocol == "gg":
        return ([x for x in query if x["domain"] == "ground"],
                [x for x in gallery if x["domain"] == "ground"], False)
    if protocol == "ag":
        # Official Protocol 4: collapse physical cameras to the two view
        # domains, then apply the ordinary same-pid/same-cam junk filter.
        return query, gallery, True
    if protocol == "g2ag":
        # Custom protocol: ground query against the complete aerial+ground
        # gallery.  Physical camera IDs must be preserved here.
        return ([x for x in query if x["domain"] == "ground"],
                gallery, False)
    raise KeyError("Unknown CARGO protocol: {}".format(protocol))


def _subset_features(all_samples, all_features, selected):
    index = {sample["path"]: i for i, sample in enumerate(all_samples)}
    return np.stack([all_features[index[x["path"]]] for x in selected])


def _metrics(query_features, gallery_features, query, gallery,
             use_domain_camids=False, max_rank=20):
    similarity = np.matmul(query_features, gallery_features.T)
    order = np.argsort(-similarity, axis=1)
    gallery_pids = np.asarray([x["pid"] for x in gallery])
    if use_domain_camids:
        gallery_cams = np.asarray([
            0 if x["domain"] == "aerial" else 1 for x in gallery
        ])
    else:
        gallery_cams = np.asarray([x["camid"] for x in gallery])

    cmc_sum = np.zeros(max_rank, dtype=np.float64)
    aps = []
    inps = []
    valid_queries = 0
    for q_index, q_sample in enumerate(query):
        ranked = order[q_index]
        q_pid = q_sample["pid"]
        q_cam = (0 if q_sample["domain"] == "aerial" else 1) \
            if use_domain_camids else q_sample["camid"]
        remove = ((gallery_pids[ranked] == q_pid)
                  & (gallery_cams[ranked] == q_cam))
        kept = ranked[~remove]
        matches = gallery_pids[kept] == q_pid
        positive_positions = np.flatnonzero(matches)
        if positive_positions.size == 0:
            continue

        first = int(positive_positions[0])
        if first < max_rank:
            cmc_sum[first:] += 1.0
        cumulative = np.cumsum(matches, dtype=np.float64)
        precisions = cumulative[positive_positions] / (positive_positions + 1.0)
        aps.append(float(precisions.mean()))
        last_rank = int(positive_positions[-1]) + 1
        inps.append(float(positive_positions.size) / float(last_rank))
        valid_queries += 1

    if valid_queries == 0:
        raise RuntimeError("No valid query for the selected CARGO protocol")
    return {
        "cmc": cmc_sum / valid_queries,
        "mAP": float(np.mean(aps)),
        "mINP": float(np.mean(inps)),
        "valid_queries": valid_queries,
        "skipped_queries": len(query) - valid_queries,
        "query_images": len(query),
        "gallery_images": len(gallery),
    }


def evaluate_cargo_protocols(model, data_root, height=288, width=144,
                             batch_size=64, workers=8,
                             protocols=PROTOCOLS):
    protocols = tuple(protocols)
    unknown = sorted(set(protocols) - set(PROTOCOLS))
    if unknown:
        raise ValueError("Unknown CARGO protocols: {}".format(unknown))
    query = _read_split(data_root, "query")
    gallery = _read_split(data_root, "gallery")
    query_features = _extract_split(
        model, query, height, width, batch_size, workers)
    gallery_features = _extract_split(
        model, gallery, height, width, batch_size, workers)

    results = {}
    print("========== CARGO Evaluation ==========")
    for protocol in protocols:
        q_samples, g_samples, domain_camids = _protocol_samples(
            query, gallery, protocol)
        q_features = _subset_features(query, query_features, q_samples)
        g_features = _subset_features(gallery, gallery_features, g_samples)
        result = _metrics(
            q_features, g_features, q_samples, g_samples,
            use_domain_camids=domain_camids)
        results[protocol] = result
        cmc = result["cmc"]
        print("[{}] query={} gallery={} valid={} skipped={}".format(
            protocol.upper(), result["query_images"],
            result["gallery_images"], result["valid_queries"],
            result["skipped_queries"]))
        print("Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%} | "
              "Rank-20: {:.2%} | mAP: {:.2%} | mINP: {:.2%}".format(
                  cmc[0], cmc[4], cmc[9], cmc[19],
                  result["mAP"], result["mINP"]))
    print("======================================")
    return results
