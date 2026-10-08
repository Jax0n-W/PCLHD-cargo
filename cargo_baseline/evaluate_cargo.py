from __future__ import absolute_import, print_function

import argparse
import json
import os.path as osp
import sys

import torch
from torch import nn


HERE = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clustercontrast import models
from clustercontrast.utils.serialization import load_checkpoint
from cargo_evaluation import PROTOCOLS, evaluate_cargo_protocols


def main():
    parser = argparse.ArgumentParser(description="Evaluate CARGO protocols")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-dir", default="./datasets/CARGO")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pooling-type", default="gem")
    parser.add_argument("--arch", default="", choices=[""] + models.names(),
                        help="override checkpoint backbone metadata")
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS,
                        default=list(PROTOCOLS))
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    checkpoint = load_checkpoint(osp.abspath(osp.expanduser(args.checkpoint)))
    arch = args.arch or checkpoint.get("backbone", "agw_cargo_stable")
    print("Evaluation backbone: {}".format(arch))
    model = models.create(
        arch, num_features=0, norm=True, dropout=0, num_classes=0,
        pooling_type=args.pooling_type)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    results = evaluate_cargo_protocols(
        model, osp.abspath(osp.expanduser(args.data_dir)),
        height=args.height, width=args.width, batch_size=args.batch_size,
        workers=args.workers, protocols=args.protocols)

    if args.output_json:
        serializable = {}
        for name, result in results.items():
            serializable[name] = dict(result)
            serializable[name]["cmc"] = result["cmc"].tolist()
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
