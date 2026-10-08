from __future__ import absolute_import, print_function

import argparse
from collections import Counter
import glob
import json
import os.path as osp
import re


PATTERN = re.compile(
    r"^Cam(?P<cam>\d+)_(?P<time>day|night)_(?P<pid>\d+)_(?P<index>\d+)\.jpg$",
    re.IGNORECASE,
)


def inspect(root):
    report = {"root": osp.abspath(root), "splits": {}}
    split_samples = {}
    for split in ("train", "query", "gallery"):
        files = sorted(glob.glob(osp.join(root, split, "Cam*", "*.jpg")))
        samples = []
        bad = []
        for path in files:
            match = PATTERN.match(osp.basename(path))
            if match is None:
                bad.append(path)
                continue
            samples.append({
                "path": path,
                "cam": int(match.group("cam")),
                "time": match.group("time").lower(),
                "pid": int(match.group("pid")),
            })
        split_samples[split] = samples
        report["splits"][split] = {
            "images": len(samples),
            "ids": len({x["pid"] for x in samples}),
            "bad_names": bad,
            "by_camera": dict(sorted(Counter(
                x["cam"] for x in samples).items())),
            "by_time": dict(sorted(Counter(
                x["time"] for x in samples).items())),
            "aerial_images": sum(x["cam"] <= 5 for x in samples),
            "ground_images": sum(x["cam"] >= 6 for x in samples),
        }

    train_ids = {x["pid"] for x in split_samples["train"]}
    test_ids = ({x["pid"] for x in split_samples["query"]}
                | {x["pid"] for x in split_samples["gallery"]})
    report["train_test_id_overlap"] = sorted(train_ids & test_ids)
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit a CARGO extraction")
    parser.add_argument(
        "--data-dir", default="./datasets/CARGO")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = inspect(osp.abspath(osp.expanduser(args.data_dir)))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
