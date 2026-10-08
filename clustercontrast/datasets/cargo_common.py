from __future__ import absolute_import, print_function

import glob
import os.path as osp
import re

from ..utils.data import BaseImageDataset


_CARGO_PATTERN = re.compile(
    r"^Cam(?P<cam>\d+)_(?P<time>day|night)_(?P<pid>\d+)_(?P<index>\d+)\.jpg$",
    re.IGNORECASE,
)


class CargoDomain(BaseImageDataset):
    """One view domain of CARGO using the official train/query/gallery tree.

    CARGO stores annotations in file names such as
    ``Cam2_day_2519_320.jpg``.  Camera 1--5 are aerial and camera 6--13
    are ground cameras.  The real training identities are parsed only to
    describe the split; the training engine replaces them with DBSCAN labels.
    """

    domain_name = None
    camera_ids = ()

    def __init__(self, root, trial=1, verbose=True, **kwargs):
        super(CargoDomain, self).__init__()
        self.dataset_dir = osp.abspath(osp.expanduser(root))
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CARGO {} loaded, trial={}".format(
                self.domain_name, trial))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        (self.num_train_pids, self.num_train_imgs,
         self.num_train_cams) = self.get_imagedata_info(train)
        (self.num_query_pids, self.num_query_imgs,
         self.num_query_cams) = self.get_imagedata_info(query)
        (self.num_gallery_pids, self.num_gallery_imgs,
         self.num_gallery_cams) = self.get_imagedata_info(gallery)

    @property
    def images_dir(self):
        return self.dataset_dir

    def _check_before_run(self):
        for path in (self.dataset_dir, self.train_dir,
                     self.query_dir, self.gallery_dir):
            if not osp.isdir(path):
                raise RuntimeError("CARGO directory is unavailable: {}".format(path))

    def _process_dir(self, split_dir, relabel=False):
        paths = []
        for camid in self.camera_ids:
            paths.extend(glob.glob(osp.join(
                split_dir, "Cam{}".format(camid), "*.jpg")))
        paths = sorted(paths)

        parsed = []
        pids = set()
        for path in paths:
            match = _CARGO_PATTERN.match(osp.basename(path))
            if match is None:
                raise ValueError("Invalid CARGO image name: {}".format(path))
            pid = int(match.group("pid"))
            camid = int(match.group("cam")) - 1
            parsed.append((path, pid, camid))
            pids.add(pid)

        pid2label = {
            pid: label for label, pid in enumerate(sorted(pids))
        }
        result = []
        for path, pid, camid in parsed:
            if relabel:
                pid = pid2label[pid]
            result.append((osp.relpath(path, self.images_dir), pid, camid))
        return result
