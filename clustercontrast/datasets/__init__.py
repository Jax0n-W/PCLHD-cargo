from __future__ import absolute_import

from .cargo_aerial import cargo_aerial
from .cargo_ground import cargo_ground


__factory = {
    "cargo_aerial": cargo_aerial,
    "cargo_ground": cargo_ground,
}


def names():
    return sorted(__factory.keys())


def create(name, root, trial=0, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](root, trial=trial, *args, **kwargs)
