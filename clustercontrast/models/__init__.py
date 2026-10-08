from __future__ import absolute_import

from .agw_cargo_stable import agw_cargo_stable


__factory = {
    "agw_cargo_stable": agw_cargo_stable,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
