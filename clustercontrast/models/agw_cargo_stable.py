"""CARGO-specific AGW entry with numerically stable GeM pooling."""

from __future__ import absolute_import

from .agw_lag_stable import embed_net_ori


def agw_cargo_stable(pretrained=False, no_local='down', **kwargs):
    """Build the original AGW topology with overflow-safe GeM.

    The stable implementation is algebraically equivalent to the original
    p=3 generalized-mean pooling, but rescales each feature channel before
    cubing it.  This avoids FP32 overflow without clipping activations or
    changing the model topology.
    """
    model = embed_net_ori(no_local='on', gm_pool='on')
    model.stable_gem = True
    model.cargo_stable_gem = True
    return model
