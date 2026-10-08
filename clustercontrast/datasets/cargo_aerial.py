from __future__ import absolute_import

from .cargo_common import CargoDomain


class cargo_aerial(CargoDomain):
    domain_name = "aerial (Cam1--Cam5)"
    camera_ids = tuple(range(1, 6))
