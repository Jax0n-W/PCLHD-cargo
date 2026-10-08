from __future__ import absolute_import

from .cargo_common import CargoDomain


class cargo_ground(CargoDomain):
    domain_name = "ground (Cam6--Cam13)"
    camera_ids = tuple(range(6, 14))
