# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine


class TorchCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        super().__init__(config_params)

    def create(self, tag):
        log_dist(f"[Torch] Checkpoint {tag} is about to be saved!", ranks=[0])

    def save(self, state_dict, path: str):
        logger.info(f"[Torch] Saving {path}...")
        if 'CEPHBUCKET' in os.environ and os.environ.get('CEPHBUCKET') is not None:
            self.ceph_save(state_dict, path)
        else:
            torch.save(state_dict, path)
        logger.info(f"[Torch] Saved {path}.")
        return None

    def get_ceph_path(self, path):
        ceph_bucket = os.environ.get('CEPHBUCKET')
        if ceph_bucket != '':
            if ceph_bucket[-1] != '/':
                ceph_bucket += '/'
            # remove /
            if path[0] == '/':
                path = path[1:]
            # remove ./
            if path[:2] == './':
                path = path[2:]
            ceph_path = ceph_bucket + path
        else:
            ceph_path = path
        return ceph_path

    def ceph_load(self, checkpoint_file, map_location=None):
        try:
            if os.environ.get('PETRELPATH', None) is not None:
                from petrel_helper import PetrelHelper
                with open(checkpoint_file, "r") as f:
                    ceph_path = f.readlines()[0].strip()
                if map_location is None:
                    map_location = 'cpu'
                return PetrelHelper.load(ceph_path, map_location=map_location)
        except:     # noqa
            logger.error("Fail to load ceph path from {}".format(checkpoint_file))     # noqa

    def ceph_save(self, state_dict, path):
        from petrel_helper import PetrelHelper
        ceph_path = self.get_ceph_path(path)
        path_tmp = path
        if "s3://" in path_tmp:
            path_tmp = path_tmp[5:]
        with open(path_tmp + '.ceph', 'w') as f:
            print(ceph_path, file=f)
        PetrelHelper.save(state_dict, ceph_path)

    def load(self, path: str, map_location=None):
        logger.info(f"[Torch] Loading checkpoint from {path}...")
        if path.endswith('.pt') and os.path.exists(path):
            partition = torch.load(path, map_location=map_location)
        elif path.endswith('.ceph') and os.path.exists(path):
            partition = self.ceph_load(path, map_location=map_location)
        else:
            raise NotImplementedError
        logger.info(f"[Torch] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[Torch] Checkpoint {tag} is ready now!")
        return True
