"""This class read MTER model files
"""

import os
import warnings

import numpy as np

from util import load_dict


class MTERReader:
    def __init__(
        self,
        input_path=None,
        rating_scale=5,
        verbose=False,
    ):
        self.uid_map = load_dict(os.path.join(input_path, "uid_map"), sep=",")
        self.iid_map = load_dict(os.path.join(input_path, "iid_map"), sep=",")
        self.aspect_id_map = load_dict(
            os.path.join(input_path, "aspect_id_map"), sep=","
        )
        self.opinion_id_map = load_dict(
            os.path.join(input_path, "opinion_id_map"), sep=","
        )
        self.U = np.load(os.path.join(input_path, "U.npy"))
        self.I = np.load(os.path.join(input_path, "I.npy"))
        self.A = np.load(os.path.join(input_path, "A.npy"))
        self.O = np.load(os.path.join(input_path, "O.npy"))
        self.G1 = np.load(os.path.join(input_path, "G1.npy"))
        self.G2 = np.load(os.path.join(input_path, "G2.npy"))
        self.G3 = np.load(os.path.join(input_path, "G3.npy"))
        self.rating_scale = rating_scale
        self.id2aspect = {v: k for k, v in self.aspect_id_map.items()}
        self.verbose = verbose
        if self.verbose:
            print("Load MTER from %s" % input_path)

    @property
    def num_items(self):
        return len(self.iid_map)

    @property
    def num_users(self):
        return len(self.uid_map)

    @property
    def num_aspects(self):
        return len(self.aspect_id_map)

    @property
    def num_opinions(self):
        return len(self.opinion_id_map)

    @property
    def raw_uid_map(self):
        return {v: k for k, v in self.uid_map.items()}

    @property
    def raw_iid_map(self):
        return {v: k for k, v in self.iid_map.items()}

    @property
    def raw_aspect_id_map(self):
        return {v: k for k, v in self.aspect_id_map.items()}

    @property
    def raw_opinion_id_map(self):
        return {v: k for k, v in self.opinion_id_map.items()}

    def get_aspect_vector(self, raw_uid, raw_iid):
        """
        This function return the aspect vector of a given user and item
        """
        uid = self.uid_map.get(raw_uid)
        iid = self.iid_map.get(raw_iid)
        return np.einsum(
            "c,Nc->N",
                np.einsum(
                    "bc,b->c",
                    np.einsum(
                        "abc,a->bc",
                        self.G1, self.U[uid]
                    ),
                    self.I[iid]
                ),
                self.A[:-1]
            )

    def get_aspect_score(self, raw_uid, raw_iid, aspect):
        uid = self.uid_map.get(raw_uid)
        iid = self.iid_map.get(raw_iid)
        aspect_id = self.aspect_id_map.get(aspect)
        if uid is None or iid is None or aspect_id is None:
            warnings.warn(
                "Aspect sentiment score is not available for "
                + "user=[%s], item=[%s], aspect=[%s], this function will return 0.0"
                % (raw_uid, raw_iid, aspect)
            )
            return 0.0
        return np.einsum(
                "c,c->",
                np.einsum(
                    "bc,b->c",
                    np.einsum(
                        "abc,a->bc",
                        self.G1, self.U[uid]
                    ),
                    self.I[iid]
                ),
                self.A[aspect_id]
            )

    def is_unk_user(self, raw_uid):
        return self.get_uid(raw_uid) is None

    def is_unk_item(self, raw_iid):
        return self.get_iid(raw_iid) is None

    def get_uid(self, raw_uid):
        return self.uid_map.get(raw_uid, None)

    def get_iid(self, raw_iid):
        return self.iid_map.get(raw_iid, None)
