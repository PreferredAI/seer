"""This class read EFM model files
"""

import os
import warnings

import numpy as np

from util import load_dict


class EFMReader:
    def __init__(
        self,
        input_path=None,
        alpha=0.85,
        num_most_cared_aspects=15,
        rating_scale=5,
        verbose=False,
    ):
        self.uid_map = load_dict(os.path.join(input_path, "uid_map"), sep=",")
        self.iid_map = load_dict(os.path.join(input_path, "iid_map"), sep=",")
        self.aspect_id_map = load_dict(
            os.path.join(input_path, "aspect_id_map"), sep=","
        )
        self.U1 = np.load(os.path.join(input_path, "U1.npy"))
        self.U2 = np.load(os.path.join(input_path, "U2.npy"))
        self.V = np.load(os.path.join(input_path, "V.npy"))
        self.H1 = np.load(os.path.join(input_path, "H1.npy"))
        self.H2 = np.load(os.path.join(input_path, "H2.npy"))
        self.alpha = alpha
        self.n_cared_aspects = num_most_cared_aspects
        self.rating_scale = rating_scale
        self.id2aspect = {v: k for k, v in self.aspect_id_map.items()}
        self.verbose = verbose
        if self.verbose:
            print("Load EFM from %s" % input_path)

    @property
    def num_items(self):
        return len(self.iid_map)

    @property
    def num_users(self):
        return len(self.uid_map)

    @property
    def raw_uid_map(self):
        return {v: k for k, v in self.uid_map.items()}

    @property
    def raw_iid_map(self):
        return {v: k for k, v in self.iid_map.items()}

    @property
    def raw_aspect_id_map(self):
        return {v: k for k, v in self.aspect_id_map.items()}

    def get_aspect_quality(self, raw_iid, aspect):
        iid = self.iid_map.get(raw_iid)
        aspect_id = self.aspect_id_map.get(aspect)
        return self.U2[iid, :].dot(self.V[aspect_id, :])

    def get_aspect_vector(self, raw_uid, raw_iid):
        uid = self.uid_map.get(raw_uid)
        iid = self.iid_map.get(raw_iid)
        return self.U1[uid, :].dot(self.V.T) * self.U2[iid, :].dot(self.V.T)

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
        return self.U1[uid, :].dot(self.V[aspect_id, :]) * self.U2[iid, :].dot(
            self.V[aspect_id, :]
        )

    def get_most_cared_aspect_ids(self, raw_uid):
        uid = self.uid_map.get(raw_uid)
        X_ = self.U1[uid, :].dot(self.V.T)
        return (-X_).argsort()[: self.n_cared_aspects]

    def get_most_cared_aspects(self, raw_uid):
        return [
            self.id2aspect.get(aid) for aid in self.get_most_cared_aspect_ids(raw_uid)
        ]

    def is_unk_user(self, raw_uid):
        return self.get_uid(raw_uid) is None

    def is_unk_item(self, raw_iid):
        return self.get_iid(raw_iid) is None

    def get_uid(self, raw_uid):
        return self.uid_map.get(raw_uid, None)

    def get_iid(self, raw_iid):
        return self.iid_map.get(raw_iid, None)

    def rank(self, raw_uid, raw_iids=[]):
        mapped_iids = []
        for raw_iid in raw_iids:
            if self.is_unk_item(raw_iid):
                continue
            mapped_iids.append(self.get_iid(raw_iid))
        mapped_iids = np.array(mapped_iids)
        mapped_uid = self.get_uid(raw_uid)
        X_ = self.U1[mapped_uid, :].dot(self.V.T)
        aspect_ids = (-X_).argsort()[: self.n_cared_aspects]
        most_cared_X_ = X_[aspect_ids]
        most_cared_Y_ = self.U2[mapped_iids, :].dot(self.V[aspect_ids, :].T)
        explicit_scores = most_cared_X_.dot(most_cared_Y_.T) / (
            self.n_cared_aspects * self.rating_scale
        )
        rating_scores = self.U1[mapped_uid, :].dot(self.U2[mapped_iids, :].T) + self.H1[
            mapped_uid, :
        ].dot(self.H2[mapped_iids, :].T)
        ranking_scores = self.alpha * explicit_scores + (1 - self.alpha) * rating_scores

        return mapped_iids[ranking_scores.argsort()[::-1]]

    def get_top_k_ranked_items(self, raw_uid, raw_iids=[], top_k=10):
        return self.rank(raw_uid, raw_iids)[:top_k]
