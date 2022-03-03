"""A minimal one-hot encoder to avoid importing large libraries such as sklearn."""
import logging

import numpy as np


class OneHotEncoder:

    def __init__(self, allow_unknown=False):
        self.logger = logging.getLogger(__name__)
        self._key_to_index_dict = dict()
        self._index_to_key_dict = dict()
        self.dim = None
        self.allow_unknown = allow_unknown

    def fit(self, data):
        for idx, key in enumerate(data):
            self._key_to_index_dict[key] = idx
            self._index_to_key_dict[idx] = key
        self.dim = len(data)

    def transform(self, data):
        if data not in self._key_to_index_dict.keys():
            if not self.allow_unknown:
                raise ValueError("OneHotEncoder received an unknown category.")
            self.logger.warning("Unknown key encountered")
            return np.zeros(self.dim, dtype=np.float32)
        x = np.zeros(self.dim, dtype=np.float32)
        x[self._key_to_index_dict[data]] = 1
        return x

    def inverse_transform(self, data):
        key = np.where(data == 1)
        if not len(key[0]) == 1 or len(data) != self.dim:
            raise ValueError("OneHotEncoder received an unknown category.")
        return self._index_to_key_dict[key[0][0]]
