import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataFrame(object):
    """Minimal pd.DataFrame analog for handling n-dimensional numpy matrices with additional
    support for shuffling, batching, and train/test splitting.
    Args:
        columns: List of names corresponding to the matrices in data.
        data: must be pd dataframe or pd series
    """

    def __init__(self, columns, data):
        assert len(columns) == len(data), 'columns length does not match data length'

        lengths = [mat.shape[0] for mat in data]
        assert len(set(lengths)) == 1, 'all matrices in data must have same first dimension'

        self.length = lengths[0]
        self.columns = columns
        self.data = data
        self.dict = dict(zip(self.columns, self.data))
        self.idx = np.arange(self.length)

    def shapes(self):
        return pd.Series(dict(zip(self.columns, [mat.shape for mat in self.data])))

    def dtypes(self):
        return pd.Series(dict(zip(self.columns, [mat.dtype for mat in self.data])))

    def shuffle(self):
        np.random.shuffle(self.idx)

    def train_test_split(self, train_size, random_state=np.random.randint(10000)):
        train_idx, test_idx = train_test_split(self.idx, train_size=train_size, random_state=random_state)
        train_df = DataFrame(copy.copy(self.columns), [mat.values[train_idx] for mat in self.data])
        test_df = DataFrame(copy.copy(self.columns), [mat.values[test_idx] for mat in self.data])
        return train_df, test_df

    def batch_generator(self, batch_size, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False):
        epoch_num = 0
        while epoch_num < num_epochs:
            if shuffle:
                self.shuffle()

            for i in range(0, self.length + 1, batch_size):
                batch_idx = self.idx[i: i + batch_size]
                if not allow_smaller_final_batch and len(batch_idx) != batch_size:
                    break
                # be carefull if we keep dataframe we need to turn it in numpy array before
                yield DataFrame(columns=copy.copy(self.columns), data=[mat.iloc[batch_idx].copy() for mat in self.data])

            epoch_num += 1

    def iterrows(self):
        for i in self.idx:
            yield self[i]

    def mask(self, mask):
        return DataFrame(copy.copy(self.columns), [mat.iloc[mask] for mat in self.data])

    def __iter__(self):
        return self.dict.items().__iter__()

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.dict[key]

        elif isinstance(key, int):
            return pd.Series(dict(zip(self.columns, [mat.iloc[self.idx[key]] for mat in self.data])))

    def __setitem__(self, key, value):
        assert value.shape[0] == len(self), 'matrix first dimension does not match'
        if key not in self.columns:
            self.columns.append(key)
            self.data.append(value)
        self.dict[key] = value
