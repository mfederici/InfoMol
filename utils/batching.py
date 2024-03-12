from typing import Any, List

import numpy as np


def make_batches(obj_list: List[Any], batch_size: int) -> List[List[Any]]:
    n_batches = int(np.ceil(len(obj_list) / batch_size))
    batch_ids = np.arange(n_batches).astype(np.int32) * batch_size
    batch_sizes = [batch_size] * n_batches
    batch_sizes[-1] = len(obj_list) - batch_size * (n_batches - 1)

    return [obj_list[batch_ids[i]:batch_ids[i]+batch_sizes[i]] for i in range(n_batches)]
