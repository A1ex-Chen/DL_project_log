@staticmethod
def collate_fn(item_tuple_batch: List[ItemTuple]) ->Tuple[ItemTuple]:
    return tuple(item_tuple_batch)
