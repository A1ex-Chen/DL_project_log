def nonempty(self) ->torch.Tensor:
    """
        Find masks that are non-empty.

        Returns:
            Tensor:
                a BoolTensor which represents whether each mask is empty (False) or not (True).
        """
    keep = [(1 if len(polygon) > 0 else 0) for polygon in self.polygons]
    return torch.from_numpy(np.asarray(keep, dtype=np.bool))
