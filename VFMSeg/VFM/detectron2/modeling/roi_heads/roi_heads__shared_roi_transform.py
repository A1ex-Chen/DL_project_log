def _shared_roi_transform(self, features: List[torch.Tensor], boxes: List[
    Boxes]):
    x = self.pooler(features, boxes)
    return self.res5(x)
