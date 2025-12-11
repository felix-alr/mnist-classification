import torch


class OneHotTransform:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, x):
        return torch.zeros(self.num_classes, dtype=float).scatter_(0, self.convert_to_tensor_if_int(x), 1.0)

    def convert_to_tensor_if_int(self, x):
        if isinstance(x, int):
            return torch.tensor([x])
        return x