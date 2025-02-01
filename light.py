import torch


class Light:
    def __init__(self, dir: torch.Tensor):
        self.dir = torch.nn.functional.normalize(dir, dim=-1)