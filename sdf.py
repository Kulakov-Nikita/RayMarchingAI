import torch


class SDF:
    def __init__(self, center_position: torch.Tensor, radius: torch.Tensor):
        self.center_position: torch.Tensor = center_position
        self.radius: torch.Tensor = radius

    def sdf(self, position: torch.Tensor):
        return torch.norm(position - self.center_position, dim=-1) - self.radius