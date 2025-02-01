import numpy as np
import torch


class Camera:
    def __init__(self, position: torch.Tensor, width: int, height: int, device, dtype):
        self.width: int = width
        self.height: int = height
        self.fov: float = np.pi / 3
        self.position: torch.Tensor = position
        self.aspect_ratio =  self.height / self.width

        y, x = torch.meshgrid(torch.linspace(-1, 1, self.width, device=device, dtype=dtype),
                          torch.linspace(-self.aspect_ratio, self.aspect_ratio, self.height, device=device, dtype=dtype),
                          indexing='ij')
        
        self.directions = torch.stack([x, y, torch.full_like(x, 1.0)], dim=-1)
        self.directions = torch.nn.functional.normalize(self.directions, dim=-1)

    def get_start_rays_positions(self):
        # rays_num = self.directions.shape[0] * self.directions.shape[1]
        # return torch.stack([self.position]**rays_num, dim=-1)
        return self.position

