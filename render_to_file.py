import sys
import time

import numpy as np
import torch
import cv2

from camera import Camera
from light import Light
from sdfAdapter import SDFAdapter


class Renderer:
    def __init__(self, width: int, height: int, max_steps: int, epsilon: float, max_dist: float, device: torch.device, tensors_type: torch.dtype):
        self.max_steps: int = max_steps
        self.epsilon: float = epsilon  # Точность
        self.max_dist: float = max_dist  # Дальность отсечения

        self.device: torch.device = device
        self.tensor_type: torch.dtype = tensors_type
        self.camera = Camera(position=torch.tensor([0, 0, -3], device=self.device, dtype=tensors_type),
                    width=width, height=height, device=self.device, dtype=self.tensor_type)
        self.sdfObject: SDFAdapter = None
        self.lights: list[Light] = []

    def set_scene(self, sdfObject: SDFAdapter, lights: list[Light]) -> None:
        self.sdfObject = sdfObject
        self.lights = lights

    def estimate_normal(self, pos, sdf):
        e = 1e-3
        dx = torch.tensor([e, 0, 0], device=pos.device)
        dy = torch.tensor([0, e, 0], device=pos.device)
        dz = torch.tensor([0, 0, e], device=pos.device)
        normal = torch.stack([
            sdf.sdf(pos+ dx) - sdf.sdf(pos-dx),
            sdf.sdf(pos+ dy) - sdf.sdf(pos-dy),
            sdf.sdf(pos+ dz) - sdf.sdf(pos-dz)
        ], dim=-1)
        return torch.nn.functional.normalize(normal, dim=-1)

    def render_frame(self) -> np.ndarray:
        distances = torch.zeros([self.camera.width * self.camera.height, 1], device=self.device, dtype=self.tensor_type)
        origins = self.camera.get_start_rays_positions()
        positions = (origins + distances * self.camera.directions)

        for _ in range(self.max_steps):
            d = self.sdfObject.sdf(positions)

            if torch.all(d < self.epsilon):
                break

            distances += d
            positions = origins + distances * self.camera.directions
            mask = (distances < self.epsilon) | (distances > self.max_dist)

            if mask.all():
                break

        normals = self.estimate_normal(positions, self.sdfObject)
        brightness = torch.clamp(torch.sum(normals * self.lights[0].dir, dim=-1), 0, 1)
        brightness[distances > self.max_dist] = 0

        return (brightness.reshape((self.camera.height, self.camera.width)).cpu().numpy() * 255).astype(np.uint8)

def main():
    output_path: str = sys.argv[1]
    width: int = int(sys.argv[2])
    height: int = int(sys.argv[3])
    max_step: int = 30
    precision: float = 1e-3
    max_dist: float = 10.0

    for i in range(4, len(sys.argv), 2):
        print(sys.argv[i])
        match(sys.argv[i]):
            case '--max_step':
                max_step = int(sys.argv[i + 1])
            case '--precision':
                precision = float(sys.argv[i + 1])
            case '--max_dist':
                max_dist = float(sys.argv[i + 1])

    renderer = Renderer(width=width, height=height, max_steps=max_step, epsilon=precision, max_dist=max_dist,
                        device=torch.device('cuda'), tensors_type=torch.float32)

    renderer.set_scene(sdfObject=SDFAdapter(),
                       lights=[Light(torch.tensor([1,-1,-1], device=torch.device("cuda"), dtype=torch.float32))])

    start_time = time.time()
    frame = renderer.render_frame()
    end_time = time.time()
    print(f"Render time: {end_time-start_time}")
    cv2.imwrite(output_path, frame)





if __name__ == '__main__':
    main()