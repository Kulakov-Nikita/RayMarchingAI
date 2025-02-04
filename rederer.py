import numpy as np
import pygame
import torch
from camera import Camera
from sdf import SDF
from light import Light
from sdfAdapter import SDFAdapter


class Renderer:
    def __init__(self, max_steps: int, epsilon: float, max_dist: float):
        self.max_steps: int = max_steps
        self.epsilon: float = epsilon  # Точность
        self.max_dist: float = max_dist  # Дальность отсечения
        self.dtype = torch.float16
        self.device = torch.device("cuda")

        # Инициализация PyGame
        pygame.init()
        # FIX IT !!!
        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()

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

    def render_frame(self, objects: list[SDF], lights: list[Light], camera: Camera) -> np.ndarray:
        distances = torch.zeros([camera.width, camera.height], device=self.device, dtype=self.dtype)
        origins = camera.get_start_rays_positions()
        positions = (origins + distances.unsqueeze(-1) * camera.directions).reshape((camera.width*camera.height, 3))
        for _ in range(self.max_steps):
            # if positions.ndim == 3:
            #     positions = positions.reshape((camera.width*camera.height, 3))
            d = objects[0].sdf(positions).reshape((camera.width, camera.height))
            # for obj in objects[1:]:
            #      distances = torch.minimum(distances, obj.sdf(positions))
            if torch.all(d < self.epsilon):
                break

            distances += d
            positions = (origins + distances.unsqueeze(-1) * camera.directions).reshape((camera.width*camera.height, 3))
            mask = (distances < self.epsilon) | (distances > self.max_dist)
            if mask.all():
                break
        
        # FIX IT !!!
        normals = self.estimate_normal(positions, objects[0])  # <- NOT 0
        brightness = torch.clamp(torch.sum(normals * lights[0].dir, dim=-1), 0, 1)
        brightness[distances.reshape((camera.width * camera.height, 1)) > self.max_dist] = 0

        return (brightness.reshape((camera.width, camera.height)).cpu().numpy() * 255).astype(np.uint8)
    

def main():
    # objects = [SDF(torch.tensor([0,0,0], device=torch.device("cuda"), dtype=torch.float16),
    #               torch.tensor(1.0, device=torch.device("cuda"), dtype=torch.float16))]
    objects = [SDFAdapter()]
    camera = Camera(position=torch.tensor([0,0,-3], device=torch.device("cuda"), dtype=torch.float32),
                    width=1280, height=720, device=torch.device("cuda"), dtype=torch.float32)
    lights = [Light(torch.tensor([1,-1,-1], device=torch.device("cuda"), dtype=torch.float32))]
    renderer = Renderer(max_steps=40, epsilon=1e-3, max_dist=10.0)
    running = True
    fps=0
    frame_count=0
    font = pygame.font.SysFont("Arial", 24)  # Шрифт для отображения FPS

    while running:
        start_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
             running = False

        image = renderer.render_frame(objects, lights, camera)

        # Вычисляем FPS
        frame_count += 1
        if frame_count % 10 == 0:
            end_time = pygame.time.get_ticks()  # Засекаем время окончания
            fps = 1000 / (end_time - start_time)
            frame_count = 0

        pygame.surfarray.blit_array(renderer.screen, np.stack([image] * 3, axis=-1))

        # Отображаем FPS
        fps_text = font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        renderer.screen.blit(fps_text, (10, 10))

        pygame.display.flip()
    
    pygame.quit()


if __name__ == '__main__':
    main()
