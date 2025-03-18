import pickle

from deepSDF import DeepSDF, fit
import torch
import numpy as np


class SDFAdapter:
    def __init__(self):
        # self.deepSDF, self.latent_code = fit()
        self.deepSDF = DeepSDF(latent_dim=16)
        self.deepSDF.load_state_dict(torch.load("model_1.model", weights_only=True))
        self.deepSDF.eval()
        self.deepSDF = self.deepSDF.cuda()
        with open("latent_code_1.lc", "rb") as file:
            self.latent_code = pickle.load(file)

    def sdf(self, positions: torch.Tensor):
        with torch.no_grad(): 
            sdf_values = self.deepSDF(self.latent_code[:1].expand(positions.shape[0], -1), positions)
        return sdf_values
    
if __name__ == '__main__':
    resolution=30
    x = np.linspace(-1.5, 1.5, resolution) 
    y = np.linspace(-1.5, 1.5, resolution) 
    z = np.linspace(-1.5, 1.5, resolution) 
    X, Y, Z = np.meshgrid(x, y, z) 
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T 
    points_torch = torch.tensor(points, dtype=torch.float32) 
    sdfAdapter = SDFAdapter()