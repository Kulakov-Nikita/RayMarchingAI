import pickle

import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import torch._dynamo
torch._dynamo.config.suppress_errors = True
 
# Архитектура DeepSDF 
class DeepSDF(nn.Module): 
    def __init__(self, latent_dim):
        super().__init__() 
        self.latent_dim = latent_dim 
        self.model = nn.Sequential( 
            nn.Linear(latent_dim + 3, 256), nn.ReLU(),
            nn.Linear(256, 1)  # Выход - значение SDF
        ) 
     
    def forward(self, latent_code, coords):
        x = torch.cat([latent_code, coords], dim=-1) 
        return self.model(x) 
 
# Функция генерации обучающих данных 
def generate_sdf_samples(num_samples=10_000, radius=1.0):
    points = np.random.uniform(-1.5, 1.5, size=(num_samples, 3))
    sdf_values = np.linalg.norm(points, axis=1) - radius
    return torch.tensor(points, dtype=torch.float32, device=torch.device("cuda")), torch.tensor(sdf_values, dtype=torch.float32, device=torch.device("cuda")).unsqueeze(1)

def fit():
    # Инициализация модели и оптимизатора 
    latent_dim = 16
    model = DeepSDF(latent_dim=latent_dim).cuda()
    model.compile()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.MSELoss() 
    
    # Обучение модели 
    num_epochs = 10
    batch_size = 128 
    latent_code = torch.zeros((batch_size, latent_dim), device=torch.device("cuda"))  # Представление фигуры в латентном пространстве
    
    for epoch in range(num_epochs): 
        points, sdf_values = generate_sdf_samples() 
        dataset = torch.utils.data.TensorDataset(points, sdf_values) 
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) 
        
        total_loss = 0 
        for batch_points, batch_sdf in dataloader: 
            optimizer.zero_grad() 
            
            # Расширяем latent_code до нужного batch_size 
            batch_latent_code = latent_code[:batch_points.shape[0]] 
            
            sdf_pred = model(batch_latent_code, batch_points)
            loss = criterion(sdf_pred, batch_sdf) 
            loss.backward() 
            optimizer.step() 
            total_loss += loss.item() 
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.6f}") 

    return model, latent_code
 
def visualize_sdf(model, latent_code, resolution=30): 
    x = np.linspace(-1.5, 3, resolution)
    y = np.linspace(-1.5, 1.5, resolution) 
    z = np.linspace(-1.5, 1.5, resolution) 
    X, Y, Z = np.meshgrid(x, y, z) 
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T 
    points_torch = torch.tensor(points, dtype=torch.float32, device=torch.device('cuda'))

    with torch.no_grad(): 
        sdf_values = model(latent_code[:1].expand(points_torch.shape[0], -1), points_torch).cpu().numpy().flatten()
     
    surface_points = points[np.abs(sdf_values) < 0.05] 
     
    fig = plt.figure(figsize=(8, 8)) 
    ax = fig.add_subplot(111, projection='3d') 
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], c='b', marker='o', s=1, alpha=0.5) 
    ax.set_xlabel("X") 
    ax.set_ylabel("Y") 
    ax.set_zlabel("Z") 
    ax.set_title("SDF-предсказанная поверхность") 
    plt.show()

def visualize_sample_sdf(resolution=30):
    x = np.linspace(-1.5, 1.5, resolution)
    x2 = np.linspace(0, 3, resolution)
    y = np.linspace(-1.5, 1.5, resolution)
    z = np.linspace(-1.5, 1.5, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    X2, Y2, Z2 = np.meshgrid(x2, y, z)
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    points2 = np.vstack([X2.ravel(), Y2.ravel(), Z2.ravel()]).T
    points_torch = torch.tensor(points, dtype=torch.float32, device=torch.device('cuda'))
    points_torch2 = torch.tensor(points2, dtype=torch.float32, device=torch.device('cuda'))

    sdf_values_one = np.linalg.norm(points_torch.cpu(), axis=1) - 1
    sdf_values_two = np.linalg.norm(points_torch2.cpu(), axis=1) - 1
    sdf_values = np.min([sdf_values_one, sdf_values_two], axis=0)


    surface_points = points[np.abs(sdf_values) < 0.05]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], c='b', marker='o', s=1, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("SDF-предсказанная поверхность")
    plt.show()

if __name__ == '__main__':
    model, latent_code = fit()
    # visualize_sdf(model, latent_code)
    torch.save(model.state_dict(), "model_1.model")
    with open("latent_code_1.lc", "wb") as file:
        pickle.dump(latent_code, file)
    # visualize_sample_sdf()
