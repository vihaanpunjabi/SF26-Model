import torch
import torch.nn as nn
import torch.optim as optim
from models import FireUNet, FirePredictor, PhysicsAdjuster
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Device (Use GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# 2. Instantiate Models
detection_model = FireUNet().to(device)
prediction_model = FirePredictor().to(device)
physics_model = PhysicsAdjuster().to(device)

optimizer = optim.Adam(list(detection_model.parameters()) + list(prediction_model.parameters()), lr=0.001)
criterion = nn.BCELoss()

def generate_fake_data(batch_size=4):
    images = torch.rand(batch_size, 3, 128, 128).to(device)
    masks = torch.zeros(batch_size, 5, 1, 128, 128).to(device)
    target = torch.zeros(batch_size, 1, 128, 128).to(device)
    
    center_x, center_y = 64, 64
    for i in range(5):
        masks[:, i, :, center_x-i:center_x+i+5, center_y-i:center_y+i+5] = 1.0
    
    target[:, :, center_x-5:center_x+10, center_y-5:center_y+10] = 1.0
    
    return images, masks, target

print("Starting Training Simulation...")
epochs = 50

for epoch in range(epochs):
    images, history_masks, target_mask = generate_fake_data()
    
    predicted_mask, _ = prediction_model(history_masks)
    
    wind_speed = torch.tensor(0.8)
    wind_dir = torch.tensor(45.0)
    final_prediction = physics_model(predicted_mask, wind_speed, wind_dir)
    
    loss = criterion(final_prediction, target_mask)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

torch.save(detection_model.state_dict(), "fire_detect_model.pth")
torch.save(prediction_model.state_dict(), "fire_predict_model.pth")
print("Models saved successfully!")

plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.title("History (Last Frame)")
plt.imshow(history_masks[0, -1, 0].cpu().numpy(), cmap='gray')
plt.subplot(1,3,2)
plt.title("AI + Physics Prediction")
plt.imshow(final_prediction[0, 0].cpu().detach().numpy(), cmap='jet')
plt.subplot(1,3,3)
plt.title("Actual Target")
plt.imshow(target_mask[0, 0].cpu().numpy(), cmap='gray')
plt.show()