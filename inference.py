import torch
import torch.nn as nn
from models import FireUNet, FirePredictor, PhysicsAdjuster
import numpy as np
import cv2
import matplotlib.pyplot as plt

class FireSpreadPredictor:
    """End-to-end fire spread prediction pipeline"""
    
    def __init__(self, device='cpu', detection_model_path=None, prediction_model_path=None):
        self.device = torch.device(device)
        
        # Initialize models
        self.detection_model = FireUNet().to(self.device)
        self.prediction_model = FirePredictor().to(self.device)
        self.physics_model = PhysicsAdjuster().to(self.device)
        
        # Load pre-trained weights if available
        if detection_model_path:
            self.detection_model.load_state_dict(torch.load(detection_model_path, map_location=self.device))
        if prediction_model_path:
            self.prediction_model.load_state_dict(torch.load(prediction_model_path, map_location=self.device))
        
        # Set to eval mode
        self.detection_model.eval()
        self.prediction_model.eval()
        self.physics_model.eval()
    
    def detect_fire(self, image):
        """
        Detect fire in a single image
        Args:
            image: numpy array or tensor of shape (H, W, 3) or (1, 3, H, W)
        Returns:
            fire_mask: tensor of shape (1, 1, H, W) with fire probability
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.dim() == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)
        
        image = image.to(self.device)
        with torch.no_grad():
            fire_mask = self.detection_model(image)
        return fire_mask
    
    def predict_spread(self, history_masks, wind_speed=0.5, wind_direction=0.0):
        """
        Predict fire spread based on history
        Args:
            history_masks: tensor of shape (1, 5, 1, H, W) - 5 frames of history
            wind_speed: normalized speed (0.0-1.0)
            wind_direction: direction in degrees
        Returns:
            final_prediction: predicted fire mask after physics adjustment
        """
        history_masks = history_masks.to(self.device)
        
        with torch.no_grad():
            # Get base prediction from ConvLSTM
            predicted_mask, _ = self.prediction_model(history_masks)
            
            # Apply physics adjustments (wind, terrain)
            wind_speed_tensor = torch.tensor(wind_speed).to(self.device)
            wind_dir_tensor = torch.tensor(wind_direction).to(self.device)
            final_prediction = self.physics_model(predicted_mask, wind_speed_tensor, wind_dir_tensor)
        
        return final_prediction
    
    def visualize_prediction(self, history_masks, prediction, target=None, save_path=None):
        """Visualize the prediction results"""
        fig, axes = plt.subplots(1, 3 if target is None else 4, figsize=(15, 4))
        
        # Last frame of history
        axes[0].imshow(history_masks[0, -1, 0].cpu().numpy(), cmap='gray')
        axes[0].set_title('Last Frame (History)')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(prediction[0, 0].cpu().detach().numpy(), cmap='jet')
        axes[1].set_title('AI + Physics Prediction')
        axes[1].axis('off')
        
        # Target if provided
        if target is not None:
            axes[2].imshow(target[0, 0].cpu().numpy(), cmap='gray')
            axes[2].set_title('Actual Target')
            axes[2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


if __name__ == '__main__':
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = FireSpreadPredictor(device=device)
    
    # Create dummy data
    dummy_history = torch.rand(1, 5, 1, 128, 128)
    
    # Run prediction
    prediction = predictor.predict_spread(dummy_history, wind_speed=0.8, wind_direction=45.0)
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
