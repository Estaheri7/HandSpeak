import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import torch
import torchvision.transforms as transforms

from src.models.resnet import ResNet
from src.models.mlp import MLP
from src.handspeak import HandSpeak

def main():
    # Define the same transformation as the training phase
    transform = transforms.Compose([
                            transforms.Resize((128, 128)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.947], std=[0.177])
                        ])
    
    # 29 labels (A-Z + 0, Space, Backspace)
    class_labels = {0: 'AMNSTE', 1: 'COPQZX', 2: 'DFBUVLKRW', 
                    3: 'GHYJI', 4: 'space'}
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Define the same model as the training phase
    model = ResNet(3)
    mlp_amnste = MLP(15, 6)
    mlp_dfbuvlkrw = MLP(15, 9)
    mlp_copqzx = MLP(15, 6)
    mlp_ghyji = MLP(15, 5)

    models = {
    'amnste': mlp_amnste,
    'dfbuvlkrw': mlp_dfbuvlkrw,
    'ghyji': mlp_ghyji,
    'copqzx': mlp_copqzx,
    }

    # Load trained weights
    model.load_state_dict(torch.load('weights/group_predictor_weights.pth'))
    for name, mlp_model in models.items():
        mlp_model.load_state_dict(torch.load(f'weights/{name}_weights.pth'))
        mlp_model.eval()
    
    # Set model mode to evaluation
    model.eval()
    
    handspeak = HandSpeak(model, mlp_amnste, mlp_dfbuvlkrw, mlp_copqzx, mlp_ghyji,
                          device, class_labels, transform)
    handspeak.run()

if __name__ == '__main__':
    main()