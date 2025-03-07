import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import torch
import torchvision.transforms as transforms

from src.models.resnet import ResNet
from src.handspeak import HandSpeak

def main():
    # Define the same transformation as the training phase
    transform = transforms.Compose([
                            transforms.Resize((128, 128)),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.947], std=[0.177])
                        ])
    
    # 29 labels (A-Z + 0, Space, Backspace)
    class_labels = {
                        0: '0', 1: 'A', 2: 'B', 3: 'C', 4: 'D',
                        5: 'E', 6: 'F', 7: 'G', 8: 'H',
                        9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M',
                        14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 
                        19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W',
                        24: 'X', 25: 'Y', 26: 'Z', 27: 'backspace', 28: 'space'
                    }
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Define the same model as the training phase
    model = ResNet(1)
    # Load trained weights
    model.load_state_dict(torch.load('weights/asl_model_weights_extended.pth'))
    # Set model mode to evaluation
    model.eval()

    handspeak = HandSpeak(model, device, class_labels, transform)
    handspeak.run()

if __name__ == '__main__':
    main()