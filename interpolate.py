import argparse
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torchvision

from network import Generator


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--model_file_name',
                    default='generator.pth',
                    help='The file name of a trained model')
args = PARSER.parse_args()

INTERPOLATION_ID = int(time.time())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = f'trained_models/{args.model_file_name}'
SAVE_INTERPOLATION_DIR = f'interpolate/{INTERPOLATION_ID}'
STEPS = 600 # 600 intermediary vectors.

os.makedirs(SAVE_INTERPOLATION_DIR)

generator_model = Generator().to(DEVICE)
generator_model.load_state_dict(torch.load(MODEL_PATH))
generator_model.eval()
print(f'Loaded model "{MODEL_PATH}"')

# Generate two random 512-length vectors sampled from a normal distribution
VECTOR_A = torch.randn(1, 512, device=DEVICE)
VECTOR_B = torch.randn(1, 512, device=DEVICE)

# Interpolate between VECTOR_A and VECTOR_B.
interpolation_num = 0
for alpha in np.linspace(0, 1, STEPS):
    intermediary_vector = (1 - alpha) * VECTOR_A + alpha * VECTOR_B

    # Save image obtained from intermediary vector.
    with torch.no_grad():
        generated_image = generator_model(intermediary_vector).detach()
    generated_image = F.interpolate(generated_image, size=(128, 128), mode='nearest')
    torchvision.utils.save_image(generated_image, f'{SAVE_INTERPOLATION_DIR}/{interpolation_num:03d}.jpg', padding=2, normalize=True)
    
    # Print log every 200 interpolation steps.
    if interpolation_num % 200 == 0:
        print(f'Saved intermediary vector {interpolation_num}.')
        
    interpolation_num += 1           
