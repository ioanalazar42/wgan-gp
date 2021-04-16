import argparse
import numpy as np
import os
import random
import torch

from network import Critic
from skimage import io


def load_image(path):
    image = io.imread(path)
    image = 2 * (image/255) - 1 # Mean normalize image.
    image = np.expand_dims(image.transpose(2, 0, 1), axis=0) # [img_size, img_size, 3] -> [3, img_size, img_size] -> [1, 3, img_size, img_size]
    return image.astype(np.float32)

# Directory that contains single 128x128 images generated using pretrained models. 
IMAGE_DIR_PATH = 'generated_with_preloaded_models/1x1'

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--model_file_name',
                    default='critic.pth',
                    help='The file name of a trained model.')
PARSER.add_argument('--image_path',
                    default=f'random',
                    help='The path to an image.')
args = PARSER.parse_args()

MODEL_PATH = f'trained_models/{args.model_file_name}'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up pretrained Critic.
critic_model = Critic().to(DEVICE)

critic_model.load_state_dict(torch.load(MODEL_PATH))
critic_model.eval()
print(f'\nLoaded model "{MODEL_PATH}"')

if args.image_path == 'random':
    print('Picking random image to score...')
    images = os.listdir(IMAGE_DIR_PATH)
    random_index = random.randint(0, len(images) - 1)
    IMAGE_PATH = os.path.join(IMAGE_DIR_PATH, images[random_index])
else:
    IMAGE_PATH = args.image_path


# Load the image and give it a score.
image = torch.tensor(load_image(IMAGE_PATH), device=DEVICE)
print(f'Loaded image "{IMAGE_PATH}"')
print(f'Critic score: {critic_model(image).item():.3f}')
