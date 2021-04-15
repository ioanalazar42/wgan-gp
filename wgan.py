import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from datareader import load_images
from network import Critic, Generator
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
from utils import sample_gradient_l2_norm

# Define constants.
EXPERIMENT_ID = int(time.time()) # Used to create new directories to save results of individual experiments.

DEFAULT_IMG_DIR = 'images/{}'.format(EXPERIMENT_ID)
DEFAULT_TENSORBOARD_DIR = 'tensorboard/{}'.format(EXPERIMENT_ID)
DEFAULT_MODEL_DIR = 'models/{}'.format(EXPERIMENT_ID)
IMG_SIZE = 128

PARSER = argparse.ArgumentParser()

PARSER.add_argument('--data_dir', default='/home/datasets/celeba-aligned')
PARSER.add_argument('--load_critic_model_path')
PARSER.add_argument('--load_generator_model_path')
PARSER.add_argument('--save_image_dir', default=DEFAULT_IMG_DIR)
PARSER.add_argument('--save_model_dir', default=DEFAULT_MODEL_DIR)
PARSER.add_argument('--tensorboard_dir', default=DEFAULT_TENSORBOARD_DIR)
PARSER.add_argument('--dry_run', default=False, type=bool)
PARSER.add_argument('--model_save_frequency', default=15, type=int)
PARSER.add_argument('--training_set_size', default=99999999, type=int)
PARSER.add_argument('--epoch_length', default=100, type=int)
PARSER.add_argument('--gradient_penalty_factor', default=10, type=float)
PARSER.add_argument('--learning_rate', default=0.0001, type=float)
PARSER.add_argument('--mini_batch_size', default=256, type=int)
PARSER.add_argument('--num_critic_training_steps', default=2, type=int)
PARSER.add_argument('--num_epochs', default=500, type=int)

args = PARSER.parse_args()

# Create directories for images, tensorboard results and saved models.
if not args.dry_run:
    os.makedirs(args.save_image_dir)
    os.makedirs(args.tensorboard_dir)
    os.makedirs(args.save_model_dir)
    WRITER = SummaryWriter(args.tensorboard_dir) # Set up TensorBoard.
else:
    print('Dry run! Just for testing, data is not saved')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up the GAN.
critic_model = Critic().to(device)
generator_model = Generator().to(device)

# Load pre-trained models if they are provided.
if args.load_critic_model_path:
    critic_model.load_state_dict(torch.load(args.load_critic_model_path))

if args.load_generator_model_path:
    generator_model.load_state_dict(torch.load(args.load_generator_model_path))

# Set up Adam optimizers for both models.
critic_optimizer = optim.Adam(critic_model.parameters(), lr=args.learning_rate, betas=(0, 0.9))
generator_optimizer = optim.Adam(generator_model.parameters(), lr=args.learning_rate, betas=(0, 0.9))

# Create a random batch of latent space vectors that will be used to visualize the progression of the generator.
fixed_latent_space_vectors = torch.randn(64, 512, device=device)

# Load and preprocess images.
images = load_images(args.data_dir, args.training_set_size)

# Add network architectures for Critic and Generator to TensorBoard.
WRITER.add_graph(critic_model, torch.tensor(images[:1], device=device))
WRITER.add_graph(generator_model, fixed_latent_space_vectors)

for epoch in range(args.num_epochs):
    start_time = timer()
    
    # Variables for recording statistics.
    average_critic_real_performance = 0.0  # C(x) - The critic wants this to be as big as possible for real images.
    average_critic_generated_performance = 0.0  # C(G(x)) - The critic wants this to be as small as possible for generated images.
    average_critic_loss = 0.0
    average_generator_loss = 0.0

    # Train: perform `args.epoch_length` mini-batch updates per "epoch".
    for i in range(args.epoch_length):
        # Train the critic:
        for i in range(args.num_critic_training_steps):
            critic_model.zero_grad()

            # Evaluate a mini-batch of real images.
            random_indexes = np.random.choice(len(images), args.mini_batch_size)
            real_images = torch.tensor(images[random_indexes], device=device)

            real_scores = critic_model(real_images)

            # Evaluate a mini-batch of generated images.
            random_latent_space_vectors = torch.randn(args.mini_batch_size, 512, device=device)
            generated_images = generator_model(random_latent_space_vectors)

            generated_scores = critic_model(generated_images.detach())

            gradient_l2_norm = sample_gradient_l2_norm(critic_model, real_images, generated_images, device)
            
            # Update the weights.
            loss = torch.mean(generated_scores) - torch.mean(real_scores) + args.gradient_penalty_factor * gradient_l2_norm  # The critic's goal is for `generated_scores` to be small and `real_scores` to be big.
            loss.backward()
            critic_optimizer.step()

            # Record some statistics.
            average_critic_loss += loss.item() / args.num_critic_training_steps / args.epoch_length
            average_critic_real_performance += real_scores.mean().item() / args.num_critic_training_steps / args.epoch_length
            average_critic_generated_performance += generated_scores.mean().item() / args.num_critic_training_steps / args.epoch_length

        # Train the generator:
        generator_model.zero_grad()
        generated_scores = critic_model(generated_images)

        # Update the weights.
        loss = -torch.mean(generated_scores)  # The generator's goal is for `generated_scores` to be big.
        loss.backward()
        generator_optimizer.step()

        # Record some statistics.
        average_generator_loss += loss.item() / args.epoch_length
 
    # Record time elapsed for current epoch.
    time_elapsed = timer() - start_time

    print(('Epoch: {} - Critic Loss: {:.6f} - Generator Loss: {:.6f} - Average C(x): {:.6f} - Average C(G(x)): {:.6f} - Time: {:.3f}s')
        .format(epoch, average_critic_loss , average_generator_loss, average_critic_real_performance, average_critic_generated_performance, time_elapsed))
    
    # Save model parameters, tensorboard data, generated images.
    if (not args.dry_run):
        # Save generated images.
        with torch.no_grad():
            generated_images = generator_model(fixed_latent_space_vectors).detach()
        torchvision.utils.save_image(generated_images, '{}/{}-{}x{}.jpg'.format(args.save_image_dir, epoch, IMG_SIZE, IMG_SIZE), padding=2, normalize=True)
        
        # Create a grid of generated images to save to Tensorboard.
        grid_images = torchvision.utils.make_grid(generated_images, padding=2, normalize=True)

        # Save tensorboard data.
        WRITER.add_image('training/generated-images', grid_images, epoch)
        WRITER.add_scalar('training/generator/loss', average_generator_loss, epoch)
        WRITER.add_scalar('training/critic/loss', average_critic_loss, epoch)
        WRITER.add_scalar('training/critic/real-performance', average_critic_real_performance, epoch)
        WRITER.add_scalar('training/critic/generated-performance', average_critic_generated_performance, epoch)
        WRITER.add_scalar('training/epoch-duration', time_elapsed, epoch)

        # Save the model parameters at a specified interval.
        if (epoch > 0 and (epoch % args.model_save_frequency == 0
            or epoch == args.num_epochs - 1)):
            save_critic_model_path = '{}/critic_{}-{}.pth'.format(args.save_model_dir, EXPERIMENT_ID, epoch)
            print('\nSaving critic model as "{}"...'.format(save_critic_model_path))
            torch.save(critic_model.state_dict(), save_critic_model_path)
        
            save_generator_model_path = '{}/generator_{}-{}.pth'.format(args.save_model_dir, EXPERIMENT_ID, epoch)
            print('Saving generator model as "{}"...\n'.format(save_generator_model_path,))
            torch.save(generator_model.state_dict(), save_generator_model_path)

print('Finished training!')
