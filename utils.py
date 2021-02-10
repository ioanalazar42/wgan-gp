import torch

from torch import autograd


def sample_gradient_l2_norm(critic_model, real_images, generated_images, device):
    '''Calculates the critic's gradient for a random interpolation of `real_images` and `generated_images` and returns its L2-norm.'''

    mini_batch_size = real_images.shape[0]

    # Use a mixture of real images and generated images merged together.
    percentages = torch.rand(mini_batch_size, 1, 1, 1, device=device)
    interpolated_images = percentages * real_images + (1 - percentages) * generated_images
    interpolated_images = interpolated_images.detach().requires_grad_(True)  # PyTorch requires this ¯\_(ツ)_/¯

    scores = critic_model(interpolated_images)

    # The `outputs` and `inputs` parameters accept a list of tensors, but we're only passing a single tensor
    # to each (`scores` to outputs and `interpolated_images` to inputs). Still, `autograd.grad(...)` returns
    # a list of gradients for each (output, input) pair, but since we've only passed one input and one output,
    # the list will have a single element and so we use `[0]` to retrieve it.
    gradients = autograd.grad(outputs=scores,
                              inputs=interpolated_images,
                              grad_outputs=torch.ones(scores.shape, device=device),
                              retain_graph=True,
                              create_graph=True,
                              only_inputs=True)[0]

    # Change the gradient's shape from [mini_batch_size, 3, 128, 128] to [mini_batch_size, 3*128*128].
    gradients = gradients.view(mini_batch_size, -1)

    l2_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
    return torch.mean((l2_norm - 1) ** 2)
