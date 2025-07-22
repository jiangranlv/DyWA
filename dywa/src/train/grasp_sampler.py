import random

import numpy as np
import torch
from typing import Optional, Tuple
from train.mcmc_polynomial_decay import PolynomialSchedule


class Sampler:
    # max_length has to be large to not overfit!! -> 100,000
    def __init__(self, model, y_shape: Tuple, batch_size: int, num_negatives: int, device: torch.device, max_len: int = 1):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            y_shape - Shape of the target to model
            batch_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.y_shape = y_shape
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.device = device
        self.max_len = max_len
        self.examples = [(torch.rand((1,)+y_shape)*2-1) for _ in range(self.batch_size * self.num_negatives)]

    def sample_new_exmps(self, img: torch.Tensor, steps: int = 100, step_size: int = 0.5):
        """
        Function for getting a new batch of "fake" targets.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.batch_size * self.num_negatives, 0.05)
        rand_ys = torch.rand((n_new,) + self.y_shape) * 2 - 1
        old_ys = torch.cat(random.choices(self.examples, k=(self.batch_size * self.num_negatives) -n_new), dim=0)
        inp_ys = torch.cat([rand_ys, old_ys], dim=0).detach().to(self.device)
        inp_ys = inp_ys.reshape(self.batch_size, self.num_negatives, -1)

        # Perform MCMC sampling
        inp_ys = Sampler.generate_samples(self.model, img, inp_ys, steps=steps, step_size=step_size)
        # Add new targets to the buffer and remove old ones if needed
        self.examples = list(inp_ys.reshape((1, self.batch_size*self.num_negatives, self.y_shape[1])).to(
            torch.device("cpu")).chunk(self.batch_size*self.num_negatives, dim=1)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_ys

    @staticmethod
    def generate_samples(model, img: torch.Tensor, inp_ys: torch.Tensor, steps: int = 100, step_size: int = 0.5, return_ys_per_step: bool = False):
        """
        Function for sampling targets for a given model.
        Inputs:
            model - Neural network to use for modeling the energy function
            img - input observation
            inp_ys - Targets to start from for sampling. If you want to generate new targets, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_ys_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_ys.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_ys.shape, device=inp_ys.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []
        max_grad = 0.03 # Threshold on maximum gradients

        # create polynomial decay scheduler
        sampler_stepsize_final = 1e-5
        sampler_stepsize_power = 2.0
        schedule = PolynomialSchedule(step_size, sampler_stepsize_final,
                                    sampler_stepsize_power, steps)
        # Loop over K (steps)
        for k in range(steps):
            # Add noise to the input.
            noise.normal_(0, 0.005)
            inp_ys.data.add_(noise.data)
            inp_ys.data.clamp_(min=-1.0, max=1.0)

            # Calculate gradients for the current input.
            # negative sign comes from MCMC sampling algorithm
            energy = -model(img, inp_ys)
            energy.sum().backward()
            inp_ys.grad.data.clamp_(-max_grad, max_grad)

            # Apply gradients to our current samples
            # step_size = schedule.get_rate(k)
            inp_ys.data.add_(-step_size * inp_ys.grad.data)
            inp_ys.grad.detach_()
            inp_ys.grad.zero_()
            inp_ys.data.clamp_(min=-1.0, max=1.0)

            if return_ys_per_step:
                imgs_per_step.append(inp_ys.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train()

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_ys_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_ys