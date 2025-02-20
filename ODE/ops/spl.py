import numpy as np
import torch
import torch.nn as nn


def compute_gradients_length(gradients, channel=False):
    gradients_length = []
    for current_gradient in gradients:
        current_gradient = current_gradient.cpu().numpy()
        if channel:
            length = 0
            for g in current_gradient:
                g = g.flatten()
                length += np.linalg.norm(g)
            length = length / len(current_gradient)
        else:
            g = current_gradient.flatten()
            length = np.linalg.norm(g)
        gradients_length.append(length)

    return torch.tensor(gradients_length).to(gradients.device)


class SelfPacedLearning(nn.Module):
    def __init__(self, lmda=0.1):
        super().__init__()
        self.lmda = lmda

    def forward(self, loss, gradients, difficulty_type):
        # print("Difficulty Type: {}".format(difficulty_type))
        if difficulty_type == "loss":
            example_difficulty = loss
        elif difficulty_type == "gradients":
            example_difficulty = compute_gradients_length(gradients)
        else:
            raise NotImplementedError

        weight_matrix = self.compute_weight_matrix(example_difficulty)
        loss = loss * weight_matrix
        loss = loss.sum() / torch.count_nonzero(loss)
        return loss

    def compute_weight_matrix(self, loss):
        weight_matrix = torch.zeros_like(loss)
        _, indices = torch.topk(loss, int(loss.numel() * self.lmda), largest=False)
        weight_matrix[indices] = 1

        return weight_matrix


class SymmetricSelfPacedLearning(nn.Module):
    def __init__(self, cfg, current_epoch):
        super().__init__()
        self.cfg = cfg
        self.eta = 1
        self.current_epoch = current_epoch
        self.epoch_step_size = 2 / (self.cfg.OPTIM.MAX_EPOCH - 1)
        self.weight_first = 2 - self.current_epoch * self.epoch_step_size
        self.weight_last = 2 - self.weight_first

    def forward(self, loss, gradients, difficulty_type):
        if difficulty_type == "loss":
            example_difficulty = loss
        elif difficulty_type == "gradients":
            example_difficulty = compute_gradients_length(gradients)
        elif difficulty_type == "LGDM":
            gradients_length = compute_gradients_length(gradients)
            example_difficulty = 0.5 * loss + 0.5 * gradients_length
        else:
            raise NotImplementedError

        weight_matrix = self.compute_weight_matrix(example_difficulty)
        loss = loss * weight_matrix
        loss = loss.mean()

        return loss, example_difficulty

    def compute_weight_matrix(self, example_difficulty):
        batch_step_size = (self.weight_first - self.weight_last) / (
            len(example_difficulty) - 1
        )
        weight_matrix = torch.zeros_like(example_difficulty)
        indices = torch.argsort(example_difficulty)

        for i, index in enumerate(indices):
            weight_matrix[index] = self.weight_first - batch_step_size * i

        return weight_matrix
