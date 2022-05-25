import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# by RoshanRane in https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing/exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as "plot_grad_flow(model.named_parameters())" to
        visualize the gradient flow.
    """
    avg_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and (p.grad is not None) and ('bias' not in n):
            layers.append(n)
            avg_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())

    plt.figure(figsize=(7, 20))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color='c')
    plt.bar(np.arange(len(avg_grads)), avg_grads, alpha=0.1, lw=1, color='b')
    plt.hlines(0, 0, len(avg_grads)+1, lw=2, color='k')
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation='vertical')
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom=-0.001, top=0.5)   # zoom in on the lower gradient regions
    plt.xlabel('Layers')
    plt.ylabel('Average Gradient')
    plt.title('Gradient flow')
    plt.grid(True)
    plt.legend([Line2D([0], [0], color='c', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='k', lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(5)
    plt.ioff()