import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # use determinisitic algorithm
    print("Seed set", seed)

def weight_visualize(weight, modal):
    num_modal = len(modal)
    fig, axes = plt.subplots(1, num_modal, figsize=(12, 4))
    for i, m in enumerate(modal):
        sns.heatmap(weight[m], ax=axes[i], cmap='Blues') 
    return fig