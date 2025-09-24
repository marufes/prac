import torch
from loss_function import build_target, Focal_Loss, CE_Loss, Dice_loss
import distributed_utils as utils
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Loss function
# ---------------------------
def criterion(inputs, target, num_classes: int = 2, focal_loss: bool = True, dice_loss: bool = True):
    losses = {}

    for name, x in inputs.items():
        if focal_loss:
            loss = Focal_Loss(x, target, ignore_index=255)
        else:
            loss = CE_Loss(x, target, ignore_index=255)

        if dice_loss:
            dice_target = build_target(target, num_classes, ignore_index=255)
            dice_loss_val = Dice_loss(x, dice_target)
            loss = loss + dice_loss_val

        losses[name] = loss

    return losses['out']   # ✅ properly inside function


# ---------------------------
# Helper: visualize Sobel edges
# ---------------------------
def visualize_edges(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().astype(np.uint8)

    sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)

    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(grad_mag, 50, 255, cv2.THRESH_BINARY)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("Gradient Magnitude (Sobel)")
    plt.imshow(grad_mag, cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Thresholded Edges")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.show()
