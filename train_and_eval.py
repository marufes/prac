########################################################################################################################
# Training & Evaluation APIs
########################################################################################################################

import torch
from torch import nn
import torch.nn.functional as F
import distributed_utils as utils
from loss_function import build_target, Focal_Loss, CE_Loss, Dice_loss

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Loss function
# ---------------------------
def criterion(inputs, target, num_classes: int = 2, focal_loss: bool = True, dice_loss: bool = True, show_sobel=False):
    """
    Compute loss for a batch of predictions.
    inputs: dict of model outputs (e.g., {'out': output})
    target: ground truth tensor
    num_classes: number of classes
    focal_loss: use focal loss if True
    dice_loss: add dice loss if True
    show_sobel: visualize Sobel edges of target (optional)
    """
    losses = {}

    if show_sobel:
        # Move target to CPU and convert to numpy for OpenCV
        img = target.detach().cpu().numpy().astype(np.uint8)
        if img.ndim == 4:  # if batch dimension exists
            img = img[0, 0]  # visualize first sample and first channel
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges = cv2.threshold(grad_mag, 50, 255, cv2.THRESH_BINARY)
        plt.title("Gradient Magnitude (Sobel)")
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        plt.show()

    # Compute loss
    for name, x in inputs.items():
        if focal_loss:
            loss = Focal_Loss(x, target, ignore_index=255)
        else:
            loss = CE_Loss(x, target, ignore_index=255)

        if dice_loss:
            dice_target = build_target(target, num_classes, ignore_index=255)
            loss += Dice_loss(x, dice_target)

        losses[name] = loss

    return losses['out']  # return the main loss


# ---------------------------
# Evaluation
# ---------------------------
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Eval:'

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target, num_classes=num_classes, focal_loss=True, dice_loss=True)

            output1 = output['out']
            confmat.update(target.flatten(), output1.argmax(1).flatten())
            metric_logger.update(loss=loss.item())

    return metric_logger.meters["loss"].global_avg, confmat


# ---------------------------
# Training one epoch
# ---------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=100, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, num_classes=2, focal_loss=True, dice_loss=True)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


# ---------------------------
# Learning rate scheduler
# ---------------------------
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
