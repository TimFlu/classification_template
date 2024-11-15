import sys
#TODO: Change the path to the working directory
sys.path.append('/storage/homefs/tf24s166/code/cifar10/') 
from utils.datasets import CustomDataset
from utils.log import comet_log_metrics

import os
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
#TODO: Import pretrained weights
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def metric_evaluation(labels, pred, binary=True, num_classes=None):
    # For AUC calculation
    if binary:
        # Binary AUC
        auc = roc_auc_score(labels, pred)
        # Convert predictions to binary (0 or 1) for metrics
        pred = [1 if i > 0.5 else 0 for i in pred]
    else:
        # Multiclass AUC: Check if all classes are represented
        if len(np.unique(labels)) < num_classes:
            auc = np.nan  # Handle missing classes
        else:
            # One-hot encode labels and predictions for AUC
            labels_one_hot = np.eye(num_classes)[labels]
            pred_one_hot = np.eye(num_classes)[np.argmax(pred, axis=1)]
            auc = roc_auc_score(labels_one_hot, pred_one_hot, multi_class='ovr')

        # Convert predictions to class indices for metrics
        pred = np.argmax(pred, axis=1)
        
    # Accuracy
    accuracy = accuracy_score(labels, pred)

    # Precision, Recall, F1-Score
    if binary:
        # For binary classification, no averaging needed
        precision = precision_score(labels, pred, zero_division=0)
        recall = recall_score(labels, pred, zero_division=0)
        f1 = f1_score(labels, pred, zero_division=0)
    else:
        # For multiclass, use macro-averaging (or adjust as needed)
        precision = precision_score(labels, pred, average='macro', zero_division=0)
        recall = recall_score(labels, pred, average='macro', zero_division=0)
        f1 = f1_score(labels, pred, average='macro', zero_division=0)

    return accuracy, precision, recall, f1, auc

def train_model(device, comet_logger, cfg):
    ### Create the dataset and loaders ###
    root = cfg.data.data_path

    # Define the transformations to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet50 requires 224x224 images
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(root, train=True, transform=transform)
    test_dataset = CustomDataset(root, train=False, transform=transform)

    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    
    ### Create and Load the pretrained Model ###
    output_classes = len(train_dataset.classes) if not train_dataset.binary else 1
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(2048, output_classes) 
    model.to(device)

    learning_rate = cfg.training.learning_rate
    weight_decay = cfg.training.weight_decay
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    class_weights = torch.tensor(train_dataset.class_weights).to(device).float()
    logger.info(f"Class weights: {class_weights}")
    if train_dataset.binary:
        loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    else:
        loss_fun = torch.nn.CrossEntropyLoss(weight=class_weights)

    
    # Train and Test
    for epoch in range(cfg.training.epochs):
        logger.info(f'-------- Epoch: {epoch+1}/{cfg.training.epochs} --------')

        # Store predictions and labels over the epoch
        epoch_labels = {'train': [], 'test': []}
        epoch_preds = {'train': [], 'test': []}

        # Store the loss and metrics over the epoch
        train_loss = []
        test_loss = []
        metric_names = ["{}_accuracy", "{}_precision", "{}_recall", "{}_f1", "{}_roc_auc"]

        # Train the model
        logger.info("Training the model ...")
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_fun(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # Calculate metrics
            labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()

            if train_dataset.binary:
                # Binary classification: apply sigmoid to get probabilities
                pred = F.sigmoid(pred).detach().cpu().numpy()
                epoch_preds['train'] += list(pred.reshape(-1))  # Flatten for binary case
            else:
                # Multiclass classification: apply softmax to get class probabilities
                pred = F.softmax(pred, dim=1).detach().cpu().numpy()
                epoch_preds['train'] += list(pred)  # Keep 2D structure for multiclass

            # Append labels
            epoch_labels['train'] += list(labels.reshape(-1))
            
            if i % 200 == 0:
                accuracy, precision, recall, f1, auc = metric_evaluation(epoch_labels['train'], epoch_preds['train'],
                                                                         binary=train_dataset.binary,
                                                                         num_classes=output_classes)
                logger.info(f"Training Batch {i+1}/{len(train_loader)}: Current Batch Loss: {loss.item()}")
                logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, roc_auc: {auc}")
                
        # After training all batches, calculate the metrics and log them
        accuracy, precision, recall, f1, auc = metric_evaluation(epoch_labels['train'], epoch_preds['train'],
                                                                         binary=train_dataset.binary,
                                                                         num_classes=output_classes)
        for metric_name, value in zip(metric_names, [accuracy, precision, recall, f1, auc]):
            comet_log_metrics(comet_logger, {metric_name.format('train'): value}, epoch, cfg)    

        # Test the model
        logger.info("Testing the model ...")
        model.eval()
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                pred = model(images)
                loss = loss_fun(pred, labels)
                test_loss.append(loss.item())
                
                # Calculate metrics
                labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()

                if train_dataset.binary:
                    # Binary classification: apply sigmoid to get probabilities
                    pred = F.sigmoid(pred).detach().cpu().numpy()
                    epoch_preds['test'] += list(pred.reshape(-1))  # Flatten for binary case
                else:
                    # Multiclass classification: apply softmax to get class probabilities
                    pred = F.softmax(pred, dim=1).detach().cpu().numpy()
                    epoch_preds['test'] += list(pred)  # Keep 2D structure for multiclass

                # Append labels
                epoch_labels['test'] += list(labels.reshape(-1))
                
                if i % 50 == 0:
                    accuracy, precision, recall, f1, auc = metric_evaluation(epoch_labels['test'], epoch_preds['test'],
                                                                            binary=train_dataset.binary,
                                                                            num_classes=output_classes)
                    logger.info(f"Testing Batch {i+1}/{len(train_loader)}: Current Batch Loss: {loss.item()}")
                    logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, roc_auc: {auc}")
        
        # After training all batches, calculate the metrics and log them
        accuracy, precision, recall, f1, auc = metric_evaluation(epoch_labels['test'], epoch_preds['test'],
                                                                         binary=train_dataset.binary,
                                                                         num_classes=output_classes)
        for metric_name, value in zip(metric_names, [accuracy, precision, recall, f1, auc]):
            comet_log_metrics(comet_logger, {metric_name.format('test'): value}, epoch, cfg)
        

        comet_log_metrics(comet_logger, {
                    "mean batch train_loss": np.mean(train_loss),
                    "mean batch test_loss": np.mean(test_loss)},
                    epoch, 
                    cfg)
        
            
        logger.info(f"Avg Batch Train / Test Loss: {np.mean(train_loss)} / {np.mean(test_loss)}")


        

        
