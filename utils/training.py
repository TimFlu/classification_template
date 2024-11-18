import sys
#TODO: Change the path to the working directory
sys.path.append('/storage/homefs/tf24s166/code/cifar10/') 
from utils.datasets import CustomDataset
from utils.log import comet_log_metrics, comet_log_figure
from utils.plots import plot_confidence_histogram, plot_reliability_diagram
from utils.utils import metric_evaluation, EarlyStopping, expected_calibration_error
from utils.models import TemperatureScaling
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


def train_temperature_scaling(model, val_loader, device, comet_logger, cfg):
    model.eval()
    temperature_module = TemperatureScaling(model).to(device)
    optimizer = torch.optim.LBFGS([temperature_module.temperature], lr=0.01, max_iter=50)
    loss_fun = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad
        logits_list, labels_list = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                logits_list.append(logits)
                labels_list.append(labels)
        
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        scaled_logits = temperature_module(logits)
        loss = loss_fun(scaled_logits, labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)

    print(f"Optimal temperature: {temperature_module.temperature.item():.4f}")
    return temperature_module


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
    if cfg.model.use_pretrained:
        model.load_state_dict(torch.load(cfg.model.path_to_weights))
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

    ### Initialize Early Stopping ###
    if cfg.early_stopping.initialize:
        early_stopper = EarlyStopping(patience=cfg.early_stopping.patience, min_delta=cfg.early_stopping.min_delta)


    ### Train and Test ###
    best_test_loss = 10e10
    for epoch in range(1, 1+cfg.training.epochs):
        if cfg.model.use_pretrained:
            logger.info('Using pretrained model, skipping training and testing...')
            break
        logger.info(f'-------- Epoch: {epoch}/{cfg.training.epochs} --------')

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
            if train_dataset.binary:
                labels = labels.reshape(-1, 1).float()
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
                pred = F.sigmoid(pred).detach().cpu().numpy().reshape(-1)
                epoch_preds['train'] += list(pred)  # Flatten for binary case
            else:
                # Multiclass classification: apply softmax to get class probabilities
                pred = F.softmax(pred, dim=1).detach().cpu().numpy()
                epoch_preds['train'] += list(pred)  # Keep 2D structure for multiclass

            # Append labels
            epoch_labels['train'] += list(labels.reshape(-1))
            
            if i % 200 == 0:
                accuracy, precision, recall, f1, auc = metric_evaluation(list(labels.reshape(-1)), list(pred),
                                                                         binary=train_dataset.binary,
                                                                         num_classes=output_classes)
                logger.info(f"Training Batch {i+1}/{len(train_loader)}: Current Batch Loss: {loss.item()}")
                logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, roc_auc: {auc:.3f}")
                
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
            if train_dataset.binary:
                labels = labels.reshape(-1, 1).float()
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
                    pred = F.sigmoid(pred).detach().cpu().numpy().reshape(-1)
                    epoch_preds['test'] += list(pred)  # Flatten for binary case
                else:
                    # Multiclass classification: apply softmax to get class probabilities
                    pred = F.softmax(pred, dim=1).detach().cpu().numpy()
                    epoch_preds['test'] += list(pred)  # Keep 2D structure for multiclass

                # Append labels
                epoch_labels['test'] += list(labels.reshape(-1))
                
                if i % 200 == 0:
                    accuracy, precision, recall, f1, auc = metric_evaluation(list(labels.reshape(-1)), list(pred),
                                                                            binary=train_dataset.binary,
                                                                            num_classes=output_classes)
                    logger.info(f"Testing Batch {i+1}/{len(train_loader)}: Current Batch Loss: {loss.item()}")
                    logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, roc_auc: {auc:.3f}")

        
        # After testing all batches, calculate the metrics and log them
        accuracy, precision, recall, f1, auc = metric_evaluation(epoch_labels['test'], epoch_preds['test'],
                                                                         binary=train_dataset.binary,
                                                                         num_classes=output_classes)
        for metric_name, value in zip(metric_names, [accuracy, precision, recall, f1, auc]):
            comet_log_metrics(comet_logger, {metric_name.format('test'): value}, epoch, cfg)
        

        ### After Test and Training evaluations ###
        fig_conf_hist = plot_confidence_histogram(epoch_labels['test'], epoch_preds['test'], save_path=os.getcwd(), num_bins=10)
        comet_log_figure(comet_logger, fig_conf_hist, 'confidence_histogram', epoch, cfg)
        fig_reliability = plot_reliability_diagram(epoch_labels['test'], epoch_preds['test'], save_path=os.getcwd(), num_bins=10)
        comet_log_figure(comet_logger, fig_reliability, 'reliability_diagram', epoch, cfg)

        comet_log_metrics(comet_logger, {"mean batch train_loss": np.mean(train_loss),
                    "mean batch test_loss": np.mean(test_loss)}, epoch, cfg)
        comet_log_metrics(comet_logger, {"Expected Calibration Error": expected_calibration_error(epoch_labels['test'], epoch_preds['test'])}, 
                          epoch, cfg)


        # Check if new best model
        if np.mean(test_loss) < best_test_loss:
            logger.info('New best model, saving ...')
            best_test_loss = np.mean(test_loss)
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'best_model.pth'))

        # Log the results
        logger.info(f"Avg Batch Train / Test Loss: {np.mean(train_loss)} / {np.mean(test_loss)}")
        logger.info(f"Test Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, roc_auc: {auc:.3f}")

        # Early stopping
        if cfg.early_stopping.initialize:
            early_stopper(np.mean(test_loss))
            if early_stopper.early_stop:
                break

        
    # Calibrate the model
    calibration_model = train_temperature_scaling(model, test_loader, device, comet_logger, cfg)
    calibrated_preds_list = []
    epoch_test_labels = []
    for images, labels in test_loader:
        inputs = images.to(device)
        calibrated_preds = calibration_model.predict(inputs)
        epoch_test_labels += list(labels.reshape(-1))
        if train_dataset.binary:
            # Binary classification: apply sigmoid to get probabilities
            calibrated_preds = list(F.sigmoid(calibrated_preds).detach().cpu().numpy().reshape(-1))
            calibrated_preds_list += calibrated_preds
        else:
            # Multiclass classification: apply softmax to get class probabilities
            calibrated_preds = list(F.softmax(calibrated_preds, dim=1).detach().cpu().numpy())
            calibrated_preds_list += calibrated_preds

    fig_conf_hist = plot_confidence_histogram(epoch_test_labels, calibrated_preds_list, save_path=os.getcwd(), num_bins=10)
    comet_log_figure(comet_logger, fig_conf_hist, 'confidence_histogram_calibrated', epoch, cfg)
    fig_reliability = plot_reliability_diagram(epoch_test_labels, calibrated_preds_list, save_path=os.getcwd(), num_bins=10)
    comet_log_figure(comet_logger, fig_reliability, 'reliability_diagram_calibrated', epoch, cfg)
