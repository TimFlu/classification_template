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
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
#TODO: Import pretrained weights
from torchvision.models import ResNet50_Weights


def train_temperature_scaling(model, val_loader, device, comet_logger, cfg):
    model.eval()
    temperature_module = TemperatureScaling(model).to(device)
    # optimizer = torch.optim.LBFGS([temperature_module.temperature], lr=0.01, max_iter=30)
    optimizer = torch.optim.Adam([temperature_module.temperature], lr=0.01)
    loss_fun = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()

        total_loss = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            #logits = model(images)
            logits = checkpoint(model, images, use_reentrant=False)
            logits = temperature_module(logits)
            loss = loss_fun(logits, labels)
            total_loss += loss

        total_loss.backward()
        return total_loss
    
    optimizer.step(closure)

    logger.info(f"Optimal temperature: {temperature_module.temperature.item():.4f}")
    comet_log_metrics(comet_logger, {"Optimal Temperature": temperature_module.temperature.item()}, 0, cfg)

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
    output_classes = len(train_dataset.class_weights)
    classification_case = train_dataset.case
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(2048, output_classes) 
    if cfg.model.use_pretrained:
        model.load_state_dict(torch.load(cfg.model.path_to_weights, weights_only=True))
    model.to(device)

    learning_rate = cfg.training.learning_rate
    weight_decay = cfg.training.weight_decay
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    class_weights = torch.tensor(train_dataset.class_weights).to(device).float()
    logger.info(f"Class weights: {class_weights}")

    if train_dataset.binary or train_dataset.labels.ndim != 1: # Binary- and multi-label classification
        loss_fun = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        activ_func = F.sigmoid
    else:
        loss_fun = torch.nn.CrossEntropyLoss(weight=class_weights) # Multi-class classification
        activ_func = lambda x: F.softmax(x, dim=1) #TODO might need to specify the dim
    logger.info(f'Using activation function: {activ_func.__name__}')

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
            pred = activ_func(pred).detach().cpu().numpy()
            epoch_preds['train'].append(pred) 
            epoch_labels['train'].append(labels)
            
            if i % 400 == 0:
                accuracy, precision, recall, f1, auc = metric_evaluation(labels, pred, classification_case)
                logger.info(f"Training Batch {i+1}/{len(train_loader)}: Current Batch Loss: {loss.item()}")
                logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, roc_auc: {auc:.3f}")
                
        # After training all batches, calculate the metrics and log them
        epoch_preds['train'] = np.concatenate(epoch_preds['train'], axis=0)
        epoch_labels['train'] = np.concatenate(epoch_labels['train'], axis=0)
        accuracy, precision, recall, f1, auc = metric_evaluation(epoch_labels['train'], epoch_preds['train'], classification_case)
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
                pred = activ_func(pred).detach().cpu().numpy()
                epoch_preds['test'].append(pred) 
                epoch_labels['test'].append(labels)
                
                if i % 200 == 0:
                    accuracy, precision, recall, f1, auc = metric_evaluation(labels, pred, classification_case)
                    logger.info(f"Testing Batch {i+1}/{len(test_loader)}: Current Batch Loss: {loss.item()}")
                    logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, roc_auc: {auc:.3f}")

        # After testing all batches, calculate the metrics and log them
        epoch_preds['test'] = np.concatenate(epoch_preds['test'], axis=0)
        epoch_labels['test'] = np.concatenate(epoch_labels['test'], axis=0)
        accuracy, precision, recall, f1, auc = metric_evaluation(epoch_labels['test'], epoch_preds['test'], classification_case)
        for metric_name, value in zip(metric_names, [accuracy, precision, recall, f1, auc]):
            comet_log_metrics(comet_logger, {metric_name.format('test'): value}, epoch, cfg)
        

        ### After Test and Training evaluations ###
        plot_confidence_histogram(epoch_labels['test'], epoch_preds['test'], save_path=os.getcwd(),
                                                  cfg=cfg, step=epoch, logger=comet_logger, case=classification_case ,num_bins=cfg.calibration.num_bins)
        plot_reliability_diagram(epoch_labels['test'], epoch_preds['test'], save_path=os.getcwd(), 
                                 cfg=cfg, step=epoch, logger=comet_logger, case=classification_case, num_bins=cfg.calibration.num_bins)

        comet_log_metrics(comet_logger, {"mean batch train_loss": np.mean(train_loss),
                    "mean batch test_loss": np.mean(test_loss)}, epoch, cfg)
        comet_log_metrics(comet_logger, {"Expected Calibration Error": expected_calibration_error(epoch_labels['test'], epoch_preds['test'], classification_case, num_bins=cfg.calibration.num_bins)}, 
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

        
    ### Calibrate the model ###
    logger.info("Calibrating the model ...")
    calibration_model = train_temperature_scaling(model, test_loader, device, comet_logger, cfg)
    calibrated_preds_list = []
    epoch_test_labels = epoch_labels['test']
    #epoch_test_preds = epoch_preds['test']
    for images, labels in test_loader:
        inputs = images.to(device)
        calibrated_preds = calibration_model.predict(inputs)
        calibrated_preds_list.append(activ_func(calibrated_preds).detach().cpu().numpy())
    
    calibrated_preds_list = np.concatenate(calibrated_preds_list, axis=0)
    # Log calibration figures for calibrated and non calibrated outputs
    plot_confidence_histogram(epoch_test_labels, calibrated_preds_list, save_path=os.getcwd(),
                              cfg=cfg, step=epoch, logger=comet_logger, case=classification_case, cal=True, num_bins=cfg.calibration.num_bins)
    plot_reliability_diagram(epoch_test_labels, calibrated_preds_list, save_path=os.getcwd(), 
                             cfg=cfg, step=epoch, logger=comet_logger, case=classification_case, cal=True, num_bins=cfg.calibration.num_bins)

    # if cfg.model.use_pretrained:
    #     fig_conf_hist = plot_confidence_histogram(epoch_test_labels, preds_list, save_path=os.getcwd(), num_bins=cfg.calibration.num_bins)
    #     fig_reliability = plot_reliability_diagram(epoch_test_labels, preds_list, save_path=os.getcwd(), num_bins=cfg.calibration.num_bins)

