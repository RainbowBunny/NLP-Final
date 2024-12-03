from __future__ import unicode_literals, print_function, division
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import numpy as np
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import logging.config
import logging
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_train(train_ds, model, optimizer, loss_function, batch_size, coef = 0, regularizer = None):
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 8)
    train_loss = []
    preds = []
    truth = []

    nbatch = len(train_loader)
    
    for batch_idx, (inputs, target) in enumerate(train_loader):        
        if (type(inputs) == dict):
            inputs = {key: value.to(device) for key, value in inputs.items()}
        else:
            inputs = inputs.to(device)
        target = target.to(device)
        
        output = model(inputs)
        classes = torch.flatten(output)

        if coef:    loss = loss_function(classes, target) + coef * regularizer(classes, target)
        else:       loss = loss_function(classes, target)

        train_loss.append(loss.item() / target.shape[0])

        output_classes = (classes >= 0.5).long()
        target_classed = (target >= 0.5).long()
        preds.extend(output_classes.cpu())
        truth.extend(target_classed.cpu())
        
        optimizer.zero_grad();  loss.backward()
        optimizer.step()
        
        if (batch_idx % 500 == 499):
            logging.info(f'    Batch {batch_idx + 1}/{nbatch}: cel = {train_loss[-1]}')

    return np.mean(train_loss), preds, truth

def run_eval(valid_ds, model, loss_function, batch_size = 32):
    valid_loader = DataLoader(valid_ds, batch_size = batch_size, shuffle = False, num_workers = 8)
    valid_loss = []
    preds = []
    trues = []
    
    with torch.no_grad():
        for inputs, target in valid_loader:
            if (type(inputs) == dict):
                inputs = {key: value.to(device) for key, value in inputs.items()}
            else:
                inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)
            classes = torch.flatten(output)
            loss = loss_function(classes, target)
            
            output_classes = (classes >= 0.5).long()
            target_classed = (target >= 0.5).long()
            preds.extend(output_classes.cpu())
            trues.extend(target_classed.cpu())

            valid_loss.append(loss.item() / target.shape[0])

    return np.mean(valid_loss), preds, trues

class CheckPointArgs:
    def __init__(self, model_name, experiment_name, checkpoint_dir = 'checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        self.saved_checkpoint = os.path.join(checkpoint_dir, f'{model_name}_{experiment_name}_checkpoint.pth')
        self.model_name = model_name
        self.experiment_name = experiment_name

class TrainArgs:
    def __init__(self, num_epochs = 100, learning_rate = 2e-5, batch_size = 32):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

class Trainer:
    def __init__(self,
        model,
        train_ds,
        valid_ds,
        checkpoint_args,
        training_args,
        logging_config = {'level': logging.INFO}
    ):
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        
        self.model = model.to(device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr = training_args.learning_rate, betas = (0.9, 0.999), weight_decay = 4e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 1, gamma = 0.9)
        
        self.checkpoint_args = checkpoint_args
        self.training_args = training_args

        self.min_cel = 1
        self.best_model = model

        logging.basicConfig(**logging_config)
        logging.captureWarnings(True)

        if os.path.exists(checkpoint_args.saved_checkpoint):
            loaded_checkpoint = torch.load(checkpoint_args.saved_checkpoint, map_location = device)

            self.model.load_state_dict(loaded_checkpoint['model'])
            self.best_model.load_state_dict(loaded_checkpoint['best_model'])
            
            self.train_cel = loaded_checkpoint['train_cel']
            self.valid_cel = loaded_checkpoint['valid_cel']

            self.min_cel = loaded_checkpoint['min_cel']
            self.train_precision = loaded_checkpoint['train_precision']
            self.train_recall = loaded_checkpoint['train_recall']
            self.train_accuracy = loaded_checkpoint['train_accuracy']
            self.train_f1 = loaded_checkpoint['train_f1']
            self.eval_precision = loaded_checkpoint['eval_precision']
            self.eval_recall = loaded_checkpoint['eval_recall']
            self.eval_accuracy = loaded_checkpoint['eval_accuracy']
            self.eval_f1 = loaded_checkpoint['eval_f1']
        else:    
            os.system('mkdir -p ' + checkpoint_args.checkpoint_dir)
            
            self.start_epoch = -1
            self.train_cel = []
            self.valid_cel = []
            self.train_precision = []
            self.train_recall = []
            self.train_accuracy = []
            self.train_f1 = []
            self.eval_precision = []
            self.eval_recall = []
            self.eval_accuracy = []
            self.eval_f1 = []
    
    def train(self):
        for _ in range(self.training_args.num_epochs):
            logging.info(f'Epoch {len(self.train_cel) + 1}: Start at {datetime.now()}')
            # run epochs:
            torch.cuda.empty_cache()
            
            self.model.train()
            cel, preds, truth = run_train(self.train_ds, self.model, self.optimizer, nn.CrossEntropyLoss(), self.training_args.batch_size)
            self.train_cel.append(cel)

            self.train_precision.append(precision_score(preds, truth))
            self.train_recall.append(recall_score(preds, truth))
            self.train_accuracy.append(accuracy_score(preds, truth))
            self.train_f1.append(f1_score(preds, truth))
            logging.info(f'>>  Train Precision: {self.train_precision[-1]}')
            logging.info(f'>>  Train Recall: {self.train_recall[-1]}')
            logging.info(f'>>  Train Accuracy: {self.train_accuracy[-1]}')
            logging.info(f'>>  Train F1: {self.train_f1[-1]}')

            self.scheduler.step()
            
            self.model.eval()
            cel, preds, truth = run_eval(self.valid_ds, self.model, nn.CrossEntropyLoss(), self.training_args.batch_size)
            
            self.eval_precision.append(precision_score(preds, truth))
            self.eval_recall.append(recall_score(preds, truth))
            self.eval_accuracy.append(accuracy_score(preds, truth))
            self.eval_f1.append(f1_score(preds, truth))
            logging.info(f'>>  Train Precision: {self.eval_precision[-1]}')
            logging.info(f'>>  Train Recall: {self.eval_recall[-1]}')
            logging.info(f'>>  Train Accuracy: {self.eval_accuracy[-1]}')
            logging.info(f'>>  Train F1: {self.eval_f1[-1]}')

            self.valid_cel.append(cel)
            
            # pocket algorithm
            if (self.min_cel > self.valid_cel[-1]):
                self.min_cel = self.valid_cel[-1] 
                self.best_model = self.model
                
            torch.save({
                'model': self.model.state_dict(),
                'train_cel': self.train_cel,
                'valid_cel': self.valid_cel,
                'min_cel': self.min_cel,
                'best_model': self.best_model.state_dict(),
                'train_precision': self.train_precision,
                'train_accuracy': self.train_accuracy,
                'train_recall': self.train_recall,
                'train_f1': self.train_f1,
                'eval_precision': self.eval_precision,
                'eval_accuracy': self.eval_accuracy,
                'eval_recall': self.eval_recall,
                'eval_f1': self.eval_f1
            }, self.checkpoint_args.saved_checkpoint)

            logging.info(f'>>  Eval cel = {self.valid_cel[-1]}')
            logging.info(f'Epoch {len(self.train_cel)}: Finish at {datetime.now()}')

if __name__ == '__main__':
    print(device)