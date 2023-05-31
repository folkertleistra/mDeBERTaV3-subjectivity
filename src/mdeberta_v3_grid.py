#!/bin/bash
# This file contains all the code that was used for running our grid searches.
# Grid Searches were performed using Weights and Biases (https://wandb.ai/site).
# If you'd like to replicate these results, you need to create your own WandB account and set it up in this file.

# Importing all libraries
import datetime
import os
import random as python_random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import wandb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (AdamW, AutoModelForSequenceClassification,
                          AutoTokenizer, BertForSequenceClassification,
                          get_linear_schedule_with_warmup)

### This script only works on CUDA devices. ###
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'CUDA device found: {torch.cuda.get_device_name(0)}')

else:
    print('No GPU available, exiting...')
    sys.exit(1)

### Setting up the seed, 42 was used to obtain our fine-tuned models ###
seed = 42

np.random.seed(seed)
tf.random.set_seed(seed)
python_random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

### Below are all constants and hyper-parameters that we experimented with ###
model_to_use = 'microsoft/mdeberta-v3-base'     # HuggingFace model
max_length = 40                                 # Maximum tokenization length - we only used 40

# WandB specific
grid_iterations = 25                            # The amount of iterations for which to perform a grid search
wand_project_name = '<YOUR_PROJECT_NAME>'       # The name of the project on which the data will be stored on WandB

# Grid values that can be randomly selected from
weight_decay_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
warmup_step_values = [100, 200, 300, 400, 500]
learning_rate_values = [2e-5, 3e-5, 4e-5, 5e-5, 6e-5]
batch_size_values = [16, 32, 64]
epoch_values = [2, 3, 4, 5, 6, 7, 8]

### Loading the data ###
train_path = 'data/multilingual_adapted.tsv'
dev_path = '<path/to/validation/data>'          # Add your own validation data here, can be found on CLEF 2023 GitLab.

print(f'Loading train data: {train_path} \n')
df_train = pd.read_csv(train_path, sep='\t')
print(f'Loading dev data: {dev_path} \n')
df_dev = pd.read_csv(dev_path, sep='\t')

### Initializing tokenizer ###
tokenizer = AutoTokenizer.from_pretrained(model_to_use)

### Convert string labels to numeric using a LabelBinarizer ###
encoder = LabelBinarizer()
Y_train_bin = encoder.fit_transform(df_train['label'].to_list())
Y_dev_bin = encoder.fit_transform(df_dev['label'].to_list())

Y_train_bin = [i[0] for i in Y_train_bin]
Y_dev_bin = [i[0] for i in Y_dev_bin]

def tok_data(sentences):
    """
    Function that for each sentence in a list tokenizes the sentence to BERT specific format.
    Tokenizer and settings that are defined above are used
    :param sentences: list of sentences
    :return: Two lists containing the input ids and attention masks
    """
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                          sent,
                          add_special_tokens = True,
                          max_length = max_length,
                          padding='max_length',
                          truncation=True,
                          return_attention_mask = True,
                          return_tensors = 'pt'
                    )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

### Train data tokenization ###
train_labels = torch.tensor(Y_train_bin)
train_input_ids, train_attention_masks = tok_data(df_train['sentence'].to_list())

### Dev data tokenization ###
dev_labels = torch.tensor(Y_dev_bin)
dev_input_ids, dev_attention_masks = tok_data(df_dev['sentence'].to_list())

def create_dataloader(input_ids, attention_masks, labels, bs):
    """
    Creates a TensorDataset for the input_ids, attention_masks and labels.
    The TensorDataset is placed into a DataLoader using a RANDOM sampler. This means the data gets shuffled.
    """
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=bs)

def create_optimizer_and_scheduler(lr, ws, len_td, model_params, epochs, wd):
    """
    Creates the optimizer and scheduler to be used during training.
    We only experimented with the AdamW optimizer.
    The learning rate (lr), Weight Decay (wd), epochs and Warmup Steps (ws) can be adjusted
    """
    optimizer = AdamW(model_params,
                    lr=lr,
                    weight_decay=wd)

    total_steps = len_td * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps = ws,
                                              num_training_steps = total_steps)

    return optimizer, scheduler

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def get_best_val_stats(stats):
    """
    Function that will look throught the collected stats and determine which accuracy was the highest and
    which f1 and accuracy were the best. The BEST meaning: obtained in the epoch with the lowest validation loss.
    :param stats: training stats that were collected during training
    :return: highest acc score, epoch in which the highest acc was obtained, best acc score, lowest validation loss,
    epoch in which the lowest validation loss was obtained, best f1 score
    """
    highest_acc = 0
    highest_acc_epoch = 0
    min_val_loss = stats[0]['Valid. Loss']
    min_val_loss_epoch = 0
    best_val_acc = 0
    best_val_f1 = 0
    for stat in stats:
        if stat['Valid. Accur.'] >= highest_acc:
            highest_acc = stat['Valid. Accur.']
            highest_acc_epoch = stat['epoch']

        if stat['Valid. Loss'] <= min_val_loss:
            best_val_acc = stat['Valid. Accur.']
            best_val_f1 = stat['Valid. Macro F1']
            min_val_loss = stat['Valid. Loss']
            min_val_loss_epoch = stat['epoch']

    return highest_acc, highest_acc_epoch, best_val_acc, min_val_loss, min_val_loss_epoch, best_val_f1


def print_cr(labels, predictions):
    """
    Function that prints the Sklearn classification report for gold labels and logits
    :param labels: gold labels as a list
    :param predictions: logits generated by the BERT model
    :return: classification report as a dictionary
    """
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(labels, axis=0)

    cr = classification_report(flat_true_labels, flat_predictions, digits=4)
    print(cr)
    return classification_report(flat_true_labels, flat_predictions, digits=4, output_dict=True)


def wanb_train():
    """
    Main function that performs a grid search using WandB and trains the model
    :return:
    """
    predictions, true_labels = defaultdict(list), defaultdict(list)
    val_losses = {}

    # We'll store a number of quantities such as training and validation loss, validation accuracy, and timings.
    training_stats = []
    total_t0 = time.time()
    with wandb.init(config=sweep_config):
        config = wandb.config
        print(f'Config that we will be using: {config}')

        model = AutoModelForSequenceClassification.from_pretrained(
            model_to_use,
            num_labels = 2,
            output_attentions = False,
            output_hidden_states = False,
            )
        model.cuda()
        model.resize_token_embeddings(len(tokenizer))

        # Create dataloaders
        train_dataloader = create_dataloader(train_input_ids, train_attention_masks, train_labels, config.batch_size)
        dev_dataloader = create_dataloader(dev_input_ids, dev_attention_masks, dev_labels, config.batch_size)
        optimizer, scheduler = create_optimizer_and_scheduler(config.lr, 0, len(train_dataloader),
                                                              model.parameters(), config.epochs, config.wd)

        for epoch_i in range(config.epochs):

            print('======== Epoch {:} / {:} ========'.format(epoch_i, config.epochs - 1))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0
            model.train()

            for step, batch in enumerate(train_dataloader):

                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)

                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()

                result = model(b_input_ids,
                              attention_mask=b_input_mask,
                              labels=b_labels,
                              return_dict=True)

                loss = result.loss
                total_train_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = format_time(time.time() - t0)

            print("  Average training loss: {0:.3f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            print("Running Validation...")

            t0 = time.time()

            model.eval()

            # Tracking variables
            total_eval_loss = 0

            for batch in dev_dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                with torch.no_grad():
                    result = model(b_input_ids,
                                  token_type_ids=None,
                                  attention_mask=b_input_mask,
                                  labels=b_labels,
                                  return_dict=True)

                loss = result.loss
                logits = result.logits

                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                true_labels[epoch_i].append(label_ids)
                predictions[epoch_i].append(logits)

            # Printing the cr
            val_cr = print_cr(true_labels[epoch_i], predictions[epoch_i])
            cr_val_f1 = val_cr['macro avg']['f1-score']
            cr_val_acc = val_cr['accuracy']
            print("CR Accuracy: {0:.3f}".format(cr_val_acc))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(dev_dataloader)
            val_losses[epoch_i] = avg_val_loss
            validation_time = format_time(time.time() - t0)
            print("  Validation Loss: {0:.3f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                    {
                    'epoch': epoch_i,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': cr_val_acc,
                    'Valid. Macro F1': cr_val_f1,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                    }
                )
        best_val_loss_epoch = min(val_losses, key=val_losses.get)
        lowest_val = val_losses[best_val_loss_epoch]

        print_cr(true_labels[best_val_loss_epoch], predictions[best_val_loss_epoch])

        highest_val_acc, highest_val_acc_epoch, best_val_acc, min_val_loss, min_val_epoch, best_val_f1 = \
            get_best_val_stats(training_stats)

        # Log the metric that need to be tracked to WandB
        wandb.log({"min_val_loss": lowest_val, 'epoch_loss': best_val_loss_epoch + 1})
        wandb.log({'max_val_acc': highest_val_acc, 'epoch_acc': highest_val_acc_epoch + 1})
        wandb.log({'best_val_loss_acc': best_val_acc, 'best_val_loss_epoch': min_val_epoch + 1})
        wandb.log({'best_val_f1': best_val_f1})

        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

sweep_config = {'method': 'random'}

parameters_dict = {
    'epochs': {
        'values' : epoch_values
    },
    'batch_size': {
      'values': batch_size_values
    },
    'lr': {
        'values': learning_rate_values
    },
    'warmup_steps': {
        'values': warmup_step_values
    },
    'wd': {
        'values': weight_decay_values
    }
    }

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project=wand_project_name)

wandb.agent(sweep_id, wanb_train, count=grid_iterations)