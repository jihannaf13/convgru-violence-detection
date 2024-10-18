import time
import pytorch_lightning as pl
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import neptune
from neptune.types import File
from rich.progress import Progress
from sklearn.metrics import precision_recall_curve
from lightning.pytorch.loggers import NeptuneLogger
import os

from config.base_config import DEFAULT_NEPTUNE_CONFIG

class_names = ["Violence", "Non Violence"]

def initialize_neptune(status=True):
    if status:
        # Neptune Logger
        neptune_logger = NeptuneLogger(
            api_key=DEFAULT_NEPTUNE_CONFIG['api_key'],  # Accessing the token from environment variable
            project=DEFAULT_NEPTUNE_CONFIG['project'],  # replace with your Neptune project
            tags=["pytorch-lightning", "classification"]  # optional, for better organization
        )
    else:
        neptune_logger = None
    return neptune_logger
class TrainingTimeLogger(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.epoch_times = []

    def on_train_epoch_end(self, trainer, pl_module):
        current_time = time.time()
        epoch_time = current_time - self.start_time
        self.epoch_times.append(epoch_time)
        self.start_time = current_time
        trainer.logger.experiment["epoch_time"].log(epoch_time)

    def on_train_end(self, trainer, pl_module):
        total_time = sum(self.epoch_times)
        trainer.logger.experiment["total_training_time"].log(total_time)

def load_checkpoint(model, 
                    checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def save_concat_dataset_to_npz(dataset, 
                               output_file):
    """
    Save frames and labels from a concatenated dataset to a single .npz file.
    
    Args:
        dataset (ConcatDataset): Concatenated dataset to save.
        output_file (str): Path to the output .npz file.
    """
    all_frames = []
    all_labels = []
    
    with Progress() as progress:
        task = progress.add_task(
            "[green]Converting dataset to .npz...", 
            total=len(dataset)  
        )
        
        for idx in range(len(dataset)):
            frames, label = dataset[idx]
            all_frames.append(frames.numpy())
            all_labels.append(label.numpy())
            progress.advance(task)
    
    all_frames = np.array(all_frames)
    all_labels = np.array(all_labels)
    
    np.savez(
        output_file, 
        frames=all_frames, 
        labels=all_labels
    )

def find_optimal_threshold(predictions, true_labels):
    precisions, recalls, thresholds = precision_recall_curve(true_labels, 
                                                             predictions)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

    best_threshold = thresholds[np.argmax(f1_scores)]

    print(f'Best Threshold: {best_threshold}')
    
    return best_threshold

def save_concat_temp_npz(temp_dir, 
                    output_dir):
    # List of NPZ files
    temp_feature_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith("temp_features_") and f.endswith(".npz")])
    temp_label_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith("temp_labels_") and f.endswith(".npz")])

    all_features = []
    all_labels = []

    # Load and concatenate features
    for temp_feature_file in temp_feature_files:
        temp_data = np.load(temp_feature_file)
        all_features.append(torch.tensor(temp_data['frames']))
        temp_data.close()

    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)

    # Load and concatenate labels
    for temp_label_file in temp_label_files:
        temp_data = np.load(temp_label_file)
        all_labels.extend(temp_data['labels'])
        temp_data.close()

    all_labels = torch.tensor(all_labels)

    # Save the final concatenated features and labels
    np.savez(output_dir, frames=all_features.numpy(), labels=all_labels.numpy())
    print(f"Final concatenated dataset saved to {output_dir}")

    for temp_file in temp_feature_files + temp_label_files:
        os.remove(temp_file)
