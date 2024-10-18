# Import Library
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import neptune
import numpy as np
from torchvision import transforms
from lightning.pytorch import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar, EarlyStopping
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

# Import Class
from dataset_factories.convert_dataset import load_dataset_train
from model_factories.cnn_convgru import PlCNNConvGRUClassifier
# from model_factories.pl_cnn_convlstm_3_2 import PlCNNConvLSTMClassifier
# With Print
# from model_factories.pl_cnn_convgru_3_2_print import PlCNNConvGRUClassifier
# from model_factories.pl_cnn_convlstm_3_2_print import PlCNNConvLSTMClassifier
from config.base_config import DEFAULT_DATASET_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG, DEFAULT_NEPTUNE_CONFIG, DEFAULT_HYPERPARAMS
from utils.utils import TrainingTimeLogger, find_optimal_threshold

from utils.utils import initialize_neptune

if __name__ == '__main__':
    seed_everything(42, workers=True)

    #Print all the information that used to dataset
    print("Dataset Information")
    for key, value in DEFAULT_DATASET_CONFIG.items():
        print(f'{key}: {value}')

    train_dir = DEFAULT_DATASET_CONFIG['train_cnn_npz_dir']
    val_dir = DEFAULT_DATASET_CONFIG['validation_cnn_npz_dir']

    # Load Train Dataset
    train_frames, train_labels = load_dataset_train(train_dir)
    
    # Load Validation Dataset
    val_frames, val_labels = load_dataset_train(val_dir)

    model_type = "convgru"

    if(model_type == "convgru"):
        # CNN + ConvGRU
        print("Model: ConvGRU")
        model = PlCNNConvGRUClassifier(
            DEFAULT_MODEL_CONFIG['input_size'], 
            DEFAULT_MODEL_CONFIG['hidden_sizes'], 
            DEFAULT_MODEL_CONFIG['kernel_sizes'], 
            DEFAULT_MODEL_CONFIG['n_layers'], 
            DEFAULT_MODEL_CONFIG['num_classes'],
            batch_size=DEFAULT_DATASET_CONFIG['batch_size'],
            num_workers=DEFAULT_DATASET_CONFIG['num_workers'],
            train_frames=train_frames,
            train_labels=train_labels,
            val_frames=val_frames,
            val_labels=val_labels,
            learning_rate=DEFAULT_TRAINING_CONFIG['learning_rate'],
            weight_decay=DEFAULT_TRAINING_CONFIG['weight_decay'],
            use_amp=True,
            use_scan=DEFAULT_TRAINING_CONFIG['use_scan'],   
            )
    # else:
    #     # CNN + ConvLSTM
    #     print("Model: ConvLTSM")
    #     model = PlCNNConvLSTMClassifier(
    #         DEFAULT_MODEL_CONFIG['input_size'], 
    #         DEFAULT_MODEL_CONFIG['hidden_sizes'], 
    #         DEFAULT_MODEL_CONFIG['kernel_sizes'], 
    #         DEFAULT_MODEL_CONFIG['n_layers'], 
    #         DEFAULT_MODEL_CONFIG['num_classes'],
    #         batch_size=DEFAULT_DATASET_CONFIG['batch_size'],
    #         num_workers=DEFAULT_DATASET_CONFIG['num_workers'],
    #         train_frames=train_frames,
    #         train_labels=train_labels,
    #         val_frames=val_frames,
    #         val_labels=val_labels,
    #         learning_rate=DEFAULT_TRAINING_CONFIG['learning_rate'],
    #         weight_decay=DEFAULT_TRAINING_CONFIG['weight_decay'],
    #         use_amp=True,
    #         use_scan=DEFAULT_TRAINING_CONFIG['use_scan'],
    #         )
        
    # Set the datasets separately
    model.set_datasets(
        train_frames=train_frames,
        train_labels=train_labels,
        val_frames=val_frames,
        val_labels=val_labels,
        test_frames=None,
        test_labels=None
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc', 
        save_top_k=1, 
        mode='max'
        )
    
    # Define early stopping callback (Not Used)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        min_delta=0.00,      # Minimum change to qualify as an improvement
        patience=5,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # 'min' for loss, 'max' for accuracy
    )
    
    # Learning Rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    rich_progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )
    
    # Training Time
    training_time_logger = TrainingTimeLogger()

    #Change to True to initialize neptune
    neptune_logger = initialize_neptune(True)
    if neptune_logger is not None:
        neptune_logger.log_hyperparams(DEFAULT_HYPERPARAMS)
        current_run_version = neptune_logger.version
    
    # Training
    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=DEFAULT_TRAINING_CONFIG['num_epochs'],
        logger=neptune_logger,
        callbacks=[
            training_time_logger,
            checkpoint_callback,
            lr_monitor,
            rich_progress_bar,
            # early_stop_callback,
        ],
        precision='16-mixed' if model.use_amp else 32,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        deterministic=True,
        log_every_n_steps=5,
        )

    # Fit the model
    trainer.fit(model)

    # Find the optimal threshold based on validation data
    best_threshold = find_optimal_threshold(model.validation_predictions, 
                                            model.validation_true_labels)

    # Append the best threshold to the existing text file
    with open(DEFAULT_TRAINING_CONFIG["best_threshold_path"], 'a') as f:
        f.write(f'{current_run_version}: {best_threshold}\n')

    neptune_logger.log_hyperparams({"best_threshold": best_threshold})

    # Save model
    # torch.save(model,
    #            DEFAULT_TRAINING_CONFIG['saved_model_path']+f"model_{current_run_version}.pth")
      
    # Save state dict 
    torch.save(model.state_dict(),
               DEFAULT_TRAINING_CONFIG['saved_model_path']+f"model_{current_run_version}_state_dict.pth")
    
    print("Finish Training")