# Import library
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchmetrics.classification
import numpy as np
import matplotlib.pyplot as plt

# Import Class
from model_factories.convgru import ConvGRU
# from att_scan import ScanAttentionUpdated
from attention_mechanism.att_scan import ScanAttentionUpdated
from config.base_config import DEFAULT_TRAINING_CONFIG, DEFAULT_DATASET_CONFIG, DEFAULT_MODEL_CONFIG

# Directory
dropout = DEFAULT_TRAINING_CONFIG['dropout']
num_frames = DEFAULT_DATASET_CONFIG['num_frames']
ratio_cbam = DEFAULT_MODEL_CONFIG['ratio_cbam']
kernel_size_cbam = DEFAULT_MODEL_CONFIG['kernel_size_cbam']
ratio_scan = DEFAULT_MODEL_CONFIG['ratio_scan']

torch.autograd.set_detect_anomaly(True)

class CNNDataset(Dataset):
    def __init__(self, frames, labels, num_frames):
        self.frames = frames
        self.labels = labels
        self.num_frames = num_frames
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        return frame, label
    
class PlCNNConvGRUClassifier(pl.LightningModule):
    """
    CNN-ConvGRU Classifier model.

    This module combines a CNN model for feature extraction with a ConvGRU for temporal processing
    and classification.

    Attributes:
    - cnn (torch.nn.Module): CNN model for feature extraction.
    - convgru (ConvGRU): ConvGRU model for temporal processing.
    - fc (torch.nn.Linear): Fully connected layer for classification.

    Methods:
    - __init__: Initializes CNNConvGRUClassifier with provided models and parameters.
    - forward: Performs forward pass to classify input tensor.
    """
    def __init__(
            self, 
            convgru_input_size, 
            hidden_sizes, 
            kernel_sizes, 
            n_layers, 
            num_classes, 
            batch_size=None, 
            num_workers=None,
            train_frames=None, 
            train_labels=None, 
            val_frames=None, 
            val_labels=None, 
            test_frames=None, 
            test_labels=None,
            learning_rate=None,
            weight_decay=None,
            use_amp=False,
            use_scan=False,
            ):
        """
        Initializes CNNConvGRUClassifier with provided models and parameters.

        Args:
        - cnn_model (torch.nn.Module): Pre-trained CNN model for feature extraction.
        - convgru_input_size (int): Input size for ConvGRU.
        - hidden_sizes (list): List of hidden sizes for ConvGRU layers.
        - kernel_sizes (list): List of kernel sizes for ConvGRU layers.
        - n_layers (int): Number of layers in ConvGRU.
        - num_classes (int): Number of output classes for classification.
        """
        super().__init__()
        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=["train_frames", 
                                          "train_labels", 
                                          "val_frames", 
                                          "val_labels", 
                                          "test_frames", 
                                          "test_labels"])
        self.num_classes = num_classes

        # Initialize Spatial Attention with Channel Attention before ConvGRU
        self.spatial_attention = ScanAttentionUpdated(convgru_input_size,
                                                      ratio_scan) if use_scan else None
        self.convgru = ConvGRU(
            convgru_input_size, 
            hidden_sizes, 
            kernel_sizes, 
            n_layers
            )

        self.batch_norm = nn.BatchNorm2d(hidden_sizes[-1])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            hidden_sizes[-1], 
            self.num_classes
            )
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_amp = use_amp

        # Create datasets from provided frames and labels
        self.train_dataset = CNNDataset(train_frames, train_labels, num_frames)
        self.val_dataset = CNNDataset(val_frames, val_labels, num_frames)
        self.test_dataset = CNNDataset(test_frames, test_labels, num_frames)

        self.train_accuracy = torchmetrics.classification.Accuracy(
            task='multiclass', 
            num_classes=self.num_classes
            )
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task='multiclass', 
            num_classes=self.num_classes
            )
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        # Initialize buffers for validation predictions and true labels
        self.validation_predictions = []
        self.validation_true_labels = []

    def forward(self, x):
        """
        Performs forward pass to classify input tensor.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width).

        Returns:
        - out (torch.Tensor): Output tensor of shape (batch_size, num_classes).
        """

        batch_size, num_frames, features, _, _ = x.shape

        # Initialize hidden state for ConvGRU
        hidden = None
        self.attention_maps = []  # Clear attention maps for each forward pass
        # Process features through ConvGRU
        for i in range(num_frames):
            frame = x[:, i, :, :, :]
            if self.spatial_attention:
                # Initialize hidden state if it's None (typically in the first iteration)
                frame, attention_map = self.spatial_attention(x[:, i, :, :, :], 
                                                              hidden, 
                                                              return_attention = True)
                self.attention_maps.append(attention_map)  # Store attention map
            
            #ConvGRU
            hidden = self.convgru(frame, hidden)

        print(f'Attention map len: {len(self.attention_maps)}')

        # Extract final hidden state and perform pooling
        final_hidden_state = hidden[-1]
        
        #Batch Normalization
        normalized = self.batch_norm(final_hidden_state)
        
        #Drop out
        dropped_out = self.dropout(normalized)
        
        #Average Pooling
        pooled = torch.mean(dropped_out, dim=(2, 3))
        
        # Classify pooled features
        out = self.fc(pooled)
        return out
    
    def visualize_attention(self, frame_index):
        """
        Visualizes the attention map on the specified frame index.

        Args:
        - frame_index (int): The index of the frame in the sequence.
        """
        if not self.attention_maps:
            print("No attention maps available. Please run a forward pass first")
            return

        attention_map = self.attention_maps[frame_index].cpu().numpy()

        # Remove singleton dimensions (e.g., batch size dimension)
        if attention_map.shape[0] == 1:
            attention_map = attention_map.squeeze(0)  # Remove the batch dimension
        
        # If the attention map has multiple channels, aggregate them
        if attention_map.ndim == 3:  # Shape is now (2048, 8, 8)
            attention_map = attention_map.mean(axis=0)  # Aggregate across channels

        # Normalize the attention map for better visualization
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Visualize the attention map
        plt.imshow(attention_map, cmap='jet', alpha=0.5)
        plt.colorbar()
        plt.title(f"Aggregated Attention Map for Frame {frame_index}")
        plt.show()

    def training_step(self, batch, batch_idx):
        frames, labels = batch
        outputs = self(frames)
        loss = self.criterion(outputs, labels)
        self.train_loss(loss)
        self.train_accuracy(outputs, labels)
        self.log(
            'train_loss', 
            self.train_loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
            )
        self.log(
            'train_acc', 
            self.train_accuracy, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
            )
        return loss

    def validation_step(self, 
                        batch, 
                        batch_idx):
        frames, labels = batch
        outputs = self(frames)
        loss = self.criterion(outputs, labels)
        self.val_loss(loss)
        self.val_accuracy(outputs, labels)
        self.log(
            'val_loss', 
            self.val_loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
            )
        self.log(
            'val_acc', 
            self.val_accuracy, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
            )
        
        # Save predictions and true labels
        self.validation_predictions.append(outputs.softmax(dim=1)[:, 1].cpu().numpy())  # Save probabilities of class 1
        self.validation_true_labels.append(labels.cpu().numpy())

        return loss
    
    def on_validation_epoch_start(self):
        # Reset the buffers at the start of each validation epoch
        self.validation_predictions = []
        self.validation_true_labels = []
    
    def on_validation_epoch_end(self):
        # Flatten lists of predictions and true labels
        self.validation_predictions = np.concatenate(self.validation_predictions)
        self.validation_true_labels = np.concatenate(self.validation_true_labels)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.hparams.learning_rate,
                                          weight_decay=self.hparams.weight_decay)
        return self.optimizer
    
    # def configure_optimizers(self):
    #     self.optimizer = torch.optim.RMSprop(self.parameters(),
    #                                         lr=self.hparams.learning_rate,
    #                                         weight_decay=self.hparams.weight_decay)
    #     return self.optimizer

    
    # def configure_optimizers(self):
    #     self.optimizer = torch.optim.SGD(self.parameters(),
    #                                      lr=self.hparams.learning_rate,
    #                                      momentum=0.9,  # You can adjust the momentum value as needed
    #                                      weight_decay=self.hparams.weight_decay)
    #     return self.optimizer

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            persistent_workers=True, 
            pin_memory=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            persistent_workers=True, 
            pin_memory=True
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            persistent_workers=True, 
            pin_memory=True
            ) 

    def set_datasets(self, train_frames, train_labels, val_frames, val_labels, test_frames=None, test_labels=None):
        num_frames = DEFAULT_DATASET_CONFIG['num_frames']
        self.train_dataset = CNNDataset(train_frames, train_labels, num_frames)
        self.val_dataset = CNNDataset(val_frames, val_labels, num_frames)
        self.test_dataset = CNNDataset(test_frames, test_labels, num_frames) if test_frames is not None else None