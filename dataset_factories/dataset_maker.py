#Import Library
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import seed_everything
import os

# Import Class
from config.base_config import DEFAULT_DATASET_CONFIG
from dataset_factories.convert_dataset import convert_to_npz_train, convert_to_npz_val, load_dataset, convert_to_tensor
from model_factories.base_model import ResNet50Extractor, MobileNetV3Extractor
from utils.utils import save_concat_temp_npz

# Input Directories
train_v_dir = DEFAULT_DATASET_CONFIG['train_violence_dir']
train_nv_dir = DEFAULT_DATASET_CONFIG['train_non_violence_dir']
validation_dir = DEFAULT_DATASET_CONFIG['validation_root_dir']

# Temporary File Directories
npz_train_temp_dir = DEFAULT_DATASET_CONFIG['npz_train_temp_dir']
npz_val_temp_dir = DEFAULT_DATASET_CONFIG['npz_val_temp_dir']

# Output Directories
npz_train_dir = DEFAULT_DATASET_CONFIG['train_npz_dir']
npz_val_dir = DEFAULT_DATASET_CONFIG['validation_npz_dir']
npz_train_cnn_dir = DEFAULT_DATASET_CONFIG['train_cnn_npz_dir']
npz_val_cnn_dir = DEFAULT_DATASET_CONFIG['validation_cnn_npz_dir']

seed_everything(42, workers=True)

batchsize = DEFAULT_DATASET_CONFIG['batch_size']
numworkers = DEFAULT_DATASET_CONFIG['num_workers']

# Base model
# base_model = MobileNetV3Extractor().half().cuda()
base_model = ResNet50Extractor().half().cuda()

def cnn_extract(dataset,
                cnn_model):
    print("Start extracting features")
    
    cnn_model.eval()

    data_loader = DataLoader(
        dataset, 
        batch_size=batchsize, 
        shuffle=True, 
        num_workers=numworkers, 
        persistent_workers=True, 
        pin_memory=True
    )

    cnn_features = []   
    all_labels = []
    with torch.no_grad():
        for batch_frames, batch_labels in data_loader:
            batch_size, num_frames, channels, height, width = batch_frames.size()
            # Initialize a list to hold features for the current batch
            batch_features = []
            for i in range(num_frames):
                frame = batch_frames[:, i, :, :, :].half().cuda()
                features = cnn_model(frame)
                batch_features.append(features.cpu())

            # Stack features for the current batch along the frame dimension
            # Shape: (batch_size, num_frames, features, 1, 1)
            batch_features = torch.stack(batch_features, 
                                         dim=1)  
            
            # Append to the main cnn_features list
            cnn_features.append(batch_features)
            
            # Repeat labels for each frame in the video
            all_labels.extend(batch_labels.cpu().numpy())
            
    # Shape: (total_videos, num_frames, features, 1, 1)
    cnn_features = torch.cat(cnn_features, 
                             dim=0)
      
    all_labels = torch.tensor(all_labels)
    
    print("Finish extracting features")

    return cnn_features, all_labels

def apply_cnn(dataset_dir,
              temp_dir,
              cnn_model):
    
    print("Apply CNN and Transfer Learning")

    # Apply Transfer Learning
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Load the .npz file
    frames, labels = load_dataset(dataset_dir)
    total_length = len(frames)
    chunk_size = 100

    temp_feature_files = []
    temp_label_files = []
    
    for start_idx in range(0, total_length, chunk_size):
        end_idx = min(start_idx + chunk_size, total_length)
        frames_chunk = frames[start_idx:end_idx]
        labels_chunk = labels[start_idx:end_idx]
        
        frames_tens, labels_tens = convert_to_tensor(frames_chunk,
                                                     labels_chunk, 
                                                     end_idx - start_idx,
                                                     chunk_size)
        
        dataset = TensorDataset(frames_tens, labels_tens)
        
        chunk_features, chunk_labels = cnn_extract(dataset, 
                                                   cnn_model)
        
        temp_feature_file = f"{temp_dir}/temp_features_{start_idx}.npz"
        temp_label_file = f"{temp_dir}/temp_labels_{start_idx}.npz"
        
        np.savez(temp_feature_file, 
                 frames=chunk_features.numpy())
        np.savez(temp_label_file, 
                 labels=chunk_labels.numpy())
        
        temp_feature_files.append(temp_feature_file)
        temp_label_files.append(temp_label_file)

    # save_concat_temp_npz(temp_dir, cnn_model)

def create_dataset(status):
    if (status == 'make_dataset'):
        print("Creating Dataset to .npz")
        # Training Data
        convert_to_npz_train(train_v_dir,
                             train_nv_dir,
                             npz_train_dir)
        # Validation Data
        convert_to_npz_val(validation_dir,
                           npz_val_dir)
    else:
        pass
    
    # Apply CNN
    apply_cnn(dataset_dir=npz_train_dir,
              temp_dir=npz_train_temp_dir,
              cnn_model=base_model)
    
    apply_cnn(dataset_dir=npz_val_dir,
              temp_dir=npz_val_temp_dir,
              cnn_model=base_model)
    

    
    
