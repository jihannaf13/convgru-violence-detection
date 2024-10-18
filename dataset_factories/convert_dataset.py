# Import Library
from torch.utils.data import ConcatDataset
from torchvision import transforms
import numpy as np
import torch
from rich.progress import Progress

# Import Class
from dataset_factories.dataset_maker import TrainDataset, ValidationDataset
from utils.utils import save_concat_dataset_to_npz

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 
              0.456, 
              0.406], 
        std=[0.229, 
             0.224, 
             0.225]
        ),
])

def convert_to_npz_train(train_v_dir,
                         train_nv_dir,
                         output_dir):
    print("Start converting training data")

    # Create datasets for both classes
    dataset_violence = TrainDataset(train_v_dir, 
                                    label=1,
                                    transform=transform)
    dataset_non_violence = TrainDataset(train_nv_dir, 
                                        label=0,
                                        transform=transform)    
    
    # Combine the datasets into one DataLoader
    combined_dataset = ConcatDataset([
        dataset_violence + dataset_non_violence
    ])
    
    # Save all frames and labels to a single .npz file
    save_concat_dataset_to_npz(
        combined_dataset, 
        output_dir
    )

    print(f"Dataset saved to {output_dir}")

def convert_to_npz_val(validation_dir,
                       output_dir):
    print("Start converting validation data")

    # Create datasets for both classes
    dataset_val = ValidationDataset(validation_dir, 
                                    transform=transform)
    
    # Save all frames and labels to a single .npz file
    save_concat_dataset_to_npz(
        dataset_val, 
        output_dir
    )

    print(f"Dataset saved to {output_dir}")

def load_dataset(path):
    # Initialize the progress bar using rich
    with Progress() as progress:
        print(f"Start loading dataset from {path}")
        # Create a task for loading data
        load_task = progress.add_task("[green]Loading dataset...", total=None)
        
        # Load the dataset from the .npz file
        dataset = np.load(path)
        progress.advance(load_task)  # Advance once the loading is complete
        
        # Extract frames and labels (simulating progress for extraction)
        frames = dataset['frames']
        labels = dataset['labels']

        print("Dataset loaded")

    return frames, labels

def convert_to_tensor(frames,
                      labels,
                      total_length,
                      chunk_size):
    # Initialize the progress bar using rich
    with Progress() as progress:
        frames_tensors = []
        labels_tensors = []
        print("Start converting data to tensors")
        # Create a task for loading data
        load_task = progress.add_task("[green]Loading dataset...",
                                      total=total_length)
        
        for start_idx in range(0, total_length, chunk_size):
            end_idx = min(start_idx + chunk_size, total_length)
            chunk_frames = frames[start_idx:end_idx]
            chunk_labels = labels[start_idx:end_idx]

            chunk_frame_tensors = [torch.tensor(frame, 
                                                dtype=torch.float16) for frame in chunk_frames]
            chunk_label_tensors = [torch.tensor(label, 
                                                dtype=torch.long) for label in chunk_labels]

            frames_tensors.extend(chunk_frame_tensors)
            labels_tensors.extend(chunk_label_tensors)

            progress.advance(load_task, end_idx - start_idx)

        # Convert the list of tensors to a single tensor if needed
        final_frames = torch.stack(frames_tensors, 
                                   dim=0)      
        final_labels = torch.stack(labels_tensors)

        print("Frame[0] Shape: ", final_frames[0].shape)
        print("Labels[0] Shape: ", final_labels[0])
        print("Data converted to tensors")

    return final_frames, final_labels

def load_dataset_train(path):
    print(f"Start loading dataset from {path}")
    frames_np, labels_np = load_dataset(path)
    # Convert NumPy arrays to PyTorch tensors
    frames = [torch.tensor(frame, 
                           dtype=torch.float16) for frame in frames_np]
    frames = torch.stack(frames, 
                         dim=0)
    labels = torch.tensor(labels_np, 
                          dtype=torch.long)
    print("After Converting")
    print(f"Videos[0]: {frames[0].shape}")
    print(f"Labels[0]: {labels[0]}")

    return frames, labels
