import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from rich.progress import Progress

from config.base_config import DEFAULT_DATASET_CONFIG
num_frames = DEFAULT_DATASET_CONFIG["num_frames"]

class TrainDataset(Dataset):
    """
    Custom Dataset class for loading video data and their corresponding labels for training.

    Args:
        video_dir (str): Directory containing the videos for either violence or non-violence.
        label (int): Label for the videos. Typically, 0 for non-violence and 1 for violence.
        transform (callable, optional): A function/transform to apply to each frame of the video.
        max_frames (int, optional): Maximum number of frames to consider from each video. Default is 16.

    Attributes:
        video_dir (str): Directory containing the videos.
        label (int): Label for the videos.
        transform (callable): Function/transform to apply to each frame.
        max_frames (int): Maximum number of frames to consider from each video.
        videos (list): List of video file names in the video directory.
    """
    def __init__(self, video_dir, label, transform=None, max_frames=num_frames):
        self.video_dir = video_dir
        self.label = label
        self.transform = transform
        self.max_frames = max_frames
        self.videos = os.listdir(video_dir)

    def __len__(self):
        """
        Returns the total number of videos in the dataset.

        Returns:
            int: Total number of videos.
        """
        return len(self.videos)

    def __getitem__(self, idx):
        """
        Retrieves the frames and label for a video at a specified index.

        Args:
            idx (int): Index of the video to retrieve.

        Returns:
            tuple: A tuple containing:
                - frames (torch.Tensor): A tensor of shape (max_frames, 3, width, height) representing the frames of the video.
                - label (torch.Tensor): A tensor containing the label for the video.
        """
        video_path = os.path.join(self.video_dir, self.videos[idx])
        frames = self.extract_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = self.pad_or_truncate(frames)

        frames = [torch.from_numpy(frame).to(torch.float16) if isinstance(frame, np.ndarray) else frame.clone().detach().to(torch.float16) for frame in frames]
        
        frames = torch.stack(frames, dim=0)
        label = torch.tensor(self.label, dtype=torch.long)
        return frames, label

    def extract_frames(self, video_path):
        """
        Extracts frames from a given video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            list: A list of frames extracted from the video. Each frame is a NumPy array.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (DEFAULT_DATASET_CONFIG['width'], 
                                       DEFAULT_DATASET_CONFIG['height']))
            frames.append(frame)
        cap.release()
        return frames

    def pad_or_truncate(self, frames):
        """
        Pads or truncates the list of frames to ensure it has exactly max_frames elements.

        Args:
            frames (list): List of frames to pad or truncate.

        Returns:
            list: A list of frames with exactly max_frames elements.
        """
        if len(frames) > self.max_frames:
            frames = frames[:self.max_frames]
        else:
            pad_size = self.max_frames - len(frames)
            padding = [torch.zeros_like(frames[0]) for _ in range(pad_size)]
            frames.extend(padding)
        return frames

class ValidationDataset(Dataset):
    """
    Custom Dataset class for loading video data and their corresponding labels for validation.

    Args:
        root_dir (str): Root directory containing the 'Violence' and 'NonViolence' subdirectories.
        transform (callable, optional): A function/transform to apply to each frame of the video.
        max_frames (int, optional): Maximum number of frames to consider from each video. Default is 16.

    Attributes:
        root_dir (str): Root directory containing the videos.
        transform (callable): Function/transform to apply to each frame.
        max_frames (int): Maximum number of frames to consider from each video.
        violence_dir (str): Directory containing the violence videos.
        non_violence_dir (str): Directory containing the non-violence videos.
        violence_videos (list): List of paths to violence video files.
        non_violence_videos (list): List of paths to non-violence video files.
        labels (list): List of labels corresponding to each video file (1 for violence, 0 for non-violence).
        video_paths (list): List of paths to all video files.
    """
    def __init__(self, root_dir, transform=None, max_frames=num_frames):
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames

        self.violence_dir = os.path.join(root_dir, 'Violence')
        self.non_violence_dir = os.path.join(root_dir, 'NonViolence')

        self.violence_videos = [os.path.join(self.violence_dir, f) for f in os.listdir(self.violence_dir) if f.endswith('.avi') or f.endswith('.mp4')]
        self.non_violence_videos = [os.path.join(self.non_violence_dir, f) for f in os.listdir(self.non_violence_dir) if f.endswith('.avi') or f.endswith('.mp4')]

        self.labels = [1] * len(self.violence_videos) + [0] * len(self.non_violence_videos)
        self.video_paths = self.violence_videos + self.non_violence_videos

    def __len__(self):
        """
        Returns the total number of videos in the dataset.

        Returns:
            int: Total number of videos.
        """
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        Retrieves the frames and label for a video at a specified index.

        Args:
            idx (int): Index of the video to retrieve.

        Returns:
            tuple: A tuple containing:
                - frames (torch.Tensor): A tensor of shape (max_frames, 3, width, height) representing the frames of the video.
                - label (torch.Tensor): A tensor containing the label for the video.
        """
        video_path = self.video_paths[idx]
        frames = self.extract_frames(video_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = self.pad_or_truncate(frames)
        frames = torch.stack(frames, dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        print(f"Frames Shape Val: {frames.shape}")
        print(f"Labels Shape Val: {label}")
        return frames, label

    def extract_frames(self, video_path):
        """
        Extracts frames from a given video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            list: A list of frames extracted from the video. Each frame is a NumPy array.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (DEFAULT_DATASET_CONFIG['width'], 
                                       DEFAULT_DATASET_CONFIG['height']))
            frames.append(frame)
        cap.release()
        return frames

    def pad_or_truncate(self, frames):
        """
        Pads or truncates the list of frames to ensure it has exactly max_frames elements.

        Args:
            frames (list): List of frames to pad or truncate.

        Returns:
            list: A list of frames with exactly max_frames elements.
        """
        if len(frames) > self.max_frames:
            frames = frames[:self.max_frames]
        else:
            pad_size = self.max_frames - len(frames)
            padding = [torch.zeros_like(frames[0]) for _ in range(pad_size)]
            frames.extend(padding)
        return frames