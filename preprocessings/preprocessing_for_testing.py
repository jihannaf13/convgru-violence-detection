# Import Library
import cv2
import torch
import torchvision.transforms as transforms
import os   
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from torch.cuda.amp import autocast

# Import Class
from config.base_config import DEFAULT_DATASET_CONFIG, DEFAULT_TRAINING_CONFIG, DEFAULT_PREDICT_CONFIG, DEFAULT_MODEL_CONFIG
# from model_factories.pl_cnn_convgru_3_3 import PlCNNConvGRUClassifier
from model_factories.cnn_convgru import PlCNNConvGRUClassifier
# from model_factories.pl_cnn_convlstm_3_2 import PlCNNConvLSTMClassifier
from model_factories.convgru import ConvGRUGradCAM

# Directory
# batch_size = DEFAULT_DATASET_CONFIG['batch_size']
number_of_frames = DEFAULT_PREDICT_CONFIG['number_of_frames']

def load_state_dict(statedict_path,
                    model_type):
    # Load the saved model
    print(f"Load State Dict from {statedict_path}")
    state_dict = torch.load(statedict_path)

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
    #         learning_rate=DEFAULT_TRAINING_CONFIG['learning_rate'],
    #         weight_decay=DEFAULT_TRAINING_CONFIG['weight_decay'],
    #         use_amp=True,
    #         use_scan=DEFAULT_TRAINING_CONFIG['use_scan'],
    #         )
        
    model.load_state_dict(state_dict)

    # Return the model
    return model

def preprocess_frame(frame, 
                     transform):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = transform(frame)  # Apply transformation
    return frame

def show_video(video_path, 
               predictions, 
               frame_count, 
               model_name, 
               output_dir='video_output/annotated_videos'):
    # Ensure the output directory exists
    os.makedirs(output_dir, 
                exist_ok=True)
    
    threshold = get_threshold_for_model(model_name)
    # threshold = 0.5
    print(f"Threshold for {model_name}: {threshold}")
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Form the output file path
    output_path = os.path.join(output_dir, 
                               f"{model_name}_{video_name}.mp4")

    # Annotate the video and show to the user
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 output
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = 0
    total_predictions = len(predictions)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Draw annotation on frame
        prediction_index = min(current_frame // number_of_frames,
                               total_predictions - 1)
        # prediction_index = (current_frame) // batch_size
        prediction_label = "Violence" if predictions[prediction_index] >= threshold else "Non-Violence"
        color = (0, 0, 255) if prediction_label == 'Violence' else (0, 255, 0)
        cv2.putText(frame, f'Prediction: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f'Total frame: {frame_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f'Current frame: {current_frame + 1}', (10, 110),    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)
        
        # Display the frame
        cv2.imshow('Annotated Video', frame)
        current_frame += 1
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return frame_count

def show_video_CAM(overlaid_frames,
                   video_path, 
                   predictions, 
                   frame_count, 
                   model_name, 
                   output_dir='video_output/annotated_videos/CAM'):
    # Ensure the output directory exists
    os.makedirs(output_dir, 
                exist_ok=True)
    
    threshold = get_threshold_for_model(model_name)
    # threshold = 0.5
    print(f"Threshold for {model_name}: {threshold}")
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Form the output file path
    output_path = os.path.join(output_dir, f"{model_name}_{video_name}.mp4")

    # Annotate the video and show to the user
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 output
    fps = 20  # Set FPS for the output video
    height, width, _ = overlaid_frames[0].shape
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = 0
    total_predictions = len(predictions)
    
    for frame in overlaid_frames:
        # Draw annotation on frame
        prediction_index = min(current_frame // number_of_frames, total_predictions - 1)
        prediction_label = "Violence" if predictions[prediction_index] >= threshold else "Non-Violence"
        color = (0, 0, 255) if prediction_label == 'Violence' else (0, 255, 0)
        
        # Overlay the prediction label and frame counter on the frame
        cv2.putText(frame, f'Prediction: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f'Total frames: {frame_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f'Current frame: {current_frame + 1}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)
        
        # Display the frame
        cv2.imshow('Annotated Video', frame)
        current_frame += 1
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    out.release()
    cv2.destroyAllWindows()
    return frame_count

def get_threshold_for_model(model_name, 
                            filepath=DEFAULT_TRAINING_CONFIG["best_threshold_path"]):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Remove the word "model" from the filename
    modified_filename = model_name.replace('model_', '')

    for line in lines:
        model, threshold = line.strip().split(':')
        if model == modified_filename:
            return float(threshold)
    
    raise ValueError(f"Threshold for model {model_name} not found.")

def predict_video_violence(video_path, 
                           cnn_model, 
                           convgru_model):
    cnn_model.eval()
    convgru_model.eval()

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Path: {video_path}")
    print(f'Total frame: {frame_count}')
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((DEFAULT_DATASET_CONFIG['width'], DEFAULT_DATASET_CONFIG['height'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    frames = []
    with torch.no_grad():
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame = preprocess_frame(frame, transform)
            frames.append(frame)
            
    frames = torch.stack(frames).cuda()
    print(f'Frames Shape: {frames.shape}')

    cnn_features = []
    with torch.no_grad():
        for frame in frames:
            frame = frame.unsqueeze(0)
            features = cnn_model(frame)
            cnn_features.append(features.squeeze(0))

    cnn_features = torch.stack(cnn_features)
    print(f'CNN Features: {cnn_features.shape}')
    print(f'CNN Features len: {len(cnn_features)}')

    batches = torch.split(cnn_features, number_of_frames)

    probabilities = []
    with torch.no_grad():
        for batch in batches:
            # batch.requires_grad_(True)  # Ensure gradients are tracked
            # convgru_model.float()
            with autocast():
                outputs = convgru_model(batch.unsqueeze(0).cuda())
                softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
                probabilities.extend(softmax_outputs[:, 1].cpu().numpy())

    print(f'Probabilities: {probabilities}')
    return probabilities, frame_count, frames, cnn_features
    # return probabilities, frame_count

def get_video_files(folder_path):
    video_files = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp4' or '.avi'):  # Adjust this if your videos have different extensions
                video_files.append(os.path.join(root, file))
                # Assign label based on the folder name
                if 'NonViolence' in root:
                    labels.append(0)
                elif 'Violence' in root:
                    labels.append(1)
    return video_files, labels

def overlay_attention_on_frame(frame, attention_map):
    """
    Overlays the attention map on the video frame.
    
    Args:
    - frame (numpy array): The video frame (H, W, 3).
    - attention_map (numpy array): The attention map (H', W').
    
    Returns:
    - overlay_frame (numpy array): The frame with the attention map overlay.
    """
    # Ensure attention map is 2D
    if attention_map.ndim > 2:
        attention_map = np.mean(attention_map, axis=0)  # Average over any extra dimensions
    
    # Resize attention map to match the frame size
    attention_map_resized = cv2.resize(attention_map, (frame.shape[1], frame.shape[0]))

    # Normalize the attention map to the range [0, 1]
    attention_map_normalized = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
    
    # Convert the attention map to uint8 format (8-bit, single channel)
    attention_map_uint8 = np.uint8(255 * attention_map_normalized)

    # Ensure the attention map is a 2D single-channel image
    if attention_map_uint8.ndim == 2:
        attention_map_uint8 = attention_map_uint8[..., np.newaxis]  # Add a new axis to make it (H, W, 1)

    # Apply the color map (jet)
    heatmap = cv2.applyColorMap(attention_map_uint8, cv2.COLORMAP_JET)

    # Overlay the heatmap onto the original frame
    overlay_frame = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    return overlay_frame

def save_video_with_attention(frames, output_path, fps=30):
    """
    Saves a list of frames as a video file.

    Args:
    - frames (list of numpy arrays): The frames to save as a video.
    - output_path (str): The path to save the output video.
    - fps (int): Frames per second for the output video.
    """
    height, width, _ = frames[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f'Video saved to {output_path}')

def overlay_cam_on_image(img, cam, alpha=0.5):
    """
    Overlays the CAM on the image.

    Args:
    - img (numpy array): The input image.
    - cam (numpy array): The class activation map.
    - alpha (float): Transparency of the overlay.

    Returns:
    - output_image (numpy array): The image with CAM overlay.
    """
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))  # Resize CAM to match image size
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

    return overlay_img

def predict_video_violence_CAM(video_path, 
                               cnn_model, 
                               model,
                               modeltype):
    cnn_model.eval()
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Path: {video_path}")
    print(f'Total frame: {frame_count}')
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((DEFAULT_DATASET_CONFIG['width'], DEFAULT_DATASET_CONFIG['height'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    frames = []
    original_frames = []  # Store original frames for visualization
    with torch.no_grad():
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            original_frames.append(frame)  # Store original frame before preprocessing
            frame = preprocess_frame(frame, transform)
            frames.append(frame)
            
    frames = torch.stack(frames).cuda()
    print(f'Frames Shape: {frames.shape}')

    cnn_features = []
    with torch.no_grad():
        for frame in frames:
            frame = frame.unsqueeze(0)
            features = cnn_model(frame)
            cnn_features.append(features.squeeze(0))

    cnn_features = torch.stack(cnn_features)
    print(f'CNN Features: {cnn_features.shape}')
    print(f'CNN Features len: {len(cnn_features)}')

    batches = torch.split(cnn_features, number_of_frames)

    probabilities = []
    overlaid_frames = []  # Store frames with attention overlays
    # grad_cam = ConvGRUGradCAM(model, target_layer='ConvGRUCell_02')
    with torch.no_grad():
        for i, batch in enumerate(batches):
            # batch.requires_grad_(True)  # Ensure gradients are tracked
            # model.float()
            with autocast():
                outputs = model(batch.unsqueeze(0).cuda())
                softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
                probabilities.extend(softmax_outputs[:, 1].cpu().numpy())
                if (modeltype == "scanconvgru"):
                    for j in range(batch.size(0)):
                        # model.visualize_attention(j)
                        attention_map = model.attention_maps[j].cpu().numpy().mean(axis=0)
                        overlaid_frame = overlay_attention_on_frame(original_frames[i * number_of_frames + j], attention_map)
                        overlaid_frames.append(overlaid_frame)      
                elif (modeltype == "convgru"):
                    for j in range(batch.size(0)):
                        # model.visualize_attention(j)
                        frame_input = batch[j].unsqueeze(0).unsqueeze(0).cuda()
                        frame_input.requires_grad_(True)
                        with torch.set_grad_enabled(True):
                            cam = grad_cam.generate_cam(frame_input, target_class=1)  # Assume class 1 for violence

                        # Convert original frame to RGB for overlay
                        original_frame = cv2.cvtColor(original_frames[i * number_of_frames + j], cv2.COLOR_BGR2RGB)
                        overlaid_frame = overlay_cam_on_image(original_frame, cam)
                        overlaid_frames.append(overlaid_frame)  

    print(f'Overlaid frames len: {len(overlaid_frames)}')
    # Save or display the video with attention overlays
    output_path = 'outputs/output_with_attention.avi'
    save_video_with_attention(overlaid_frames, output_path)
    
    print(f'Probabilities: {probabilities}')
    return probabilities, frame_count, frames, cnn_features, overlaid_frames
    # return probabilities, frame_count