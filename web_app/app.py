import streamlit as st
import torch
import os
import cv2
import imageio
import numpy as np
from model_factories.base_model import ResNet50Extractor
from config.base_config import DEFAULT_PREDICT_CONFIG, DEFAULT_DATASET_CONFIG, DEFAULT_TRAINING_CONFIG
from preprocess_predict.preprocess_predict import predict_video_violence
from pl_heatmap import extract_frames, calculate_motion_heatmap, apply_heatmap_to_frames, resize_frames

# Directory
pth_path = DEFAULT_PREDICT_CONFIG['pth_path']
base_model = ResNet50Extractor().cuda()
batch_size = DEFAULT_DATASET_CONFIG['batch_size']

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

def show_video(video_path, 
               predictions, 
               frame_count, 
               model_name, 
               output_dir='video_output/annotated_videos'):
    # Ensure the output directory exists
    os.makedirs(output_dir, 
                exist_ok=True)
    
    threshold = get_threshold_for_model(model_name)
    print(f"Threshold: {threshold}")
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Form the output file path
    output_path = os.path.join(output_dir, 
                               f"{model_name}_{video_name}.mp4")

    # Annotate the video and show to the user
    cap = imageio.get_reader(video_path, 'ffmpeg')
    fps = cap.get_meta_data()['fps']
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264')

    current_frame = 0
    for frame in cap:
        # Draw annotation on frame
        prediction_index = current_frame // DEFAULT_DATASET_CONFIG['batch_size']
        prediction_label = "Violence" if predictions[prediction_index] >= threshold else "Non-Violence"
        color = (255, 0, 0) if prediction_label == 'Violence' else (0, 255, 0)


        # Add text annotations
        cv2.putText(frame, f'Prediction: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f'Total frame: {frame_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f'Current frame: {current_frame + 1}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        writer.append_data(frame)
        
        current_frame += 1
    
    cap.close()
    writer.close()
    return frame_count, output_path

def process_video(video_path):
    convgru_model = torch.load(pth_path)

    # Load the model and evaluation
    cnn_model = ResNet50Extractor().cuda().eval()
    convgru_model.cuda().eval()

    # Extract model name without extension
    model_name = os.path.splitext(os.path.basename(pth_path))[0]
    
    # Predict violence in the video and annotate it
    predictions, frame_count = predict_video_violence(video_path, cnn_model, convgru_model)
    
    # Show Video
    frame_count, output_video_path = show_video(video_path, predictions, frame_count, model_name)
    
    # Example usage
    grid_size = (4, 4)  # Grid size (rows, columns)
    frame_size = (240, 240)  # Size of each frame in the montage
    num_frames = frame_count
    num_frames_fix = num_frames / batch_size 
    # st.write(f"Number of Frames: {num_frames_fix}")

    # Extract frames
    frames = extract_frames(video_path, int(num_frames_fix))

    # Check extracted frames
    st.write(f"Extracted {len(frames)} frames")

    # Calculate motion heatmap for each frame
    heatmaps = calculate_motion_heatmap(frames)

    # Apply heatmap to frames
    heatmap_frames = apply_heatmap_to_frames(frames, heatmaps)

    # Resize frames
    resized_frames = resize_frames(heatmap_frames, frame_size)

    # Check resized frames
    # st.write(f"Resized frames shape: {[frame.shape for frame in resized_frames]}")
    
    return output_video_path

st.title("Violence Detection in Video")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    video_path = os.path.join("temp", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(video_path)
    print(video_path)

    if st.button("Process Video", type="primary"):
        st.write("Processing video...")
        output_video_path = process_video(video_path)
        print(output_video_path)

        # convertedVideo = "./testh264.mp4"
        # subprocess.call(args=f"ffmpeg -y -i {output_video_path} -c:v libx264 {convertedVideo}".split(" "))

        if output_video_path and os.path.exists(output_video_path):
            st.video(output_video_path)
        else:
            st.error("Processed video file could not be found or is not accessible.")

    # Clean up the temporary file
    os.remove(video_path)