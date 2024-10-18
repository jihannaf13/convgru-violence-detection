# Import Library
import torch
import os
import matplotlib.pyplot as plt
import time  # Add this import

# Import Class
# from model_factories.base_model import MobileNetV3Extractor, ResNet50Extractor
from model_factories.base_model import MobileNetV3Extractor, ResNet50Extractor
from model_factories.cnn_convgru import PlCNNConvGRUClassifier
from config.base_config import DEFAULT_PREDICT_CONFIG, DEFAULT_DATASET_CONFIG
from preprocessings.preprocessing_for_testing import predict_video_violence, show_video, load_state_dict, show_video_CAM, predict_video_violence_CAM

# Directory
statedict_path = DEFAULT_PREDICT_CONFIG['statedict_path']
pth_path = DEFAULT_PREDICT_CONFIG['pth_path']
video_path = DEFAULT_PREDICT_CONFIG['video_path']
# base_model = MobileNetV3Extractor().cuda()
base_model = ResNet50Extractor().cuda()
batch_size = DEFAULT_DATASET_CONFIG['batch_size']

if __name__ == '__main__':
    #state_dict for state_dict and whole_model for whole model
    status = "state_dict"
    # model_type = "convlstm"
    model_type = "convgru"

    if status == "state_dict":
        model = load_state_dict(statedict_path, model_type)
        path = statedict_path.split('_state_dict')[0]
    else:
        model = torch.load(pth_path)
        path = pth_path

    # Load the model and evaluation
    cnn_model = ResNet50Extractor().cuda().eval()
    model.cuda().eval()

    # Extract model name without extension
    model_name = os.path.splitext(os.path.basename(path))[0]

    start_time = time.time()  # Start timing
    # Predict violence in the video and annotate it
    predictions, frame_count, frames, cnn_features, overlaid_frames = predict_video_violence_CAM(video_path, 
                                cnn_model, 
                                model, 
                                modeltype="scanconvgru")
                                # modeltype="convgru")

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    # Show Video
    frame_count = show_video(video_path, 
                       predictions, 
                       frame_count,
                       model_name)

    frame_count = show_video_CAM(overlaid_frames,
                             video_path, 
                       predictions, 
                       frame_count,
                       model_name)
    
    # apply_gradcam_to_video(model_name, predictions, cnn_features, model, frames)
    
