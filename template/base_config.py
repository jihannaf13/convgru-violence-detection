# Dataset directories
DEFAULT_DATASET_CONFIG = {
    # Video Path
    # Example: "E:/dataset/Video/Training/NonViolence"
    'train_non_violence_dir': "Path to training non-violence videos",
    'train_violence_dir': "Path to training violence videos",
    'validation_root_dir': "Path to validation videos",
    
    # Npz File Path
    # Example: "E:/dataset/Models/npz_dataset/train.npz"
    'train_npz_dir': "NPZ File Path for training",
    'validation_npz_dir': "NPZ File Path for validation",

    # Npz Temp File Path
    # Example: "E:/dataset/Models/temp/train"
    'npz_train_temp_dir' : "NPZ Temp Path for training",
    'npz_val_temp_dir' : "NPZ Temp Path for validation",

    # CNN Npz File Path
    # Example: "E:/dataset/Models/npz_dataset/train_cnn_resnet50_8_updated_1.npz"
    'train_cnn_npz_dir': "NPZ Train CNN Path for training",
    'validation_cnn_npz_dir': "NPZ Train CNN Path for validation",

    'batch_size': 8,
    'num_workers': 1,
    'height': 240,
    'width': 240,
    'num_frames': 30
}

# Training parameters
DEFAULT_TRAINING_CONFIG = {
    'learning_rate': 1e-7,
    'weight_decay': 1e-5,
    'mode': 'max',
    'factor': 0.1,

    'patience': 2,
    'num_epochs': 2,
    # Saved model path
    'saved_model_path': "models/entire_model/",
    'dropout' : 0.2,
    # Attention Mechanism
    'use_scan': True,

    'best_threshold_path':"misc/best_threshold.txt"
}

# Model parameters
DEFAULT_MODEL_CONFIG = {
    'num_classes': 2,
    # 'input_size': 576,  # Adjust according to the output size of the CNN 
    'input_size': 2048,  # Adjust according to the output size of the CNN 
    'hidden_sizes': [32, 64, 16],
    # 'hidden_sizes': [64, 128, 32],
    'kernel_sizes': [3, 5, 3],
    'n_layers': 3,
    # Scan parameters
    'ratio_scan': 2,
}

# Neptune logger config
DEFAULT_NEPTUNE_CONFIG = {
    'api_key': "",
    'project': "",
}

# Default params to be logged
DEFAULT_HYPERPARAMS = {
    'batch_size': DEFAULT_DATASET_CONFIG['batch_size'],
    'num_workers': DEFAULT_DATASET_CONFIG['num_workers'],
    'input_size': DEFAULT_MODEL_CONFIG['input_size'],
    'learning_rate': DEFAULT_TRAINING_CONFIG['learning_rate'],
    'weight_decay': DEFAULT_TRAINING_CONFIG['weight_decay'],
    'mode': DEFAULT_TRAINING_CONFIG['mode'],
    'factor': DEFAULT_TRAINING_CONFIG['factor'],
    'patience': DEFAULT_TRAINING_CONFIG['patience'],
    'num_epochs': DEFAULT_TRAINING_CONFIG['num_epochs'],
    'dropout': DEFAULT_TRAINING_CONFIG['dropout'],
    'hidden_sizes': DEFAULT_MODEL_CONFIG['hidden_sizes'],
    'kernel_sizes': DEFAULT_MODEL_CONFIG['kernel_sizes'],
    'n_layers': DEFAULT_MODEL_CONFIG['n_layers'],
    'use_cbam': DEFAULT_TRAINING_CONFIG['use_cbam'],
    'ratio_cbam': DEFAULT_MODEL_CONFIG['ratio_cbam'],
    'kernel_size_cbam': DEFAULT_MODEL_CONFIG['kernel_size_cbam'],
    'use_scan': DEFAULT_TRAINING_CONFIG['use_scan'],
    'ratio_scan': DEFAULT_MODEL_CONFIG['ratio_scan'],
}

DEFAULT_CONVERT_CONFIG = {
    'checkpoint_path': ".neptune/VD-30/VD-30/checkpoints/epoch=11-step=1680.ckpt",
    'statedict_path': "state_dict",
    'model_path': "models"
}

# Testing
DEFAULT_PREDICT_CONFIG = {
    # ConvGRU
    # Example: "E:/dataset/Models/entire_model/model_VD-332_state_dict.pth"
    'statedict_path': "Path to State Dict",
    
    # Example: "E:/dataset/Models/model_VD-361.pth"
    'pth_path':"Path to PTH (Conditional)",
    
    # Example: "E:/dataset/Video/Testing"
    'testing_folder_path': "Path to Folder Path",

    # Example: "E:/dataset/Video/Testing/Violence/V_304.mp4"
    'video_path':"Path to Video",

    'number_of_frames': 30
}