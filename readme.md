1. Create Dataset
To minimize the cost of device computations, I decide to divide the process into some steps
    a. create a base_config.py from template (template/base_config.py) and move to config folder 
    b. Change the dir in DEFAULT_DATASET_CONFIG
    c. Run the program in make_dataset.ipynb (Recommend to test it with one video for the first time)
    d. The NPZ file is saved :)

2. Training the model
    a. Make a neptune (https://neptune.ai/) account for saving the graph and insert your api key and project name in DEFAULT_NEPTUNE_CONFIG from config -> base_config.py
    b. Open your terminal and type python .\pl_main_3_2.py to run the main program and it will give you these informations
    
    P.S if your NPZ file is big, it will take a while for loading the dataset

3. Testing
    a. Change all the path in DEFAULT_PREDICT_CONFIG from config -> base_config.py (use the state_dict for the model)
    b. Run python .\pl_predict_2.py in terminal
    c. The video will be popped out and saved in video_output folder
