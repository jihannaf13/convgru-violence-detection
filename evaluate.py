#Import Library
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import os

#Import Class
from model_factories.base_model import MobileNetV3Extractor, ResNet50Extractor
from config.base_config import DEFAULT_PREDICT_CONFIG
from preprocessings.preprocessing_for_testing import predict_video_violence, get_video_files, get_threshold_for_model, load_state_dict

# Directory
pth_path = DEFAULT_PREDICT_CONFIG['pth_path']
testing_folder_path = DEFAULT_PREDICT_CONFIG['testing_folder_path']
statedict_path = DEFAULT_PREDICT_CONFIG['statedict_path']

def plot_roc_curve(predictions, 
                   true_labels,
                   path):
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, 
                            predictions)
    # Compute Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'evaluation_data/ROC/roc_curve_{path}.png')
    plt.show()

def plot_confusion_matrix(predictions,
                          true_labels, 
                          threshold,
                          path):
    # Convert probabilities to binary predictions using the threshold
    binary_predictions = (predictions >= threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, binary_predictions)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions)
    recall = recall_score(true_labels, binary_predictions)
    f1 = f1_score(true_labels, binary_predictions)

    print(f'Accuracy: {accuracy:.4f}\n')
    print(f'Precision: {precision:.4f}\n')
    print(f'Recall: {recall:.4f}\n')
    print(f'F1 Score: {f1:.4f}\n')
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Non-Violent', 'Violent'])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')

    # Customize the plot
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Adjust layout to prevent cropping
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f'evaluation_data/CM/confusion_matrix_{path}.png')
    plt.show()

    # Save metrics to a text file
    with open(f'evaluation_data/EM/evaluation_metrics_{path}.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write('\nConfusion Matrix:\n')
        f.write(np.array2string(cm, separator=', '))

def evaluate_model(cnn_model, 
                   model,
                   path): 
    video_files, true_labels = get_video_files(testing_folder_path)
    print(f"True Labels: {true_labels}")
    print(f"Video Files: {video_files}")
    all_probabilities = []

    for video_file in video_files:
        # probabilities, _ = predict_video_violence(video_file, 
        #                                           cnn_model, 
        #                                           model)
        probabilities, frame_count, frames, cnn_features = predict_video_violence(video_file, cnn_model, model)
        video_probability = np.mean(probabilities)
        all_probabilities.append(video_probability)

    all_probabilities = np.array(all_probabilities)
    true_labels = np.array(true_labels)

    if len(true_labels) == 0 or len(all_probabilities) == 0:
        print("Error: True labels or probabilities are empty.")
    else:
        plot_roc_curve(all_probabilities, 
                       true_labels,
                       path)
        threshold = get_threshold_for_model(path)
        plot_confusion_matrix(all_probabilities, 
                              true_labels,
                              threshold,
                              path)

if __name__ == '__main__':
    #state_dict for state_dict and whole_model for whole model
    status = "state_dict"

    if status == "state_dict":
        model = load_state_dict(statedict_path,
                                model_type="convgru")
        path = statedict_path.split('_state_dict')[0]
    else:
        model = torch.load(pth_path)
        path = pth_path

    # Load model and evaluate
    cnn_model = ResNet50Extractor().cuda().eval()
    # print(batch.device)
    # model = torch.load(pth_path)
    model.cuda().eval()

    final_path = os.path.splitext(os.path.basename(path))[0]
    
    # Evaluate Model
    evaluate_model(cnn_model,
                   model,
                   final_path)