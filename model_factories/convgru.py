import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import cv2


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, 
                                    hidden_size, 
                                    kernel_size, 
                                    padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, 
                                     hidden_size, 
                                     kernel_size, 
                                     padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, 
                                  hidden_size, 
                                  kernel_size, 
                                  padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size).cuda()
            else:
                prev_state = torch.zeros(state_size)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state

class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, 
                               self.hidden_sizes[i], 
                               self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden

class ConvGRUGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        # Placeholder for the gradients and activations
        self.gradients = None
        self.activations = None

        # Register hook to the target layer to save gradients and activations
        self._register_hooks()

    def _register_hooks(self):
        # Access the target ConvGRU cell through the convgru attribute
        target_layer = getattr(self.model.convgru, self.target_layer)
        
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Zero the gradients
        self.model.zero_grad()

        # Create a one-hot tensor for the target class
        if len(output[-1].shape) == 1:
            # If the output is 1D, we only need to index directly
            one_hot_output = torch.zeros_like(output[-1])
            one_hot_output[target_class] = 1
        else:
            # If the output has more dimensions, we assume it's (batch_size, num_classes)
            one_hot_output = torch.zeros_like(output[-1])
            one_hot_output[:, target_class] = 1

        # Backward pass to get gradients for the target class
        output[-1].backward(gradient=one_hot_output)

        # Compute the weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Average gradients over width and height

        # Compute the weighted sum of the activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)  # Weighted sum across channels

        # Apply ReLU to keep only positive influences
        cam = F.relu(cam)

        # Normalize the CAM
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.cpu().numpy()
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