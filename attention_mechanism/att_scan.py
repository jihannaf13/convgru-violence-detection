import torch
import torch.nn as nn
import torch.nn.functional as F

class ScanAttentionUpdated(nn.Module):
    def __init__(self, input_channels, 
                 reduction_ratio):
        super(ScanAttentionUpdated, self).__init__()
        reduced_channels = input_channels // reduction_ratio

        # Convolutional layers for input and hidden state
        self.conv_xa = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=True)
        self.conv_ha = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=True)
        
        # Convolutional layer for output weights
        self.conv_y = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=True)

        # Bias term for the attention mechanism
        self.b_A = nn.Parameter(torch.zeros(input_channels, 1, 1))

        # Fully connected layers for channel attention
        self.fc1 = nn.Conv2d(2 * input_channels, reduced_channels, kernel_size=1, bias=True)
        # self.fc1 = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(reduced_channels, input_channels, kernel_size=1, bias=True)

    def forward(self, X_t, H_t_minus_1, return_attention=False):
        if H_t_minus_1 is None:
            H_t_minus_1 = torch.zeros_like(X_t)

        # Check if H_t_minus_1 is a list and concatenate along the channel dimension
        if isinstance(H_t_minus_1, list):
            # Concatenate along channel dimension (dim=1)
            H_t_minus_1 = torch.cat(H_t_minus_1, dim=1)
            # conv1x1 = nn.Conv2d(in_channels=224, out_channels=2048, 
            conv1x1 = nn.Conv2d(in_channels=112, out_channels=2048, 
                                kernel_size=1).cuda().half()
            H_t_minus_1 = conv1x1(H_t_minus_1)  
        
        if isinstance(H_t_minus_1, tuple):
            # Concatenate along channel dimension (dim=1)
            H_t_minus_1 = torch.cat(H_t_minus_1, dim=1)
            # After concatenation, determine the correct number of input channels
            in_channels = H_t_minus_1.size(1)  # Get the number of channels after concatenation
            # conv1x1 = nn.Conv2d(in_channels=224, out_channels=2048, 
            conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=2048, 
                                kernel_size=1).cuda().half()
            H_t_minus_1 = conv1x1(H_t_minus_1)  

        # Assuming X_t has shape (batch_size, channels, height, width)
        batch_size, C, H, W = X_t.size()

        # Apply convolution to input and hidden state
        X_t_proj = self.conv_xa(X_t)
        H_t_minus_1_proj = self.conv_ha(H_t_minus_1)

        # Combine input, hidden state with bias and apply tanh activation
        combined = torch.tanh(X_t_proj + H_t_minus_1_proj + self.b_A)
        
        # Apply convolution after tanh activation to get Y_t
        Y_t = self.conv_y(combined)
        
        # Compute spatial attention weights using Equation (8)
        Y_t_flat = Y_t.view(batch_size, C, -1)
        alpha_t_flat = F.softmax(Y_t_flat, dim=-1)
        
        # Reshape alpha_t back to original spatial dimensions
        alpha_t = alpha_t_flat.view(batch_size, C, H, W)
        
        # Compute attended features
        X_t_hat = alpha_t * X_t
        
        # Apply mean pooling to get channel features
        X_t_hat_pooled = F.adaptive_avg_pool2d(X_t_hat, 
                                               (1, 1)).view(batch_size, C)
        H_t_minus_1_pooled = F.adaptive_avg_pool2d(H_t_minus_1, 
                                                   (1, 1)).view(batch_size, C)
        
        # Compute channel attention weights
        # X_t_hat_pooled_flat = X_t_hat_pooled.view(batch_size, C, 1, 1)
        # H_t_minus_1_pooled_flat = H_t_minus_1_pooled.view(batch_size, C, 1, 1)
        concatenated = torch.cat((X_t_hat_pooled,
                                  H_t_minus_1_pooled), dim=1)
        concatenated = concatenated.view(batch_size, 2 * C, 1, 1)  # Reshape to [batch_size, channels, 1, 1]

        
        # Fully connected layers for channel attention
        intermediate = self.fc1(concatenated)  # W_{ca1} * concatenated + b_{ca1}
        intermediate = F.relu(intermediate)  # \delta

        channel_attention = self.fc2(intermediate) # W_{ca2} * (output of ReLU) + b_{ca2}
        channel_attention = torch.sigmoid(channel_attention).view(batch_size,
                                                                  C, 1, 1)
        
        # Compute final modified input feature map
        Z_t = X_t_hat * channel_attention
        
        if return_attention:
            return Z_t, alpha_t  # Return both the output and the attention map
        else:
            return Z_t
 
class ScanAttentionUpdatedBackup(nn.Module):
    def __init__(self, input_channels, reduction_ratio=8):
        super(ScanAttentionUpdatedBackup, self).__init__()
        reduced_channels = input_channels // reduction_ratio

        # Convolutional layers for input and hidden state
        self.conv_xa = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=True)
        self.conv_ha = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=True)
        
        # Convolutional layer for output weights
        self.conv_y = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=True)

        # Bias term for the attention mechanism
        self.b_A = nn.Parameter(torch.zeros(input_channels, 1, 1))

        # Fully connected layers for channel attention
        self.fc1 = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(reduced_channels, input_channels, kernel_size=1, bias=True)

    def forward(self, X_t, H_t_minus_1):
        # Assuming X_t has shape (batch_size, channels, height, width)
        batch_size, C, H, W = X_t.size()

        # Apply convolution to input and hidden state
        X_t_proj = self.conv_xa(X_t)
        H_t_minus_1_proj = self.conv_ha(H_t_minus_1)

        # Combine input, hidden state with bias and apply tanh activation
        combined = torch.tanh(X_t_proj + H_t_minus_1_proj + self.b_A)
        
        # Apply convolution after tanh activation to get Y_t
        Y_t = self.conv_y(combined)
        
        # Compute spatial attention weights using Equation (8)
        Y_t_flat = Y_t.view(batch_size, C, -1)
        alpha_t_flat = F.softmax(Y_t_flat, dim=-1)
        
        # Reshape alpha_t back to original spatial dimensions
        alpha_t = alpha_t_flat.view(batch_size, C, H, W)
        
        # Compute attended features
        X_t_hat = alpha_t * X_t
        
        # Apply mean pooling to get channel features
        X_t_hat_pooled = F.adaptive_avg_pool2d(X_t_hat, 
                                               (1, 1)).view(batch_size, C)
        H_t_minus_1_pooled = F.adaptive_avg_pool2d(H_t_minus_1, 
                                                   (1, 1)).view(batch_size, C)
        
        # Compute channel attention weights
        X_t_hat_pooled_flat = X_t_hat_pooled.view(batch_size, C, 1, 1)
        H_t_minus_1_pooled_flat = H_t_minus_1_pooled.view(batch_size, C, 1, 1)
        
        # Fully connected layers for channel attention
        intermediate = self.fc1(X_t_hat_pooled_flat + H_t_minus_1_pooled_flat)  # W_{ca1} * (X_{t,i}' + H_{t-1,j}') + b_{ca1}
        intermediate = F.relu(intermediate)  # \delta

        channel_attention = self.fc2(intermediate) # W_{ca2} * (output of ReLU) + b_{ca2}
        channel_attention = torch.sigmoid(channel_attention).view(batch_size, C, 1, 1)
        
        # Compute final modified input feature map
        Z_t = X_t_hat * channel_attention
        
        # Update hidden state
        H_t = Z_t
        
        return Z_t