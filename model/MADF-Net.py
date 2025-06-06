from collections import OrderedDict
import torch
from torch import nn
import numpy as np

# Define the autoencoder model
class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, use_cls: bool=True):
        super(AutoEncoder, self).__init__()
        self.use_cls = use_cls  # Whether to use a classifier
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),  # Reduce dimensionality
            nn.ReLU(),  # Activation function
            nn.Linear(int(input_dim/2), hidden_dim),  # Further reduce to hidden dimension
            nn.ReLU()
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, int(input_dim/2)),  # Expand dimensionality
            nn.ReLU(),
            nn.Linear(int(input_dim/2), input_dim),  # Reconstruct input dimension
            nn.Sigmoid()  # Output in the range [0, 1]
        )
        # Classifier branch (optional)
        if self.use_cls:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),  # Normalize the hidden features
                nn.Linear(hidden_dim, int(hidden_dim / 2)),  # Reduce dimensionality
                nn.ReLU(),
                nn.Linear(int(hidden_dim / 2), 1),  # Output a single value (e.g., for classification)
            )

    def forward(self, x):
        encoded = self.encoder(x)  # Encode the input
        decoded = self.decoder(encoded)  # Decode the encoded features
        if self.use_cls:
            class_out = self.classifier(encoded)  # Get classifier output
            return encoded, decoded, class_out  # Return encoder, decoder, and classifier outputs
        else:
            return encoded, decoded, None  # No classifier output

# Gating Fusion Module
class GateFusionModule(nn.Module):
    def __init__(self, dim: int):
        super(GateFusionModule, self).__init__()
        # Linear layers for feature transformation
        self.hidden1 = nn.Linear(dim, dim)
        self.hidden2 = nn.Linear(dim, dim)
        self.hidden_sigmoid = nn.Linear(dim * 2, 1)  # For gating mechanism
        self.fc = nn.Linear(dim, dim)  # Final fusion layer
        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, main_feature, aux_feature_1, aux_feature_2):
        # Gate mechanism -- Fuse two auxiliary features
        aux1 = self.hidden1(aux_feature_1)
        aux2 = self.hidden2(aux_feature_2)

        aux = torch.cat((aux1, aux2), dim=1)  # Concatenate auxiliary features
        z = self.sigmoid_f(self.hidden_sigmoid(aux))  # Compute gating weights
        aux_fusion_feature = z.view(z.size()[0], 1) * aux1 + (1 - z).view(z.size()[0], 1) * aux2  # Weighted fusion

        # Fuse with the main feature
        main_feature = self.fc(main_feature) + aux_fusion_feature
        return main_feature

# Depth Fusion Module
class DepthFusionModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DepthFusionModule, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.att = nn.Linear(hidden_size,hidden_size)
        self.conv1d1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)  # 1D convolution
        self.conv1d2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.linear2 = nn.Linear(input_size, hidden_size)
        self.sigmoid_f = nn.Sigmoid()
        self.linear3 = nn.Linear(hidden_size, output_size)
        # Generate weights for three branches
        self.hidden_sigmoid = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 3),
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x1 = self.linear1(x)
        x1 = self.sigmoid_f(x1)
        x1 = x1.unsqueeze(2)  # Add a dimension
        x1 = self.conv1d1(x1)  # Apply 1D convolution

        x2 = self.linear2(x)
        x2 = self.sigmoid_f(x2)
        x2 = x2.unsqueeze(2)

        # Fuse the outputs of the two paths
        x = x1 + x2

        x = x.squeeze(-1)
        x = self.linear3(x)

        x = nn.Softmax(dim=1)(self.sigmoid_f(self.hidden_sigmoid(x)))
        return x

# Gating Decision Fusion Module
class GateDecisionFusionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()
        # Linear layers for feature transformation
        self.hidden1 = nn.Linear(dim, dim)
        self.hidden2 = nn.Linear(dim, dim)
        self.hidden3 = nn.Linear(dim, dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim/2)),
            nn.ReLU(),
            nn.Linear(int(dim/2), 1),
        )
        self.EMA = DepthFusionModule(dim*3,dim*6,dim*3)  # Depth fusion module

    def forward(self, branch_1, branch_2, branch_3):
        # Gate mechanism
        branch1 = self.hidden1(branch_1)
        branch2 = self.hidden2(branch_2)
        branch3 = self.hidden3(branch_3)
        branch = torch.cat((branch1, branch2, branch3), dim=1)  # Concatenate branches
        z = self.EMA(branch)  # Compute gating weights

        branch_fusion_feature = z[:, 0].view(z.size()[0], 1) * branch1 + z[:, 1].view(z.size()[0], 1) * branch2 + z[:, 2].view(z.size()[0], 1) * branch3  # Weighted fusion

        out = self.classifier(branch_fusion_feature)
        return out

# Define the overall model
class AdaptiveAutoEncoderFusion(nn.Module):
    def __init__(self, input_dims: dict, hidden_dim: int,
                 use_loss: bool=True, use_multi_branch: bool=True, use_gate_decision: bool=True):
        super(AdaptiveAutoEncoderFusion, self).__init__()
        self.use_cls = use_loss  # Whether to use single-modality classifier
        self.use_multi_branch = use_multi_branch
        self.use_gate_decision = use_gate_decision

        # Create multiple autoencoders
        self.encoders = nn.ModuleDict()
        for k, v in input_dims.items():
            self.encoders[k] = AutoEncoder(v, hidden_dim, self.use_cls)  # Create an autoencoder for each modality

        # Create gating fusion modules for each branch
        self.branchGFM = nn.ModuleDict()
        if self.use_multi_branch:
            for k, v in input_dims.items():
                self.branchGFM[k] = GateFusionModule(hidden_dim)
        else:
            self.branchGFM = GateFusionModule(hidden_dim)  # Single branch

        # Create gating decision fusion module
        if self.use_multi_branch and self.use_gate_decision:
            self.gdfm = GateDecisionFusionModule(hidden_dim)
        elif self.use_multi_branch and not self.use_gate_decision:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim*3),
                nn.Linear(hidden_dim*3, int(hidden_dim/2)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim/2), 1),
            )
        elif not self.use_multi_branch and not self.use_gate_decision:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, int(hidden_dim/2)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim/2), 1),
            )

    def forward(self, **x):
        # Prepare the input -- numpy --> tensor
        for name, feature in x.items():
            if isinstance(feature, np.ndarray):
                x[name] = torch.from_numpy(feature).type(torch.float32).to('cuda')
        # Forward pass through each autoencoder
        latent_repr = OrderedDict()
        autoencoder_outputs = OrderedDict()
        ae_classifier_outputs = OrderedDict()
        for name, feature in x.items():
            latent, autoencoder_output, classifier_output = self.encoders[name](feature)
            latent_repr[name] = latent  # Latent representation for fusion
            autoencoder_outputs[name] = autoencoder_output  # Reconstructed features
            ae_classifier_outputs[name] = classifier_output.squeeze()  # Classifier output

        # Three branches: Audio, Text, Image
        if self.use_multi_branch:
            AudioBranch = self.branchGFM['Audio'](latent_repr['Audio'], latent_repr['Text'], latent_repr['Image'])
            TextBranch = self.branchGFM['Text'](latent_repr['Text'], latent_repr['Audio'], latent_repr['Image'])
            ImageBranch = self.branchGFM['Image'](latent_repr['Image'], latent_repr['Audio'], latent_repr['Text'])
        else:
            repr = self.branchGFM(latent_repr['Text'], latent_repr['Audio'], latent_repr['Image'])

        # Branch fusion
        if self.use_gate_decision and self.use_multi_branch:  # Multi-branch + gating decision fusion
            out = self.gdfm(AudioBranch, TextBranch, ImageBranch)
        elif self.use_multi_branch and not self.use_gate_decision:  # Multi-branch + no gating decision fusion
            out = self.classifier(torch.cat((AudioBranch, TextBranch, ImageBranch), dim=1))
        else:  # Single branch
            out = self.classifier(repr)
        return out.squeeze(), ae_classifier_outputs, x, autoencoder_outputs