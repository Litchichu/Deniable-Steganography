import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """

    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks - 1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length * 2,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message1, message2):
        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message1 = message1.unsqueeze(-1)
        expanded_message1.unsqueeze_(-1)
        expanded_message1 = expanded_message1.expand(-1, -1, self.H, self.W)

        # Another fake message for deniable steganography.
        expanded_message2 = message2.unsqueeze(-1)
        expanded_message2.unsqueeze_(-1)
        expanded_message2 = expanded_message2.expand(-1, -1, self.H, self.W)

        encoded_image = self.conv_layers(image)
        # concatenate expanded message and image
        concat = torch.cat([expanded_message1, expanded_message2, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w
