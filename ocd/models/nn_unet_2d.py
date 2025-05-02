import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
  def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    use_residual=True,
  ):
    super().__init__()
    self.use_residual = use_residual
    self.conv1 = nn.Conv2d(
      in_channels, out_channels, kernel_size, stride, padding, bias=False
    )
    self.instnorm1 = nn.InstanceNorm2d(out_channels)
    self.lrelu1 = nn.LeakyReLU(negative_slope=0.01)

    self.conv2 = nn.Conv2d(
      out_channels, out_channels, kernel_size, stride, padding, bias=False
    )
    self.instnorm2 = nn.InstanceNorm2d(out_channels)
    self.lrelu2 = nn.LeakyReLU(negative_slope=0.01)

    # Residual connection: 1x1 convolution to match dimensions if needed
    if use_residual and in_channels != out_channels:
      self.residual_conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
      )
    else:
      self.residual_conv = None

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.instnorm1(out)
    out = self.lrelu1(out)

    out = self.conv2(out)
    out = self.instnorm2(out)
    out = self.lrelu2(out)

    if self.use_residual:
      if self.residual_conv is not None:
        identity = self.residual_conv(identity)
      out += identity  # Add residual connection
    return out


class Encoder(nn.Module):
  def __init__(self, in_channels, base_num_features, num_pool):
    super().__init__()
    self.layers = nn.ModuleList()
    current_channels = in_channels

    for i in range(num_pool):
      out_channels = min(base_num_features * (2**i), 512)
      self.layers.append(ConvBlock(current_channels, out_channels))
      self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
      current_channels = out_channels

    # Bottleneck layer
    self.bottleneck = ConvBlock(current_channels, base_num_features * (2**num_pool))
    self.current_channels = base_num_features * (2**num_pool)

  def forward(self, x):
    skip_connections = []
    for _, layer in enumerate(self.layers):
      if isinstance(layer, ConvBlock):
        x = layer(x)
        skip_connections.append(x)  # Store output before pooling as skip connection
      elif isinstance(layer, nn.MaxPool2d):
        x = layer(x)

    x = self.bottleneck(x)
    return x, skip_connections


class Decoder(nn.Module):
  def __init__(self, base_num_features, num_pool, num_classes):
    super().__init__()
    self.layers = nn.ModuleList()
    current_channels = base_num_features * (2**num_pool)

    for i in range(num_pool - 1, -1, -1):
      out_channels = min(base_num_features * (2**i), 512)
      # Upsampling layer
      self.layers.append(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
      )
      # Convolutional block after concatenating skip connection
      # The input channels will be current_channels (from upsampling) + skip_connection_channels
      self.layers.append(ConvBlock(current_channels + out_channels, out_channels))
      current_channels = out_channels

    # Final output layer
    self.output_layer = nn.Conv2d(current_channels, num_classes, kernel_size=1)

  def forward(self, x, skip_connections):
    for _, layer in enumerate(self.layers):
      if isinstance(layer, nn.Upsample):
        x = layer(x)
        # Concatenate skip connection
        skip = skip_connections.pop(-1)
        # Pad upsampled tensor if dimensions don't match skip connection (due to pooling/conv)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
      elif isinstance(layer, ConvBlock):
        x = layer(x)

    x = self.output_layer(x)
    return x


class NNUnet2D(nn.Module):
  def __init__(self, in_channels, num_classes=3, base_num_features=32, num_pool=6):
    super().__init__()
    self.encoder = Encoder(in_channels, base_num_features, num_pool)
    self.decoder = Decoder(base_num_features, num_pool, num_classes)

  def forward(self, x):
    x, skip_connections = self.encoder(x)
    x = self.decoder(x, skip_connections)
    return x
