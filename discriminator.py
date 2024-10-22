import torch 
from torch import nn 


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        
        """
        Initializes a cnn block consisting of a Conv2d layer, InstanceNorm2d, 
        and LeakyReLU activation.

        Parameters:
        - in_channels (int): Number of channels in the input image.
        - out_channels (int): Number of filters in the convolutional layer.
        - stride (int): Stride size for the convolution.
        """
        
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        return x
    


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], stride=2, depth=4):
        
        """
        Initializes the Discriminator network. The discriminator is a PatchGAN (with patch size of 70) 
        that classifies each patch of an image as real or fake

        Parameters:
        - in_channels (int): Number of channels in the input image *2 (due to concatenation of X and Y).
        - features (list of int): List of the number of filters for each ConvBlock layer.
        - stride (int): Stride size for the convolutional layers (except the last one).
        - depth (int): Number of ConvBlock layers to stack.
        
        Shape:
        -using images with size of 256 and depth of 4 will lead to :
        -input tensor [batch_size, in_channels*2, 256, 256] -> output tensor [batch_size, 1, 30, 30]
        """
        
        super().__init__()
        assert depth == len(features), "feature numbers doesn't match depth size"

        self.in_channels = in_channels
        self.features = features
        self.stride = stride
        self.depth = depth
        
        self.initial = nn.Sequential(
            *[ConvBlock(in_channels*2 if i == 0 else features[i-1], features[i], stride=stride if i < depth - 1 else 1) 
              for i in range(depth)])
        self.output = nn.Sequential(
            ConvBlock(features[-1], 1, 1)
        )     
    
    def forward(self, x, y):
        x = torch.concat([x,y], dim=1)
        x = self.initial(x)
        x = self.output(x)
        return x



#for testing
'''
def test(batch_size):
    x = torch.randn(batch_size,3,256,256)
    y = torch.randn(batch_size,3,256,256)
        
    model = Discriminator(3, [64, 128, 256, 512], 2, 4)
    output = model(x,y)
    
    return output.shape


test1 = test(32)
print(test1)
'''