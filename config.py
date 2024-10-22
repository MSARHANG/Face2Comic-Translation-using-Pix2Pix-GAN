import torch
from torchvision import transforms


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT_DIR = "dataset"
IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 200
NUM_WORKERS = 5

LEARNING_RATE = 2e-4
BETA_1 = 0.5 
BETA_2 = 0.99
L1_LAMBDA = 50
L1_GP = 10

LOAD_MODEL = True
SAVE_MODEL = True
DISCRIMINATOR_CHECKPOINTS = 'disc1.pth.tar'
GENERATOR_CHECKPOINTS = 'gen1.pth.tar'


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
