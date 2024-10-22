## Face2Comic: Translating Faces into Comic Art using Pix2Pix GAN

This project explores the use of a Pix2Pix Generative Adversarial Network (GAN) to translate real-life images of faces into stylized comic book art. 

### Project Overview

The goal of this project is to develop Pix2Pix model from ground up that can automatically transform realistic photographs of human faces into visually appealing comic book style representations.

### Task
The task at hand is to train a Pix2Pix GAN that can generate a comic book style image of a face given a real-life photograph as input. The model should learn to identify and extract the key features of the face, like eyes, nose, and mouth, and transform them into the characteristic comic book style.

### Dataset

The [dataset](https://www.kaggle.com/code/nathannguyendev/face2comic) used for training consists of pairs of images: real-life photographs of faces and their corresponding hand-drawn comic book style counterparts. The dataset is carefully curated to ensure a diverse range of faces, expressions, and angles, allowing the model to learn a robust and generalizable representation of comic art style.

### Model Architecture

The model employed is a [Pix2Pix GAN](https://arxiv.org/abs/1611.07004), which consists of two main components:

• Generator: This U-Net based network takes a real-life face image as input and generates a comic book style image of the same face.
• Discriminator: This PatchGAN network tries to differentiate between real comic book images and the generated ones. 

During training, the generator aims to fool the discriminator by generating realistic comic book images, while the discriminator learns to distinguish real from fake. This adversarial process allows the generator to progressively improve its ability to create convincing comic book representations. 

### Results

Input(Real Photographies)                     |  GroundTruth Label            |   Generated Comic Faces
:----------------------------------:|:-------------------------:|:----------------------------
![](https://github.com/user-attachments/assets/37f68c30-2a6c-4d48-b784-cb4f87893e99)
|![](https://github.com/user-attachments/assets/5f409860-e9fd-411e-a5bd-ef9983e8c3f9)
|![](https://github.com/user-attachments/assets/81b0287d-846a-4cc4-bc93-5bfae186308a)

