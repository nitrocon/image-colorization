forked from https://github.com/mberkay0/image-colorization
Thanks a lot for youre paper in GANs and the code youre sharing with us, that helped a lot making this App.

Tested with python 3.12, py version is limited to fastAI compatibility
I am using https://github.com/Eugeny/tabby and https://git-scm.com/downloads/win to run the windows App.

# Supercolor

Supercolor is a PyQt6-based GUI application that uses deep learning models to colorize grayscale images. It supports multiple architectures such as ResNet, EfficientNet, and ShuffleNet.

## Features
- **Deep Learning-based Image Colorization**: Uses pre-trained GANs and CNNs for accurate image colorization.
- **GUI with PyQt6**: User-friendly interface for easy interaction.
- **Model Selection**: Choose from multiple architectures like ResNet, EfficientNet, and ShuffleNet.
- **GPU Acceleration**: Utilizes CUDA if available for faster processing.
- **Batch Processing**: Supports colorizing multiple images at once.
- **Custom Training**: Train your own model with a dataset of grayscale images.
- **Auto-Save Checkpoints**: Automatically saves the model at regular intervals.
- **Adjustable Warm-Up for Generator Training**: Customize warm-up epochs before full training.
- **GAN Training with Pretrained Generator**: Start GAN training with a pretrained generator.
- **Loss Functions**: Uses L1 Loss for pixel-wise accuracy, GAN Loss for realism, and L2 Regularization to prevent overfitting.
- **Advanced Data Augmentation**: Random transformations to improve generalization, including brightness, contrast, perspective, blur, and rotation.

## Installation

### Prerequisites
- Python 3.12
- NVIDIA CUDA-compatible GPU (optional, but recommended for performance)

### Setup
```bash
git clone https://github.com/nitrocon/supercolor.git
cd supercolor
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python maingui.py
```

### Training Process
Supercolor follows a structured training process involving a **Generator Pretraining Phase** followed by **GAN Training**.

#### 1. Generator Pretraining
- The generator is trained **without a discriminator** using L1 loss.
- L1 loss minimizes pixel-wise differences between grayscale and ground truth color images.
- A warm-up phase can be configured to slowly increase the learning rate.
- Training runs for a specified number of epochs, and models are auto-saved periodically.

#### 2. GAN Training
- After the generator is pretrained, the GAN training phase begins.
- A **PatchGAN discriminator** is introduced to classify real vs. fake images.
- **GAN Loss (BCE Loss or MSE Loss)** is used to optimize both networks.
- The generator learns to generate more realistic images through adversarial training.
- The learning rate schedule follows a **Cosine Annealing** strategy.

### Colorize Images
1. Load a trained model or train a new one.
2. Click "Select Folder" and choose a folder with grayscale images.
3. Click "Colorize Images in Folder" to start processing.
4. The colorized images will be saved in a subfolder named `colorized`.

## Data Augmentation
To improve model robustness, the dataset undergoes augmentation, including:
- **Brightness & Contrast Adjustments**: Random brightness (±2%) and contrast (±3%).
- **Perspective Transformations**: Random distortions applied to simulate depth.
- **Gaussian Blur**: Random blur effects with varying kernel sizes.
- **Random Rotations**: Rotates images up to 45 degrees.
- **Horizontal Flipping**: Randomly flips images left-right.

## Model Architectures
The application supports the following architectures:
- **ShuffleNet v2 (0.5, 1.0, 1.5, 2.0)**
- **EfficientNet B0**
- **ResNet (18, 34, 50)**

## Parameter List (512x512 Resolution)
| Model | Parameters |
|--------|------------|
| ShuffleNet v2 x0.5 | 5,714,129 |
| ShuffleNet v2 x1.0 | 19,428,737 |
| ShuffleNet v2 x1.5 | 40,862,961 |
| ShuffleNet v2 x2.0 | 75,720,113 |
| EfficientNet B0 | 101,862,549 |
| ResNet 18 | 33,865,305 |
| ResNet 34 | 43,973,465 |
| ResNet 50 | 341,808,473 |

## License
This project is licensed under the MIT License.
