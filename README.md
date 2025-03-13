forked from https://github.com/mberkay0/image-colorization
Thanks a lot for your paper on GANs and the code you're sharing with us, that helped a lot in making this App.

Tested with Python 3.12, Py version is limited to fastAI compatibility.
I am using https://github.com/Eugeny/tabby and https://git-scm.com/downloads/win to run the Windows App.

# Supercolor

Supercolor is a PyQt6-based GUI application that uses deep learning models to colorize grayscale images. It supports multiple architectures such as ResNet, EfficientNet, and ShuffleNet.

## Features
- **Deep Learning-based Image Colorization**: Uses pre-trained GANs and CNNs for accurate image colorization.
- **GUI with PyQt6**: User-friendly interface with buttons for training, loading models, and colorizing images.
- **Model Selection**: Choose from multiple architectures like ResNet, EfficientNet, and ShuffleNet.
- **GPU Acceleration**: Utilizes CUDA if available for faster processing.
- **Sequential Image Processing**: Processes images one by one from a selected folder.
- **Custom Training**: Train and fine-tune a model using a dataset of RGB images, where the model learns to predict color information from grayscale input in the L*a*b* color space.
- **Auto-Save Checkpoints**: Automatically saves the model at regular intervals.
- **Adjustable Warm-Up for Generator Training**: Customize warm-up epochs before full training.
- **GAN Training with Pretrained Generator**: Start GAN training with a pretrained generator.
- **Loss Functions**: Uses L1 Loss for pixel-wise accuracy, GAN Loss for realism, and L2 Regularization to prevent overfitting.
- **Advanced Data Augmentation**: Random transformations to improve generalization, including brightness, contrast, perspective, blur, and rotation.
- **Gradient Accumulation**: Reduces memory usage by accumulating gradients over multiple batches.
- **Cosine Annealing Learning Rate**: Adjusts the learning rate dynamically for stable training.
- **PatchGAN Discriminator**: Classifies image patches for improved adversarial learning.
- **Model Loading & Versioning**: Supports loading and continuing training from saved models.
- **Early Stopping**: Stops training if no improvement is detected over multiple epochs.
- **GPU Memory Optimization**: Frees unused GPU memory after each processing step.

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
Supercolor operates in the **L*a*b*** color space, where grayscale input represents the L-channel (lightness), and the model predicts the a and b color channels.
Supercolor follows a structured training process involving a **Generator Pretraining Phase** followed by **GAN Training**.

#### 1. Generator Pretraining
- The generator is trained **without a discriminator** using L1 loss.
- L1 loss minimizes pixel-wise differences between grayscale and ground truth color images.
- A warm-up phase can be configured to slowly increase the learning rate.
- Training runs for a specified number of epochs, and models are auto-saved periodically.
- Uses **gradient accumulation** to manage memory for larger batch sizes.
- **Early stopping** prevents overfitting by stopping training if no improvement is observed.
- **Models are versioned** based on training progress, allowing easy rollback.

#### 2. GAN Training
- After the generator is pretrained, the GAN training phase begins.
- A **PatchGAN discriminator** is introduced to classify real vs. fake images.
- **GAN Loss (BCE Loss or MSE Loss)** is used to optimize both networks.
- The generator learns to generate more realistic images through adversarial training.
- The learning rate schedule follows a **Cosine Annealing** strategy.
- **GPU memory is freed after each step** to optimize training stability.

### Colorize Images
1. Load a trained model or train a new one.
2. Click "Select Folder" and choose a folder with grayscale images.
3. Click "Colorize Images in Folder" to start processing.
4. The colorized images will be saved in a subfolder named `colorized`.
5. **Processed images are converted back from L*a*b* to RGB** for correct color representation.

## Data Augmentation
To improve model robustness, the dataset undergoes augmentation, including:
- **Brightness & Contrast Adjustments**: Random brightness (±2%) and contrast (±3%).
- **Saturation Adjustments**: Randomly enhances saturation (1.00 to 1.05 scale).
- **Perspective Transformations**: Random distortions applied to simulate depth.
- **Gaussian Blur**: Random blur effects with varying kernel sizes.
- **Random Rotations**: Rotates images up to 45 degrees.
- **Horizontal Flipping**: Randomly flips images left-right.
- **Random Resize Scaling**: Applies random resizing to prevent overfitting.
- **Random Noise Injection**: Adds slight noise to images to improve robustness.

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

