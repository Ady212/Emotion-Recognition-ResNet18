# Facial Emotion Recognition with ResNet-18

CS6482 Deep Learning coursework (Assignment 1, CNNs): a facial emotion classifier for three classes (**angry**, **happy**, **sad**) built with transfer learning on a pre-trained **ResNet-18** in PyTorch.

## Approach

- **Data:** images loaded with `ImageFolder` from `train/` and `test/` folders. Training uses random resized crops and horizontal flips; all images are resized to 224×224 and normalised with ImageNet statistics.
- **Model:** ImageNet-pretrained ResNet-18 with a new 3-class output layer. Early layers are frozen; `layer3`, `layer4` and the classifier are fine-tuned.
- **Training:** cross-entropy loss, Adam (lr 1e-5) and a step learning-rate schedule over 12 epochs. The best model on the test split is saved.
- **Inference:** loads the best checkpoint, predicts the emotion of a new image and prints class probabilities.

## Results

The best model reached **80.4% test accuracy** (test loss 0.504) at epoch 10. Loss and accuracy curves, example predictions and discussion are in [`report.pdf`](report.pdf).

## Running it

```bash
pip install torch torchvision torchinfo matplotlib pillow
python emotion_recognition_resnet18.py
```

Before running, set `data_dir` (a folder containing `train/` and `test/`, each with one sub-folder per class) and `image_path` (an image to classify) at the top of the script. The dataset is not included in this repository.
