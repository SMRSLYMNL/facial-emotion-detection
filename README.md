# A Facial Emotion Detection App Using CNN and FER-2013 Dataset

Facial emotion detection is an innovative technology that enables machines to interpret human emotions from facial expressions. This capability is essential in mental health, user experience, and security. Developed a facial emotion detection app using Convolutional Neural Networks (CNNs) and the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), chosen over CK+ and MMA datasets, and a Python application for real-time detection.

## CNN Architecture
* **Input Layer**
  - Input Shape: 48x48x1 (grayscale image)
  
* **Convolutional Layers**
  - Conv Layer 1: 32 filters, 3x3 kernel, ReLU activation, Batch Normalization
  - Conv Layer 2: 64 filters, 3x3 kernel, ReLU activation, Batch Normalization

* **Pooling Layers**
  - Pooling Layer 1: Max pooling, 2x2 window
  - Pooling Layer 2: Max pooling, 2x2 window
  
* **Dropout Layers**
  - Dropout Rate: 0.25 (to prevent overfitting)

* **Fully Connected Layers**
  - Dense Layer 1: 1024 neurons, ReLU activation, Dropout Rate: 0.5
  - Output Layer: 7 neurons (softmax activation for emotion classification)
 
## Training the Model
### Hyperparameters
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Epochs**: 30

## Evaluation
The model's performance was evaluated using accuracy, precision, recall, and F1-score, achieving around 75% accuracy on the test set.

![Plot](https://github.com/SMRSLYMNL/facial-emotion-detection/assets/10634784/4ee65677-2749-433c-bacf-121f45775fdd)


## Real-Time Detection App

### Python Application
Alongside training the CNN model, a Python app was implemented to detect faces in real-time using a webcam and classify the detected faces with the trained model. This app uses OpenCV for video capture and face detection and developed CNN for emotion prediction.

### Key Features
- **Real-Time Detection**: Processes live video frames and predict emotions instantly.
- **User Interface**: Simple UI showing the camera feed and detecting emotions.
- **Cross-Platform**: Compatible with both Windows and macOS.
