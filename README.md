# YOLOv8 Object Detection

Welcome to my Computer Vision project repository, focusing on object detection using the cutting-edge YOLOv8 architecture. This project encompasses the development, training, validation, and deployment of a YOLOv8 model tailored for efficient and accurate object detection tasks. My goal through this initiative is not only to demonstrate my technical proficiency in computer vision but also to contribute a ready-to-deploy, high-performance solution to the community.

![YOLOv8 image](/image.png)


## Project Overview

Object detection is a critical task in computer vision, finding applications in various domains such as autonomous driving, security surveillance, and augmented reality. Leveraging the YOLOv8 architecture, this project aims to address the need for real-time, accurate object detection, providing a foundation for further research and development in the field.

### Key Features

- **Model Training**: Utilizes `train_model.py` for training the YOLOv8 model on custom datasets, enabling the model to recognize and locate objects with high precision.
- **Model Exporting**: The `export.py` script converts the trained model into the ONNX format, ensuring compatibility across different platforms and facilitating easy integration into production environments.
- **Object Detection**: `predict.py` demonstrates the model's capability to detect objects in new images, showcasing the practical application of the trained model.
- **Object Detection in ONNX and C++**:   `main.cpp` and utilises the ONNX model for object detection on video feed with bounding boxes and label.
- **Data Anotation**: we utilised CVAT.ai to annotate over 900 basketball images for our dataset to track baskeballs and basketball rims.

## Technologies Used

- **YOLOv8**: For state-of-the-art object detection, offering an optimal balance between speed and accuracy.
- **Python**: The primary programming language used for model development and scripting.
- **C++**: Programming language used to implement object detection with the ONNX model.
- **OpenCV (cv2)**: For image processing tasks, including reading and writing.
- **ONNX**: Provides a platform-neutral format for model sharing, ensuring broad compatibility.
- **CVAT.ai**: For annotating the image dataset.

## Working Improvements
- Detect trajectory using Kalman filter

## Getting Started

### Prerequisites

- Python 3.8 or newer
- pip and virtualenv for managing dependencies

### Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/yourprojectname.git
   ```
2. Navigate to the project directory and create a virtual environment:
   ```sh
   python -m venv venv
   ```
3. Activate the virtual environment:
   - **Windows**: `.\venv\Scripts\activate`
   - **Linux/macOS**: `source venv/bin/activate`
4. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

1. **Train the Model**: Customize and run `train_model.py` to train the YOLOv8 model on your dataset.
2. **Export the Model**: Use `export.py` to convert your trained model into the ONNX format.
3. **Detect Objects**: Apply the trained model to new images using `predict.py` to perform object detection.

## Contribution

Contributions, issues, and feature requests are welcome! Feel free to check the issues page for any open issues or to submit new ones.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Alvin Yap Abidin - alvin.yapabidin@gmail.com - https://www.linkedin.com/in/alvin-yap-abidin/


---

This README provides a concise overview of a professional-grade computer vision project designed to showcase expertise in the field, suitable for a graduate looking to demonstrate their capabilities to potential employers. It includes sections on the project overview, key features, technologies used, how to get started, contribution guidelines, licensing information, and contact details.
