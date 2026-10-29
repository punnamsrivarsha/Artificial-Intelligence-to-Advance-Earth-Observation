# 🛰️ Artificial Intelligence to Advance Earth Observation

## 📌 Project Overview

The **Earth Observation using AI** project demonstrates a web-based simulator for satellite image segmentation using deep learning concepts. It focuses on the Sentinel-2 Earth observation domain and provides a dashboard to visualize simulated model training progress, loss values, mean Intersection over Union (mIoU), and engine logs.

The application is built using Python and Flask, with deep learning libraries such as PyTorch and Segmentation Models PyTorch included in the project dependencies.

## 🎯 Objectives

- Explore the application of Artificial Intelligence in Earth Observation.
- Demonstrate a web interface for configuring segmentation model architectures.
- Display simulated training progress and evaluation metrics.
- Provide a foundation for developing satellite image segmentation workflows.

## 🛠️ Technologies Used

- Python
- Flask
- PyTorch
- Torchvision
- Segmentation Models PyTorch
- Transformers
- NumPy
- Scikit-learn
- HTML and CSS

## 🖥️ Project Features

- Web-based dashboard
- Model architecture selection
- Simulated training progress
- Loss and mIoU metric display
- Real-time-style engine logs

## 📂 Project Structure

```text
Earth_Observation/
├── app.py
├── data_utils.py
├── engine.py
├── evaluation.py
├── metrics.py
├── models.py
├── trainer.py
├── requirements.txt
└── templates/
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-PUNNAMSRIVARSHA/Artificial-Intelligence-to-Advance-Earth-Observation.git
```

### 2. Navigate to the project directory

```bash
cd Artificial-Intelligence-to-Advance-Earth-Observation
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open in your browser

Visit:

```text
http://127.0.0.1:5000
```

## 📊 Important Note

The current dashboard demonstrates simulated training metrics and progress. The displayed loss and mIoU values should not be interpreted as verified results from training and evaluating a deep learning model on real Sentinel-2 imagery.

## 🚀 Future Enhancements

- Integrate real Sentinel-2 satellite imagery.
- Implement complete segmentation model architectures.
- Add dataset loading and preprocessing pipelines.
- Train and evaluate models on labeled satellite images.
- Visualize predicted segmentation masks.
- Report actual model evaluation metrics.

## 👩‍💻 Author

**Punnam Srivarsha**

## 📄 License

A license can be added when the project licensing terms have been decided.
