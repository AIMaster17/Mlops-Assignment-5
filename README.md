# Iris Classification Project

## 🌸 Overview

This is an end-to-end Machine Learning Operations (MLOps) project that implements an Iris flower species classification system. The project demonstrates best practices in ML pipeline orchestration using DVC, containerization with Docker, and deployment to AWS EC2.

## 🎯 Features

- **Machine Learning Pipeline**: Automated data preparation, model training, and evaluation using DVC
- **Web Application**: Flask-based web interface for real-time predictions
- **Containerization**: Docker containerization for consistent deployment
- **CI/CD**: GitHub Actions workflow for automated testing and validation
- **Cloud Deployment**: Ready for deployment on AWS EC2

## 📁 Project Structure

```
iris-classification-project/
│
├── .dvc/                    # DVC metadata directory
│   └── .gitignore
│
├── .github/
│   └── workflows/
│       └── ci-cd.yaml       # GitHub Actions CI/CD workflow
│
├── data/                    # Data managed by DVC
│   ├── test.csv
│   └── train.csv
│
├── models/                  # Trained model managed by DVC
│   ├── model.joblib
│   └── scaler.joblib
│
├── src/                     # Source code for the ML pipeline
│   ├── prepare.py          # Data preparation script
│   ├── train.py            # Model training script
│   └── evaluate.py         # Model evaluation script
│
├── templates/               # HTML templates for the Flask app
│   └── index.html
│
├── Dataset/                 # Original dataset
│   └── Iris.csv
│
├── .dvcignore              # Files for DVC to ignore
├── .gitignore              # Files for Git to ignore
├── app.py                  # The main Flask application file
├── dvc.yaml                # DVC pipeline definition
├── dvc.lock                # DVC lock file for reproducible runs
├── Dockerfile              # Instructions to build the Docker container
├── README.md               # High-level project summary
├── REPORT.md               # Detailed report for your assignment
└── requirements.txt        # Python project dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (for containerization)
- Git
- DVC

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd iris-classification-project
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the DVC pipeline**

   ```bash
   dvc repro
   ```

4. **Start the Flask application**

   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 🐳 Docker Deployment

### Build the Docker image

```bash
docker build -t iris-classification .
```

### Run the container

```bash
docker run -p 5000:5000 iris-classification
```

## 📊 Model Performance

The Logistic Regression model achieves high accuracy on the Iris dataset:

- **Training Accuracy**: ~97-98%
- **Test Accuracy**: ~95-100%

## 🔧 Technologies Used

- **Machine Learning**: scikit-learn
- **Web Framework**: Flask
- **Pipeline Orchestration**: DVC (Data Version Control)
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Cloud Platform**: AWS EC2

## 📝 API Endpoints

### Home Page

- **URL**: `/`
- **Method**: GET
- **Description**: Renders the prediction form

### Predict

- **URL**: `/predict`
- **Method**: POST
- **Parameters**:
  - `sepal_length`: Float (cm)
  - `sepal_width`: Float (cm)
  - `petal_length`: Float (cm)
  - `petal_width`: Float (cm)
- **Returns**: Predicted species and confidence

### Health Check

- **URL**: `/health`
- **Method**: GET
- **Returns**: Application health status

## 🌟 Key Features

### DVC Pipeline

The project uses DVC to manage the ML pipeline with three stages:

1. **Prepare**: Load and split the data
2. **Train**: Train the model
3. **Evaluate**: Evaluate model performance

### Web Interface

- Clean, modern UI with gradient backgrounds
- Real-time species prediction
- Input validation
- Confidence score display

### Docker Container

- Slim Python base image
- Multi-stage build for optimization
- Health checks included
- Production-ready configuration

## 📖 Documentation

For detailed information about the project, including step-by-step setup instructions and AWS deployment guide, please refer to [REPORT.md](REPORT.md).

## 👥 Contributing

This is an academic project for MLOps assignment. Feel free to fork and modify for learning purposes.

## 📄 License

This project is created for educational purposes.

## 🙏 Acknowledgments

- Iris dataset from UCI Machine Learning Repository
- scikit-learn for machine learning algorithms
- Flask for web framework
- DVC for pipeline orchestration

---

**Note**: Make sure to train the model using `dvc repro` before running the Flask application.
