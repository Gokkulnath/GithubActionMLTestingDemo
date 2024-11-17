# MNIST Classification with CI/CD Pipeline

This project implements a lightweight Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model is designed to achieve >95% accuracy in a single epoch while maintaining less than 25,000 parameters.

## Model Architecture

The model uses a simple but effective architecture:
- 2 Convolutional layers with batch normalization
- Max pooling after each convolution
- 2 Fully connected layers
- Dropout for regularization

Key features:
- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Parameters: <25,000
- Training accuracy: >95% in one epoch

## Project Structure 

.
├── .github
│ └── workflows
│ └── ml-pipeline.yml
├── model
│ ├── init.py
│ └── network.py
├── tests
│ ├── init.py
│ └── test_model.py
├── .gitignore
├── README.md
├── requirements.txt
└── train.py


## Setup and Installation

1. Clone the repository:

```
bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies:

bash
pip install -r requirements.txt

## Usage

### Training

To train the model:

bash
python train.py


The trained model will be saved in the `models/` directory with a timestamp suffix.

### Testing

To run the tests:

bash
pytest tests/test_model.py -v

The tests verify:
- Model architecture (input/output dimensions)
- Parameter count (<25,000)
- Training accuracy (>95%)

## CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically:
1. Sets up a Python environment
2. Installs dependencies
3. Runs all tests
4. Uploads the trained model as an artifact

The pipeline is triggered on every push to the repository.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest

## Model Artifacts

Trained models are saved with timestamps in the format: mnist_model_YYYYMMDD_HHMMSS.pth


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST Dataset: [LeCun et al.](http://yann.lecun.com/exdb/mnist/)
- PyTorch Documentation and Tutorials