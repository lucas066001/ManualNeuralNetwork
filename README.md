[![Python package](https://github.com/lucas066001/ManualNeuralNetwork/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/lucas066001/ManualNeuralNetwork/actions/workflows/python-package.yml)

# LMNN: Learning Manual Neural Networks

*Originally, the 'L' in LMNN stood for the first letter of my name, but after some thought, that seemed a bit too much 😉*

Welcome to LMNN, a handcrafted neural network library written in Python. This project was created as part of a student initiative to delve deeper into the workings of artificial intelligence and neural networks. It is designed for educational purposes and aims to enhance the understanding of neural network fundamentals through practical implementation. While it is not intended to replace specialized libraries in the field, it serves as a platform for experimentation and learning.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features
- Customizable neural network layers and activation functions
- Implementation of weight and bias initializers
- Various loss functions for model training
- Test suite for validating components and ensuring functionality
- Utilizes `matplotlib` and `scikit-learn` for generating metrics and test visualizations

## Installation

Ensure you have Python 3.12 and the required libraries installed. You can set up the environment using the following steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/lucas066001/ManualNeuralNetwork.git
    cd lmnn
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install numpy==1.26 cupy==12.5 matplotlib scikit-learn
    ```

## Usage

You can start by exploring the provided examples and tests to understand how to build and train neural networks using this library. For detailed usage instructions, refer to the source code and the comments provided in the respective modules.

## Project Structure

The project is organized into the following directories:

```
lmnn/
├── activations/
│   ├── __init__.py
│   ├── relu.py
│   ├── sigmoid.py
│   ├── struct.py  # Abstract class for activation functions
│   └── ...
├── initializers/
│   ├── __init__.py
│   ├── he_initializer.py
│   ├── struct.py  # Abstract class for initializers
│   └── ...
├── layers/
│   ├── __init__.py
│   ├── dense.py
│   ├── dropout.py
│   ├── struct.py  # Abstract class for layers
│   └── ...
├── loss/
│   ├── __init__.py
│   ├── bce.py
│   ├── struct.py  # Abstract class for loss functions
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── layers_test.py
│   ├── model_test.py
│   └── ...
├── __init__.py
└── ...
```

In each of the subdirectories `initializers`, `activations`, `loss`, and `layers`, you will find a `struct.py` file that serves as an abstract class to unify the behavior of each element.

## Dependencies

The following table lists the required dependencies and their versions:

| Package       | Version  |
|---------------|----------|
| numpy         | 1.26     |
| cupy          | 12.5     |
| matplotlib    | latest   |
| scikit-learn  | latest   |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is a student initiative aimed at experimenting with AI to better understand its workings. It is not intended to replace any specialized library but rather to push the boundaries of personal knowledge and rekindle a love for mathematics. Special thanks to all the contributors and the open-source community for their continuous support and inspiration.

Happy coding!
