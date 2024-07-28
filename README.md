<div align="center">

<p align="center">
  <img width="98" src="https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Fox/3D/fox_3d.png"/>
</p>

# Flaskyi AI Training Module (FATM)
</div>

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies](#technologies)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)

## Introduction
This is a training module for the Flaskyi AI model. The model is trained to generate high quality 4k images based on a given input image. The model is trained on a dataset of 6.5 billion images and has an accuracy of 93.5%.

## Features
- ``✅`` Train the Flaskyi AI model
- ``✅`` Use dataset
- ``✅`` Use pre-trained model
- ``✅`` Push to the huggingface hub

## Technologies
- Python
- Pip
- PyTorch
- Flaskyi

## Setup
Firstly, you need to create and activate a virtual environment:
```bash
python -m venv venv
```
and now activate the virtual environment:
```bash
# On Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate

# On MacOS
source venv/bin/activate
```

Then, you need to install the required packages:
```bash
pip install torch transformers diffusers datasets accelerate
```

## Usage
To train the model, run the following command:
```bash
python main.py
```

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.