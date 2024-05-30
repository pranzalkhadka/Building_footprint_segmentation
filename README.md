# Building Ffootprint Segmentation

## Overview
This project is aimed at performing building footprint segmentation using semantic segmentation technique. The segmentation task involves identifying and segmenting building footprints from satellite or aerial imagery.

## Table of Contents
1. [Introduction](#introduction)
3. [Installation](#installation)
4. [Technologies](#Technologies)
6. [Demo](#Demo)

## Introduction
Building footprint segmentation can be utilized in various applications such as urban planning, disaster response, and environmental monitoring. This project utilizes the UNet architecture, a popular convolutional neural network (CNN) architecture for image segmentation tasks, combined with semantic segmentation techniques to accurately detect building footprints.

## Installation
1. Normal Installation:
   ```bash
   git clone https://github.com/pranzalkhadkaBuilding_footprint_segmentation.git
   cd Building_footprint_segmentation
   python3 -m venv venv
   source venv/bin/activate
   pip install requirements.txt
   train.py
   uvicorn app:app --reload

2. Docker Installation:
    ```bash
    git clone https://github.com/pranzalkhadkaBuilding_footprint_segmentation.git
    cd Building_footprint_segmentation
    docker build -t some_name .
    docker run -p 8000:8000 name_you_used_above
    Access the application at http://localhost:8000 in your browser


## Technologies
1. Python
2. Tensorflow
3. Keras
4. FastAPI
5. Docker
6. MLflow

## Demo
![Prediction Image](/home/pranjal/Downloads/Building_footprint_segmentation/images/pred_image.jpg)