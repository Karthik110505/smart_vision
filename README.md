# Freshness Detection App

## Overview
The **Freshness Detection App** is a Streamlit application designed to analyze images of fruits and vegetables to determine their freshness levels. It leverages machine learning models hosted on RoboFlow for inference and utilizes OpenCV for image processing. Additionally, the app includes features for extracting expiry dates from product labels using Optical Character Recognition (OCR).

## Features
- **Freshness Detection**: Classifies items as Fresh, Semifresh, Semirotten, or Rotten based on uploaded images.
- **Expiry Date Detection**: Uses OCR to extract expiry dates from images of product labels.
- **Grocery Item Detection**: Identifies various grocery items in an image and counts their occurrences.

## Technologies Used
- **Python**: Programming language for backend logic.
- **Streamlit**: Framework for building the web application interface.
- **OpenCV**: Library for image processing tasks.
- **RoboFlow**: API for model inference.
- **Pytesseract**: OCR tool for text extraction from images.

## Installation
To run this application locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>```

2. Install Required Packages
    Ensure you have Python installed. Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
3. Run main.py 
    Go to the respective task folder and run the below code :
    ```bash
    streamlit run main.py
    ```
