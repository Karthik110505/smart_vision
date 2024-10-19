# Smart Vision

## Overview
The Smart Vision Project is a multiple vision-based tasks, including analyzing the freshness of fruits and vegetables, detecting expiry dates on product labels, and recognizing grocery items in images. The app leverages machine learning models hosted on RoboFlow for inference, OpenCV for image processing, and Pytesseract for Optical Character Recognition (OCR) to deliver robust and accurate results.

## Features
- **Freshness Detection**: Classifies items as Fresh, Semifresh, Semirotten, or Rotten based on uploaded images and return the fruit name along with freshness index
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
First, clone the repository to your local machine:


1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>```

2. **Install Required Packages**

    Ensure you have Python installed. Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
3. **Running the OCR Expiry Detection**
To test the OCR functionality for extracting expiry dates from images, navigate to the ocr_check directory and run the main.py file:
    ```bash
    cd ocr_check
    streamlit run main.py
    ```
4. **Running the Freshness Detection App**
For analyzing the freshness of fruits and vegetables, navigate to the ```freshness_detection``` directory and run the following:


    ```bash
    cd freshness_detection
    streamlit run main.py
    ```
3. **Running Grocery Item Detection**
To detect and count grocery items in an image, navigate to the ```recognition``` directory and run the ```main.py``` file:


    ```bash
    cd recognition  
    streamlit run main.py
    ```


## 🖼️ Sample Usage

### 1️⃣ Freshness Detection
Once you run the **Freshness Detection** app, upload an image of fruits/vegetables. The app will process the image and classify the freshness of each item. Example categories include:

- **Fresh**: High freshness score (100%).
- **Semifresh**: Medium freshness score (75%).
- **Semirotten**: Low freshness score (50%).
- **Rotten**: Very low freshness score (25%).

The app also displays bounding boxes around detected items, and labels them with their freshness levels.

### 2️⃣ Expiry Date Detection (OCR)
Upload a product label image containing expiry date information. The app will use OCR to extract and display the detected expiry date.

### 3️⃣ Grocery Item Detection
Upload an image containing grocery items. The app will detect and count the types of grocery items in the image, drawing bounding boxes around them and displaying their counts.

## 📧 Contact

For further information or inquiries, feel free to reach out at:

- **Veera Venkata Karthik Barrekala**
- **Email**: [bvvkarthik1155@gmail.com](mailto:bvvkarthik1155@gmail.com)
- **GitHub**: [VeeraVenkataKarthikBarrekala](https://github.com/Karthik110505)
