import cv2
import pytesseract
import streamlit as st
from inference_sdk import InferenceHTTPClient
import numpy as np
import base64

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="VDWdiDCrMynoYfeMyEeC"
)

# Function to encode image in base64
def encode_image_to_base64(image_bytes):
    base64_bytes = base64.b64encode(image_bytes).decode('utf-8')
    return base64_bytes

# Streamlit App
st.title("Expiry Date Detection and OCR")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to byte array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Read the image using OpenCV
    image = cv2.imdecode(file_bytes, 1)
    
    # Convert the image to RGB for display purposes
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the uploaded image and the extracted expiry date side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    with col2:
        # Encode the image to base64
        base64_image = encode_image_to_base64(file_bytes)
        
        # Perform inference using the RoboFlow API
        result = CLIENT.infer(base64_image, model_id="expiry-date-detection-ssxnm/1")
        
        # Extract predictions from the result
        predictions = result.get('predictions', [])
        
        if predictions:
            # Create a list to store extracted dates
            extracted_dates = []

            # Loop through all predictions and extract text from each ROI
            for prediction in predictions:
                x, y, w, h = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
                
                # Extract the Region of Interest (ROI) based on the bounding box
                roi = image[y - h // 2:y + h // 2, x - w // 2:x + w // 2]
                
                # Use pytesseract to perform OCR on the ROI
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update path to Tesseract
                extracted_text = pytesseract.image_to_string(roi, config='--psm 6')  # Adjust config as needed
                
                # Append the extracted text to the list
                extracted_dates.append(extracted_text.strip())
            
            # Determine the manufacturing and expiry dates
            if len(extracted_dates) == 2:
                mfg_date = extracted_dates[0]
                expiry_date = extracted_dates[1]
                
                # Check if the last two characters of both dates are digits
                if mfg_date[-2:].isdigit() and expiry_date[-2:].isdigit():
                    # Swap dates if necessary
                    if int(mfg_date[-2:]) > int(expiry_date[-2:]):
                        mfg_date, expiry_date = expiry_date, mfg_date
                else:
                    mfg_date = "Not Detected"
                    expiry_date = "Not Detected"
                    st.write("Nothing Found.")
                    st.stop()
            
            elif len(extracted_dates) == 1:
                mfg_date = extracted_dates[0]
                expiry_date = "Not Detected"
            else:
                mfg_date = "Not Detected"
                expiry_date = "Not Detected"

            # Display the manufacturing and expiry dates
            st.markdown(f"""
                <div style="background-color:#f8f9fa; padding:10px; border-radius:5px;">
                    <h2 style="color:#343a40; font-family:Arial, sans-serif; font-weight:bold; text-align:center;">
                        Extracted Dates
                    </h2>
                    <p style="font-size:24px; color:#007bff; font-family:Courier New, monospace; text-align:center;">
                        Manufacturing Date: {mfg_date}
                    </p>
                    <p style="font-size:24px; color:#007bff; font-family:Courier New, monospace; text-align:center;">
                        Expiry Date: {expiry_date}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            last_prediction = predictions[-1]
            
        else:
            st.write("No date predictions found.")
