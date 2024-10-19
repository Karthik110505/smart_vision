    import cv2
    import pytesseract
    import streamlit as st
    from inference_sdk import InferenceHTTPClient
    import numpy as np

    # Initialize the inference client
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="VDWdiDCrMynoYfeMyEeC"
    )

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
            # Perform inference using the RoboFlow API
            result = CLIENT.infer(uploaded_file.name, model_id="expiry-date-detection-ssxnm/1")
            
            # Extract predictions from the result
            predictions = result.get('predictions', [])
            
            if predictions:
                # Sort predictions by confidence in descending order
                predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
                
                # Get the bounding box with the highest confidence (likely to be expiry date)
                best_prediction = predictions[0]
                x, y, w, h = int(best_prediction['x']), int(best_prediction['y']), int(best_prediction['width']), int(best_prediction['height'])
                
                # Extract the Region of Interest (ROI) based on the bounding box
                roi = image[y - h // 2:y + h // 2, x - w // 2:x + w // 2]
                
                # Use pytesseract to perform OCR on the ROI
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update path to Tesseract
                extracted_text = pytesseract.image_to_string(roi, config='--psm 6')  # Adjust config as needed
                
                # Style for the extracted expiry date text
                st.markdown(f"""
                    <div style="background-color:#f8f9fa; padding:10px; border-radius:5px;">
                        <h2 style="color:#343a40; font-family:Arial, sans-serif; font-weight:bold; text-align:center;">
                            Extracted Expiry Date
                        </h2>
                        <p style="font-size:28px; color:#007bff; font-family:Courier New, monospace; text-align:center;">
                            {extracted_text}
                    </div>
                """, unsafe_allow_html=True)
                
            else:
                st.write("No date predictions found.")
