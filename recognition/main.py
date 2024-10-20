import cv2
import streamlit as st
import numpy as np
from collections import Counter
from inference_sdk import InferenceHTTPClient

# Initialize Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="VDWdiDCrMynoYfeMyEeC"
)

# Title for the Streamlit app
st.title("Grocery Item Detection")

# Display the information about the trained classes
trained_classes = """
The recognition model is trained on the following grocery items:
- Chocolatechip cookie
- Coca Cola Can 250ml
- Colgate 75g
- Fanta 500ml
- Fresher Guava Nectar 500ml
- Fruita Vitals Red Grapes 200ml
- Islamabad Tea 238g
- Kolson Slanty Jalapeno 18g
- LU Candi Biscuit Half Roll
- LU Prince Biscuit Half Roll
- Lays Masala
- Lays Wavy Mexican Chili 34gm
- Lifebuoy Soap
- Lipton Yellow Label Tea
- Meezan Ultra Rich Tea 190g
- Oreo Biscuit Half Roll
- Safeguard Soap Bar
- Shezan Apple 250ml
- Sooper Biscuit Ticky Pack
- Sunsilk Shampoo 160ml
- Super Crisp BBQ 30gm
- Supreme Tea 95g
- Tapal Danedar 95g
- Vaseline Lotion 100ml
- Kurkure Chutney
"""

st.info(trained_classes)

# Upload image using Streamlit file uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Process the uploaded image
if uploaded_file is not None:
    # Convert the uploaded file to a format OpenCV can read
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)

    # Save the original image temporarily for display
    temp_image_path = "temp_image.png"
    cv2.imwrite(temp_image_path, original_image)

    # Perform inference
    result = CLIENT.infer(temp_image_path, model_id="grocery-dataset-q9fj2/5")
    predictions = result.get('predictions', [])

    # Initialize a counter for the frequency of each class
    class_counter = Counter()

    # Create a copy of the original image for drawing bounding boxes
    processed_image = original_image.copy()

    # Loop through each prediction to draw bounding boxes and annotate class names
    for prediction in predictions:
        x = int(prediction['x'] - prediction['width'] / 2)
        y = int(prediction['y'] - prediction['height'] / 2)
        width = int(prediction['width'])
        height = int(prediction['height'])
        class_name = prediction['class']

        # Draw bounding boxes and class names on the processed image
        cv2.rectangle(processed_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(processed_image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Reduced font size

        # Update the class frequency count
        class_counter[class_name] += 1

    # Create two columns for displaying images side by side
    col1, col2 = st.columns(2)

    with col1:
        # Display the original uploaded image with reduced size
        st.image(original_image, channels="BGR", caption="Uploaded Image", use_column_width=True, width=400)

    with col2:
        # Display the processed image with bounding boxes with reduced size
        st.image(processed_image, channels="BGR", caption="Processed Image with Bounding Boxes", use_column_width=True, width=400)

    # Show the class frequency counts with increased font size
    st.subheader("Class Frequency", anchor=None)
    for class_name, count in class_counter.items():
        st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{class_name}: {count}</span>", unsafe_allow_html=True)
