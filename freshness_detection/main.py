import cv2
import streamlit as st
import numpy as np
import base64
from inference_sdk import InferenceHTTPClient

# Initialize client for inference
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="VDWdiDCrMynoYfeMyEeC"
)

# Define freshness index mappings
freshness_index_map = {
    "Fresh": 100,
    "Semifresh": 75,
    "Semirotten": 50,
    "Rotten": 25
}

# Define the fruits and vegetables the model was trained on
trained_fruits_vegetables = """
The model is trained on the following fruits and vegetables:
- Apple (Fresh, Rotten, Semifresh, Semirotten)
- Banana (Fresh, Rotten, Semifresh, Semirotten)
- Carrot (Fresh, Rotten)
- Cucumber (Fresh, Rotten)
- Pepper (Fresh, Rotten)
- Potato (Fresh, Rotten)
- Tomato (Fresh, Rotten)
- Mango (Fresh, Rotten, Semifresh, Semirotten)
- Melon (Fresh, Rotten, Semifresh, Semirotten)
- Orange (Fresh, Rotten, Semifresh, Semirotten)
- Peach (Fresh, Rotten, Semifresh, Semirotten)
- Pear (Fresh, Rotten, Semifresh, Semirotten)
"""

# Streamlit App
st.title("Freshness Detection App")

# Display the information about the trained fruits and vegetables
st.info(trained_fruits_vegetables)

# Upload image through Streamlit
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to a byte array for inference
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Convert the byte array to an OpenCV image (BGR format)
    image = cv2.imdecode(file_bytes, 1)

    # Convert the image from BGR to RGB for proper display in Streamlit
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the uploaded image and the processed image side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    with col2:
        # Convert the image to base64-encoded string
        image_b64 = base64.b64encode(file_bytes).decode('utf-8')

        # Perform inference using the RoboFlow API
        result = CLIENT.infer(image_b64, model_id="freshness-nnryh/1")

        # Parse predictions from the result
        predictions = result['predictions']

        # Loop through the predictions and draw bounding boxes
        for pred in predictions:
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
            confidence = pred['confidence']
            class_name = pred['class']
            
            # Replace spaces with underscores and then split by underscores
            class_name = class_name.replace(" ", "_")
            if "_" in class_name:
                cls_values = class_name.split("_", 1)  # Split only once
                # Check if the first part is in freshness_index_map
                if cls_values[0] in freshness_index_map:
                    freshness_class = cls_values[0]
                    fruit_name = cls_values[1]
                else:
                    freshness_class = cls_values[1]
                    fruit_name = cls_values[0]  # Fallback if the first part is not a valid freshness class
            else:
                fruit_name = "Pear"
                freshness_class = "Semifresh"  # Default value for freshness class

            # Get freshness index, default to 0 if not found
            freshness_index = freshness_index_map.get(freshness_class, 0)

            # Calculate the top-left and bottom-right coordinates of the bounding box
            top_left = (int(x - width / 2), int(y - height / 2))
            bottom_right = (int(x + width / 2), int(y + height / 2))

            # Draw the rectangle (bounding box) with blue color and thinner line (1 pixel)
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)  # Blue color, thickness = 1

            # Add the fruit name and only the freshness index (numeric value)
            label = f"{fruit_name} - {freshness_index}"  # Only show the index number
            font_scale = 0.4  # Small font size
            cv2.putText(image, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1)

        # Convert the processed image from BGR to RGB for proper display in Streamlit
        image_rgb_with_boxes = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the processed image with bounding boxes
        st.image(image_rgb_with_boxes, caption='Processed Image with Bounding Boxes', use_column_width=True)
