import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Created by Shuvendu
# Paths to YOLO files
modelConf = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"
classesFile = "coco.names"

# Function for YOLO object detection
def yolo_out(image):
    # Load YOLO model
    net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    inpWidth = 416
    inpHeight = 416

    # Convert PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    yolo_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(yolo_layers)

    # Post-process detections
    frame = post_process(frame, outs, classes)

    # Convert the processed frame back to RGB for display in Streamlit
    result_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return result_image


# Post-processing to draw bounding boxes and labels
def post_process(frame, outs, classes):
    frame_height, frame_width = frame.shape[:2]
    conf_threshold = 0.5
    nms_threshold = 0.4

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:  # Ensure there are detections
        for i in indices.flatten():  # Use .flatten() to handle 1D/2D cases
            box = boxes[i]
            left, top, width, height = box
            draw_bounding_box(frame, class_ids[i], confidences[i], left, top, left + width, top + height, classes)

    return frame


# Draw bounding box
def draw_bounding_box(frame, class_id, confidence, left, top, right, bottom, classes):
    # Define a fixed color palette (e.g., based on class IDs)
    COLORS = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]

    # Assign a unique color based on the class ID
    color = COLORS[class_id % len(COLORS)]

    # Label for the bounding box
    label = f"{classes[class_id]}: {confidence:.2f}"

    # Draw the bounding box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw the label above the bounding box
    cv2.putText(
        frame, label, (left, max(top - 10, 20)),  # Avoid going above the frame
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    )


# Streamlit application
def main():
    st.set_page_config(page_title="Smart Vision - YOLO Object Detector", page_icon="👁", layout="wide")

    st.title("👁 Smart Vision: YOLO Object Detector")
    st.markdown(
        """<style>
        .stButton > button {
            color: white;
            background: #FF4B4B;
            border-radius: 10px;
        }
        </style>""", unsafe_allow_html=True
    )

    st.write("Upload an image, and the app will detect objects using YOLO.")

    # Sidebar for additional details
    with st.sidebar:
        st.header("How It Works")
        st.markdown(
            """
            1. Upload an image (JPG, PNG, JPEG).
            2. Click **'Unleash YOLO Magic ✨'** to process.
            3. View detected objects with bounding boxes and labels.
            """
        )
        st.info("Ensure that `yolov3-tiny.cfg`, `yolov3-tiny.weights`, and `coco.names` are correctly configured.")

    # Upload image for object detection
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if st.button("Unleash YOLO Magic ✨"):
        if uploaded_image is not None:
            with st.spinner("Detecting objects... 🔍"):
                # Load the uploaded image
                image = Image.open(uploaded_image)

                # Unleash YOLO Magic ✨
                result = yolo_out(image)

                # Display results
                st.image(result, caption="Detected Image", use_column_width=True)
            st.success("Detection complete! 🌟")
        else:
            st.error("Please upload an image to run detection.")

    # Footer
    st.markdown(
        "---\nCreated with ❤️ by Shuvendu Barik."
    )

if __name__ == "__main__":
    main()
