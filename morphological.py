import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    st.set_page_config(
        page_title="Morphological Operations",
        layout="wide"
    )

    st.title("Morphological Operations")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    st.sidebar.subheader("Morphological Operations")
    operation = st.sidebar.selectbox("Select an operation", ["Erosion", "Dilation", "Opening", "Closing"])

    window_size = st.sidebar.slider("Window Size", min_value=3, max_value=15, value=5, step=2)
    iterations = st.sidebar.slider("Iterations", min_value=1, max_value=10, value=1)

    kernel = np.ones((window_size, window_size), np.uint8)

    if uploaded_file is not None:
        img_array = np.frombuffer(uploaded_file.read(), np.uint8)
        imgDigit = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        st.sidebar.subheader("Original Digit Image")
        st.sidebar.image(cv2.cvtColor(imgDigit, cv2.COLOR_BGR2RGB), use_column_width=True)

        if operation == "Erosion":
            result = cv2.erode(imgDigit, kernel, iterations=iterations)
        elif operation == "Dilation":
            result = cv2.dilate(imgDigit, kernel, iterations=iterations)
        elif operation == "Opening":
            result = cv2.morphologyEx(imgDigit, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == "Closing":
            result = cv2.morphologyEx(imgDigit, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        st.subheader(f"{operation} Result")

        # Display images side by side for comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(imgDigit, cv2.COLOR_BGR2RGB), use_column_width=True)

        with col2:
            st.subheader(f"{operation} Filtered Image")
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)

if __name__ == "__main__":
    main()
