# Computer Vision Parking Lot

## Overview

This project leverages computer vision techniques to analyze parking lot occupancy. The goal is to detect available parking spaces in real-time. The dataset used for this project is hosted on Google Drive and includes images and videos of parking lots for training and testing. The model is trained and tested with the images and deployed in Streamlit using video feeds.

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit

## Dataset

The dataset used in this project is available on Google Drive. It contains labeled images of parking lots for training and testing the model. You can access the dataset using the following link: [Dataset Link](https://drive.google.com/drive/folders/1mxbPv9i2dV00AL-6g2UYNpfK9ASPUcYI?usp=sharing).

## Running in Google Colab

This project was developed in Google Colab since image processing is computationally intensive. To run the notebooks, follow these steps:

1. Open the Google Colab notebooks:
    - Upload the provided notebook files.
2. Mount your Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3. Copy the dataset from the provided link to your Google Drive.
4. Update the `project_path` and `dataset_path` variables to match the location of your project folder in your Google Drive or local system.
5. Run the scripts in the order provided in the project folder to process the data and detect parking spaces.

## Streamlit Application

This project includes a Streamlit app for real-time parking lot detection. The app allows users to upload a video or use a sample video to detect parking spaces using a mask.

### How to Run the Streamlit App

1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the Streamlit app from inside the app folder (`5-APP`):
    ```bash
    streamlit run app.py
    ```
3. Features of the app:
    - Upload a video and mask file for parking lot detection.
    - Use a sample video provided in the app for testing.
    - Real-time detection of parking spaces using a mask overlay.

### Deployed Streamlit App

You can access the deployed Streamlit app using the following link: [Parking Lot Counter App](https://parking-lot-counter.streamlit.app).

## Future Improvements

- Enhance the model's robustness by incorporating additional datasets with diverse parking lot images.
- Implement a real-time video feed analysis for live parking lot monitoring.
- Develop a user-friendly web or mobile application to display parking availability.

## About the Author

Hi, I am Annie, a passionate physicist and data scientist with a keen interest in machine learning. Feel free to contact me on [LinkedIn](https://www.linkedin.com/in/annie-meneses-gonzalez-57bb9b145/).

## Contributing

Contributions are always welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
