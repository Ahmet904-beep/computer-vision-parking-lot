# Computer Vision Parking Lot

## Overview

This project leverages computer vision techniques to analyze parking lot occupancy. The goal is to detect available parking spaces in real-time using image input. The dataset used for this project is hosted on Google Drive and includes images of parking lots for training and testing.

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

## Dataset

The dataset used in this project is available on Google Drive. It contains labeled images of parking lots for training and testing the model. You can access the dataset using the following link: [Dataset Link](https://drive.google.com/drive/folders/1mxbPv9i2dV00AL-6g2UYNpfK9ASPUcYI?usp=sharing).

## Running in Google Colab

To run this project in Google Colab, follow these steps:

1. Open the Google Colab notebooks:
    - Upload the provided notebooks files.
2. Mount your Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3. Copy the dataset from the provided link to your Google Drive.
4. Update the `project_path` and `dataset_path` variable to match the location of your project folder in your Google Drive or local system.
5. Run the scripts in the order provided in the project folder to process the data and detect parking spaces.

## Future Improvements

- Enhance the model's robustness by incorporating additional datasets with diverse parking lot images.
- Implement a real-time video feed analysis for live parking lot monitoring.
- Develop a user-friendly web or mobile application to display parking availability.
- Optimize the model for edge devices to enable on-site processing.

## About the Author

This project was developed by Annie, a passionate data scientist with a keen interest in computer vision and machine learning. Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/annie-meneses-gonzalez-57bb9b145/).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
