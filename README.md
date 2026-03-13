# Truck Fill Level Estimation using YOLO

## Overview

This project estimates the **fill level inside a truck container** using computer vision.
It combines **object detection** and **instance segmentation** models to detect a truck and calculate how full the container is.

The system works in two main stages:

1. **Truck Detection** – Detect the truck in the image using a YOLO detection model.
2. **Content Segmentation** – Segment the truck container and its contents to calculate the fill percentage.

The final output displays:

* Truck bounding box
* Segmentation masks
* Estimated fill percentage
* Visual overlay on the original image

---

## Technologies Used

* Python
* OpenCV
* PyTorch
* Ultralytics YOLO
* NumPy

Libraries used in the code:

```
cv2
torch
numpy
ultralytics
```

---

## Project Structure

```
size-estimation
│
├── size_estimation.py        # Main script
├── weights
│   ├── truck.pt              # Truck detection model
│   └── size.pt               # Segmentation model
│
├── baselinevid
│   └── imag_123.jpg          # Example input image
│
└── result_image.jpg          # Output result
```

---

## How the System Works

### 1. Truck Detection

A YOLO detection model identifies trucks in the input image and produces:

* Bounding box coordinates
* Confidence score
* Class label

Example:

```
Truck #1
Bounding Box: (x1, y1) to (x2, y2)
Confidence: 92%
```

---

### 2. Truck Cropping

Once detected, the truck region is **cropped** from the image for further analysis.

This ensures the segmentation model only analyzes the truck area.

---

### 3. Segmentation

The segmentation model identifies:

* **Truck Box** (container area)
* **Content** inside the container

Masks are resized to match the truck crop dimensions.

---

### 4. Fill Percentage Calculation

Two methods are used:

#### Area-based fill

```
fill_percentage = (content_pixels / box_pixels) * 100
```

Where:

* `box_pixels` = pixels belonging to the truck container
* `content_pixels` = pixels belonging to the cargo

#### Height-based fill

The system also estimates fill level based on **vertical height of the content** relative to the container height.

This value is used for the final display.

---

### 5. Visualization

The program overlays segmentation masks:

* **Blue** → truck container
* **Green** → cargo

Fill level is displayed with color coding:

| Fill Level | Color  | Status |
| ---------- | ------ | ------ |
| <30%       | Red    | LOW    |
| 30–70%     | Yellow | MEDIUM |
| >70%       | Green  | HIGH   |

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/hishamsaif123/size-estimation.git
cd size-estimation
```

---

### 2. Install dependencies

```
pip install ultralytics
pip install opencv-python
pip install torch
pip install numpy
```

---

## Usage

Run the script:

```
python size_estimation.py
```

The script will:

1. Load the models
2. Process the input image
3. Detect trucks
4. Estimate fill percentage
5. Display the result
6. Save the output image

Output file:

```
result_image.jpg
```

---

## Example Output

The output image will show:

* Truck detection bounding box
* Segmentation masks
* Estimated fill percentage
* Fill status (LOW / MEDIUM / HIGH)

---

## Hardware Acceleration

The script automatically uses **GPU acceleration** if available:

```
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Otherwise it runs on CPU.

---

## Possible Improvements

Future improvements could include:

* Real-time video processing
* Multi-truck detection
* Integration with industrial camera systems
* Deployment on edge devices such as NVIDIA Jetson
* IoT integration for automated monitoring

---


## License

This project is for research and development purposes.
