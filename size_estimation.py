import cv2
import torch
import numpy as np
from ultralytics import YOLO


# ============================
# Helper: resize YOLO mask to image size
# ============================
def resize_mask(mask, target_shape):
    """
    mask: (H, W) from YOLO (model input size)
    target_shape: image shape (H, W, C)
    """
    return cv2.resize(
        mask.astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)


# ============================
# 1. Load models
# ============================
truck_model = YOLO(r"../Yolo-wight/truck.pt")  # DETECTION
size_model = YOLO(r"../Yolo-wight/size.pt")  # SEGMENTATION

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================
# 2. Class names
# ============================
truck_classes = truck_model.names  # {0: 'truck'}
size_classes = size_model.names  # Check what classes your model has

print("Truck detection classes:", truck_classes)
print("Segmentation classes:", size_classes)

CONF_THRESHOLD = 0.4

# ============================
# 3. Display size (optional)
# ============================
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# ============================
# 4. Load single image
# ============================
image_path = r"baselinevid/imag_123.jpg"  # Your image path
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# ============================
# 5. Process the image
# ============================
print("\n" + "=" * 50)
print("Processing Image...")
print("=" * 50)

# 5.1 Truck detection
truck_results = truck_model(frame, device=device)[0]
print(f"Detected {len(truck_results.boxes)} trucks in the image")

truck_count = 0

for box in truck_results.boxes:
    conf = float(box.conf[0])
    if conf < CONF_THRESHOLD:
        print(f"Skipping truck detection (low confidence: {conf:.2f})")
        continue

    cls_id = int(box.cls[0])
    if truck_classes[cls_id] != 'truck':
        print(f"Skipping non-truck class: {truck_classes[cls_id]}")
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    truck_count += 1

    print(f"\nTruck #{truck_count}:")
    print(f"  Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"  Confidence: {conf:.2%}")
    print(f"  Truck Region Size: {x2 - x1}x{y2 - y1} pixels")

    # Draw truck bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(frame, f"Truck {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 3)

    # ============================
    # 5.2 Crop truck
    # ============================
    truck_crop = frame[y1:y2, x1:x2]
    if truck_crop.size == 0:
        print("  Warning: Empty truck crop, skipping...")
        continue

    print(f"  Cropped Truck Size: {truck_crop.shape[0]}x{truck_crop.shape[1]} pixels")

    # ============================
    # 5.3 Segmentation
    # ============================
    seg_result = size_model(truck_crop, device=device)[0]

    truck_box_mask = None
    content_mask = None

    if seg_result.masks is not None:
        masks = seg_result.masks.data.cpu().numpy()
        classes = seg_result.boxes.cls.cpu().numpy()

        print(f"  Found {len(masks)} segmentation masks")

        for i, cls in enumerate(classes):
            class_name = size_classes[int(cls)]
            print(f"    Mask {i}: Class '{class_name}'")

            # Based on your output, the class is 'Box' (with capital B)
            if class_name == 'Box' or class_name == 'box' or class_name == 'truck_box':
                truck_box_mask = masks[i]
                print(f"      -> Selected as Truck Box mask")
            elif class_name == 'content' or class_name == 'Content':
                content_mask = masks[i]
                print(f"      -> Selected as Content mask")
    else:
        print("  No segmentation masks found!")

    # ============================
    # 5.4 Draw segmentation masks
    # ============================
    overlay = truck_crop.copy()

    if truck_box_mask is not None:
        box_mask_resized = resize_mask(truck_box_mask, truck_crop.shape)
        overlay[box_mask_resized] = (255, 0, 0)  # BLUE → truck box

    if content_mask is not None:
        content_mask_resized = resize_mask(content_mask, truck_crop.shape)
        overlay[content_mask_resized] = (0, 255, 0)  # GREEN → content

    alpha = 0.4
    truck_crop[:] = cv2.addWeighted(
        overlay, alpha, truck_crop, 1 - alpha, 0
    )

    # ============================
    # 5.5 Fill percentage calculation
    # ============================
    if truck_box_mask is None:
        print("  Warning: No truck box mask found!")
        fill_percentage = 0
        box_pixels = 0
        content_pixels = 0
    elif content_mask is None:
        print("  Warning: No content mask found!")
        fill_percentage = 0
        box_mask_resized = resize_mask(truck_box_mask, truck_crop.shape)
        box_pixels = np.sum(box_mask_resized)
        content_pixels = 0
    else:
        # Resize masks to truck crop size
        box_mask_resized = resize_mask(truck_box_mask, truck_crop.shape)
        content_mask_resized = resize_mask(content_mask, truck_crop.shape)

        # Calculate pixel counts
        box_pixels = np.sum(box_mask_resized)
        content_pixels = np.sum(content_mask_resized)

        # Calculate fill percentage based on AREA (pixel count)
        if box_pixels > 0:
            fill_percentage = (content_pixels / box_pixels) * 100
        else:
            fill_percentage = 0

        print(f"  Truck Box Pixels: {box_pixels}")
        print(f"  Content Pixels: {content_pixels}")
        print(f"  Fill Percentage (Area): {fill_percentage:.1f}%")

        # Also calculate height-based fill (your original method)
        box_rows = np.any(box_mask_resized, axis=1)
        content_rows = np.any(content_mask_resized, axis=1)

        if np.any(box_rows) and np.any(content_rows):
            box_top = np.argmax(box_rows)
            box_bottom = len(box_rows) - 1 - np.argmax(box_rows[::-1])
            content_top = np.argmax(content_rows)
            content_bottom = len(content_rows) - 1 - np.argmax(content_rows[::-1])

            box_height = box_bottom - box_top
            content_height = content_bottom - content_top

            if box_height > 0:
                height_fill_percentage = (content_height / box_height) * 100
                print(f"  Fill Percentage (Height): {height_fill_percentage:.1f}%")

                # Use height-based fill for display
                fill_percentage = height_fill_percentage

    # ============================
    # 5.6 Display fill percentage
    # ============================
    # Round to nearest 10% for display
    display_percentage = int(round(fill_percentage / 10) * 10)

    # Color code based on fill level
    if fill_percentage < 30:
        color = (0, 0, 255)  # Red for low fill
        fill_status = "LOW"
    elif fill_percentage < 70:
        color = (0, 255, 255)  # Yellow for medium fill
        fill_status = "MEDIUM"
    else:
        color = (0, 255, 0)  # Green for high fill
        fill_status = "HIGH"

    cv2.putText(frame, f"Fill: {display_percentage}% ({fill_status})",
                (x1, y2 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                color, 3)

    # Also add detailed text
    cv2.putText(frame, f"Box: {box_pixels}px, Content: {content_pixels}px",
                (x1, y2 + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    print(f"  Final Display Fill Percentage: {display_percentage}% ({fill_status})")

print("\n" + "=" * 50)
print(f"Processing Complete! Found {truck_count} trucks.")
print("=" * 50)

# ============================
# 6. Display and save results
# ============================
# Resize for display (optional)
display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

# Show the result
cv2.imshow("Truck Fill Estimation - Result", display_frame)
print("\nPress any key to close the window...")
cv2.waitKey(0)

# Save the result
output_path = "result_image.jpg"
cv2.imwrite(output_path, frame)
print(f"\nResult saved to: {output_path}")

# ============================
# 7. Cleanup
# ============================
cv2.destroyAllWindows()