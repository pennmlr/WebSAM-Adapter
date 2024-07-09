import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os


def draw_segmentations(image_path, json_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Read the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Extract segmentation data
    segmentations = data.get('segmentations', {}).get('majority-vote', [])

    # Draw each segmentation on the image
    for segmentation in segmentations:
        for segment in segmentation:
            points = np.array(segment[0], dtype=np.int32)

            # Check if points are properly formatted
            if points.size == 0 or points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("Segmentation points are not properly formatted or empty")

            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Display the image with segmentations
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    save_path = os.path.join(os.getcwd(), "test.png")
    print(save_path)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print(f"Segmented image saved as ")


# Example usage
image_path = 'screenshot.png'
json_path = 'ground-truth.json'
draw_segmentations(image_path, json_path)
