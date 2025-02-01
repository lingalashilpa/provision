pip install opencv-python transformers torch
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load the model and processor (BLIP for image captioning as an example)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# Convert image from BGR (OpenCV format) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess the image and generate caption
inputs = processor(images=image_rgb, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

# Annotate image with the generated caption
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, caption, (30, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Display the image with annotation
cv2.imshow('Annotated Image', image)

# Save the annotated image if needed
cv2.imwrite('annotated_image.jpg', image)

# Wait for key press to close window
cv2.waitKey(0)
cv2.destroyAllWindows()
