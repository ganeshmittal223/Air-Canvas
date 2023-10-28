from transformers import pipeline
from IPython.display import Image
from IPython.display import display, Image

# Initialize the OCR pipeline with the desired model
ocr = pipeline('image-to-text', model="microsoft/trocr-base-handwritten")

# Replace 'your_image.jpg' with the path to your own image file
your_image_filename = 'screenshot.png'

# Display an image


# Run OCR on your own image
result = ocr('screenshot.png')

# # Display the result
# print(result)

img_path = 'your_image_filename'
display(Image(filename=img_path))



