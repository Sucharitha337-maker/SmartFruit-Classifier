'''
from PIL import Image

# Define your image path and desired output path
for i in range(2,6):
    image_path = "../fruit_all/testing_shoot"+str(i)+".jpg"  # Replace with your actual path
    output_path = "../fruit_all/fruits-360_dataset/fruits-360/self_test/pear/testing_shoot"+str(i)+"_resized_image.jpg"

    img = Image.open(image_path)

# Resize the image to 128x128 pixels using resize method
    resized_img = img.resize((100,100))  # Antialiasing for smoother resize

# Save the resized image
    resized_img.save(output_path)

    print("Image resized successfully!")

'''

from PIL import Image

def resize_with_aspect_ratio(image, width=None, height=None):
    # Get original dimensions
    original_width, original_height = image.size

    # If both width and height are None, return the original image
    if width is None and height is None:
        return image

    # Calculate the new dimensions preserving the aspect ratio
    if width is None:
        ratio = height / float(original_height)
        new_width = int(original_width * ratio)
        new_height = height
    else:
        ratio = width / float(original_width)
        new_width = width
        new_height = int(original_height * ratio)

    # Resize and return the image
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

for i in range(2,6):
# Open an image file
    image = Image.open("../fruit_all/testing_shoot"+str(i)+".jpg")

# Resize the image while keeping the aspect ratio
    resized_image = resize_with_aspect_ratio(image, width=100, height=100)

    resized_image.save("../fruit_all/fruits-360_dataset/fruits-360/self_test/pear/testing_shoot"+str(i)+"_resized_image.jpg")