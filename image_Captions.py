from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the BLIP processor from Hugging Face
# The processor prepares images and text for the model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize the BLIP model from Hugging Face
# This model is used for image captioning
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Load an image from the file system and convert it to RGB
image = Image.open("Picture.jpg").convert("RGB")

# Prepare the image for the model
# The processor converts the image into tensor format that PyTorch can use
inputs = processor(image, return_tensors="pt")

# Generate a caption for the image
# The model returns token IDs which represent the caption
outputs_ids = model.generate(**inputs)

# Decode the token IDs into a human-readable string
# skip_special_tokens=True removes special tokens like <s> or </s>
caption = processor.decode(outputs_ids[0],skip_special_tokens=True)

# Print the generated caption
print("Generated Caption:", caption)