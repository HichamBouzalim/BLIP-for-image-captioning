from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize the model from Hugging Face
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Load an image
image = Image.open("Picture.jpg").convert("RGB")

# Prepare the image
inputs = processor(image, return_tensors="pt")

# Generate captions
outputs_ids = model.generate(**inputs)
caption = processor.decode(outputs_ids[0],skip_special_tokens=True)


print("Generated Caption:", caption)