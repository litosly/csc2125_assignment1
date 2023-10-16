from transformers import pipeline
from ultralytics import YOLO

# Image to text generator
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
# Image tags generator
model = YOLO("yolov8n-cls.pt")


def generate_image_caption(file_name):
    """Return the generated caption for image file_name

    Args:
        file_name (str): location of the file in the system

    Returns:
        str: alt text
    """
    return image_to_text(file_name)


def generate_image_tags(file_name, num_tags=5):
    """Generate image tags based with YOLO

    Args:
        file_name (str): location of the file in the system
        num_tags (int, optional): number of tags, maximum 5. Defaults to 5.

    Returns:
        list: list of string tags for the image
    """
    tags = []
    results = model(file_name)  # predict on an image
    if results:
        for id in results[0].probs.top5[:num_tags]:
            tags.append(results[0].names[id])
    return tags
