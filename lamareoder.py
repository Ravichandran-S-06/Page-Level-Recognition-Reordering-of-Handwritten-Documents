#Reordering based on semantic Simialrity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
import requests
import json

SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text.
3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
Provide the extracted text without any additional comments."""

def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def process_single_image(image_path):
    """Perform OCR on a single image."""
    image_base64 = encode_image_to_base64(image_path)
    if not image_base64:
        print(f"Failed to encode image: {image_path}")
        return None

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",  # Ensure this URL matches your service endpoint
            json={
                "model": "llama3.2-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": SYSTEM_PROMPT,
                        "images": [image_base64],
                    },
                ],
            },
        )

        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        content = json_response.get("message", {}).get("content", "")
                        full_response += content
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e} for line: {line}")
            return full_response
        else:
            print(f"API Error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def reorder_paragraphs(paragraphs):
    """Reorder paragraphs based on semantic similarity scores."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    embeddings = model.encode(paragraphs)
    similarity_matrix = cosine_similarity(embeddings)
    scores = similarity_matrix.sum(axis=1)
    ordered_indices = np.argsort(scores)[::-1]  
    return [paragraphs[i] for i in ordered_indices]

if __name__ == "__main__":
    # List of image paths
    image_paths = [
        r"C:\Users\sravi\Downloads\2.jpg", 
        r"C:\Users\sravi\Downloads\1.jpg",
        r"C:\Users\sravi\Downloads\3.jpg",
    ]

    # Process each image and combine results
    extracted_texts = []
    for i, image_path in enumerate(image_paths, start=1):
        print(f"Processing Image {i}: {image_path}")
        text = process_single_image(image_path)
        if text:
            extracted_texts.append(text)

    if extracted_texts:
        # Split extracted text into paragraphs for reordering
        all_paragraphs = "\n\n".join(extracted_texts).split("\n\n")
        reordered_paragraphs = reorder_paragraphs(all_paragraphs)

        # Print the reordered text
        print("Reordered Text:")
        print("\n\n".join(reordered_paragraphs))
    else:
        print("Failed to process images or extract text.")

