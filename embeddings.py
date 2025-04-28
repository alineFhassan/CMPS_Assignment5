# generate_embeddings.py

import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Paths
TRANSCRIPT_PATH = "data/transcript.json"
FRAMES_FOLDER = "data/frames/"
TEXT_EMBEDDINGS_PATH = "embeddings/text_embeddings.npy"
IMAGE_EMBEDDINGS_PATH = "embeddings/image_embeddings.npy"

def load_transcript(transcript_path):
    with open(transcript_path, "r") as f:
        segments = json.load(f)
    return segments

def generate_text_embeddings(segments, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [seg['text'] for seg in segments]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def generate_image_embeddings(frames_folder, model_name="openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)

    embeddings = []
    frame_files = sorted(os.listdir(frames_folder))
    for frame_file in tqdm(frame_files, desc="Encoding frames"):
        image = Image.open(os.path.join(frames_folder, frame_file)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model.get_image_features(**inputs)
        embeddings.append(outputs.cpu().detach().numpy()[0])
    return np.array(embeddings)

if __name__ == "__main__":
    os.makedirs("embeddings", exist_ok=True)

    print("Loading transcript...")
    segments = load_transcript(TRANSCRIPT_PATH)

    print("Generating text embeddings...")
    text_embeddings = generate_text_embeddings(segments)
    np.save(TEXT_EMBEDDINGS_PATH, text_embeddings)

    print("Generating image embeddings...")
    image_embeddings = generate_image_embeddings(FRAMES_FOLDER)
    np.save(IMAGE_EMBEDDINGS_PATH, image_embeddings)

    print("Done!")
