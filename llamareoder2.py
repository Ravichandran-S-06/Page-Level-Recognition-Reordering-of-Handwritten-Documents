import base64
import requests
import json
import numpy as np
import re
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text.
3. Include paragraph coordinates in [x1,y1,x2,y2] format before each paragraph, where x1,y1,x2,y2 are integers.
4. If any words or phrases are unclear, indicate this with [unclear] in your transcription.
Provide only the transcription without any additional comments."""

class HybridParagraphReordering:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.semantic_model = SentenceTransformer('all-mpnet-base-v2')

    def _cluster_columns(self, paragraphs, eps=50):
        x_coords = np.array([p['x1'] for p in paragraphs]).reshape(-1, 1)
        clustering = DBSCAN(eps=eps, min_samples=1).fit(x_coords)
        
        columns = {}
        for idx, label in enumerate(clustering.labels_):
            columns.setdefault(label, []).append(paragraphs[idx])
            
        return sorted(columns.values(), key=lambda c: np.mean([p['x1'] for p in c]))

    def _calculate_geometric_features(self, paragraphs):
        centers = np.array([[(p['x1']+p['x2'])/2, (p['y1']+p['y2'])/2] 
                          for p in paragraphs])
        dist_matrix = np.linalg.norm(centers[:, None] - centers, axis=2)
        return 1 / (1 + dist_matrix)

    def _calculate_semantic_features(self, paragraphs):
        texts = [p['text'] for p in paragraphs]
        embeddings = self.semantic_model.encode(texts)
        return cosine_similarity(embeddings)

    def _sinkhorn_normalization(self, matrix, iterations=10):
        matrix = np.exp(matrix)
        for _ in range(iterations):
            matrix /= matrix.sum(axis=1, keepdims=True)
            matrix /= matrix.sum(axis=0, keepdims=True)
        return matrix

    def reorder(self, paragraphs):
        geo_matrix = self._calculate_geometric_features(paragraphs)
        sem_matrix = self._calculate_semantic_features(paragraphs)
        combined = self.alpha * sem_matrix + (1 - self.alpha) * geo_matrix
        normalized = self._sinkhorn_normalization(combined)
        row_ind, col_ind = linear_sum_assignment(-normalized)
        return [paragraphs[i] for i in col_ind.argsort()]

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def parse_ocr_response(response):
    paragraphs = []
    current_para = {'text': '', 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    
    for line in response.split('\n'):
        if line.startswith('[') and ']' in line:
            if current_para['text']:  # Save previous paragraph
                paragraphs.append(current_para)
                current_para = {'text': '', 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
            
            coords = line.strip('[]').split(',')
            if len(coords) == 4:
                try:
                    current_para.update({
                        'x1': int(float(coords[0].strip())),
                        'y1': int(float(coords[1].strip())),
                        'x2': int(float(coords[2].strip())),
                        'y2': int(float(coords[3].strip()))
                    })
                except ValueError as e:
                    print(f"Warning: Invalid coordinate format in line: {line}. Error: {e}")
                    continue
        elif line.strip():
            current_para['text'] += line + '\n'
    
    if current_para['text']:
        paragraphs.append(current_para)
    
    return paragraphs

def extract_page_number(text):
    """Enhanced page number extraction with fixed regex patterns"""
    patterns = [
        r'(?im)^\s*(?:page|pg|p)[^\d\n]*?(\d+)\s*$',
        r'(?im)\b\d+\s*[-/]\s*\d+\b.*?(\d+)$',
        r'(?<![\.\d])\b(\d{1,3})\b(?![\.\d])',
        r'(?i)\b([ivxlcdm]{1,6})\b(?![.:]\w)',
    ]
    
    lines = text.split('\n')[:3] + text.split('\n')[-3:]
    for line in lines:
        line = line.strip()
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    num_str = match.group(1).upper()
                    if num_str.isdigit():
                        return int(num_str)
                    roman_values = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
                    total = 0
                    prev_value = 0
                    for char in reversed(num_str):
                        value = roman_values.get(char, 0)
                        if value < prev_value:
                            total -= value
                        else:
                            total += value
                        prev_value = value
                    return total
                except:
                    continue
    return None

def perform_ocr(image_path):
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2-vision",
                "messages": [{
                    "role": "user",
                    "content": SYSTEM_PROMPT,
                    "images": [base64_image],
                }],
            },
        )

        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        full_response += json_response.get("message", {}).get("content", "")
                    except json.JSONDecodeError:
                        continue
            return parse_ocr_response(full_response)
        else:
            print(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    image_paths = [
        r"C:\Users\sravi\Downloads\2.jpg", 
        r"C:\Users\sravi\Downloads\1.jpg",
        r"C:\Users\sravi\Downloads\3.jpg",
    ]

    all_paragraphs = []
    reorderer = HybridParagraphReordering()
    page_contents = []

    for image_path in image_paths:
        print(f"Processing image: {image_path}...")
        if paragraphs := perform_ocr(image_path):
            ordered_paragraphs = reorderer.reorder(paragraphs)
            page_text = " ".join(p['text'] for p in ordered_paragraphs)
            page_num = extract_page_number(page_text)
            
            if page_num is None:
                print(f"Warning: No page number detected in {image_path}. Using fallback order.")
                page_num = float('inf')
            
            page_contents.append((page_num, ordered_paragraphs))

    # Sort pages by detected page number
    page_contents.sort(key=lambda x: x[0])
    
    # Combine all paragraphs in correct order
    for _, paragraphs in page_contents:
        all_paragraphs.extend(paragraphs)

    if all_paragraphs:
        print("\nFinal Document Order:")
        for idx, para in enumerate(all_paragraphs, 1):
            print(f"Paragraph {idx} [({para['x1']},{para['y1']})-({para['x2']},{para['y2']})]:")
            print(para['text'].strip())
            print("-" * 50)
    else:
        print("No text extracted from images.")