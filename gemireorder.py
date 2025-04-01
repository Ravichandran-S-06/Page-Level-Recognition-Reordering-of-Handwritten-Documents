import os
import re
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PIL import Image

# Configure Gemini API
genai.configure(api_key="")  # Replace with your actual API key

def perform_ocr(image_path):
    """Performs OCR and returns the transcribed text with enhanced accuracy."""
    try:
        img = Image.open(image_path)
        model = genai.GenerativeModel("gemini-1.5-flash")
        system_prompt = """Act as a precision OCR engine. Preserve:
        1. Exact text structure including page numbers
        2. Paragraph order
        3. Special formatting
        Return text exactly as seen, including page numbers in their original positions."""
        response = model.generate_content([system_prompt, img])
        ocr_text = response.text.strip() if response.text else ""
        print(f"OCR Output for {os.path.basename(image_path)}:\n{ocr_text[:100]}...\n")
        return ocr_text
    except Exception as e:
        print(f"OCR Error {image_path}: {e}")
        return ""

def extract_page_number(text, path=None):
    """Enhanced page number extraction with multiple strategies."""
    # First, try filename-based extraction as a fallback
    if path:
        filename_match = re.search(r'(\d+)', os.path.basename(path))
        filename_num = int(filename_match.group(1)) if filename_match else None
    else:
        filename_num = None
    
    # Check first 5 and last 5 lines for page numbers to be more thorough
    lines = text.split('\n')
    check_lines = lines[:5] + lines[-5:]
    
    patterns = [
        # Expanded patterns for better detection
        (r'(?i)(?:page|pg|p)[\.\s]*(\d+)', 1),       # "Page 1" format
        (r'(?i)\b(\d+)(?=\s*-\s*\d+\b)', 1),          # "3-5" format
        (r'^\s*(\d{1,3})\s*$', 1),                    # Standalone number
        (r'(?i)(?:roman numeral:\s*)([IVXLCDM]+)', 0),# Roman numerals
        (r'(?i)\b(\d+)\s*of\s*\d+\b', 1),             # "3 of 10" format
        (r'(?i)\b(\d+)\s*/\s*\d+\b', 1),              # "3/10" format
        (r'(?i)^(\d+)$', 1),                          # Number at start of line
        (r'(?i).*(\d+)\s*$', 1),                      # Number at the end of a line
    ]

    for line in check_lines:
        line = re.sub(r'\s+', ' ', line.strip())  # Normalize whitespace
        for pattern, group in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    num_str = match.group(group).upper()
                    if num_str.isdigit():
                        return int(num_str)
                    # Roman numeral conversion
                    roman_values = {'I':1, 'V':5, 'X':10, 'L':50, 
                                   'C':100, 'D':500, 'M':1000}
                    total = 0
                    prev_value = 0
                    for char in reversed(num_str):
                        value = roman_values.get(char, 0)
                        total += value if value >= prev_value else -value
                        prev_value = value
                    return total
                except (ValueError, KeyError):
                    continue
    
    # If no page number found in text, return filename number as fallback
    if filename_num is not None:
        print(f"No page number in text, using filename number: {filename_num}")
        return filename_num
        
    return None

def analyze_semantic_flow(texts, page_nums=None):
    """Enhanced semantic ordering with narrative context awareness."""
    # Check if we have enough texts to analyze
    if len(texts) <= 1:
        return texts
        
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Use whole texts for better context, but truncate if too large
    processed_texts = [text[:2000] for text in texts]  # Use first 2000 chars for embedding
    embeddings = model.encode(processed_texts)
    
    # Create similarity matrix with narrative flow weighting
    similarity_matrix = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(len(texts)):
            if i != j:  # Don't compare a text to itself
                # Calculate raw similarity
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                
                # Apply narrative flow bias - paragraphs tend to follow in sequence
                # Higher weight for natural progression (i → i+1)
                if j == i + 1:
                    sim *= 1.3  # Boost consecutive sequences
                
                # Factor in known page numbers if available
                if page_nums and page_nums[i] is not None and page_nums[j] is not None:
                    if page_nums[j] > page_nums[i]:
                        sim *= 1.2  # Boost if following known page numbering
                    elif page_nums[j] < page_nums[i]:
                        sim *= 0.8  # Reduce if going against known page numbering
                
                similarity_matrix[i][j] = sim
    
    # Find optimal path using greedy algorithm with narrative constraints
    current = 0  # Start with first text
    ordered_indices = [current]
    remaining = set(range(1, len(texts)))
    
    while remaining:
        last = ordered_indices[-1]
        # Find text with highest similarity to current text
        candidates = [(i, similarity_matrix[last][i]) for i in remaining]
        next_node = max(candidates, key=lambda x: x[1])[0]
        
        ordered_indices.append(next_node)
        remaining.remove(next_node)
    
    return [texts[i] for i in ordered_indices]

def reorder_texts(image_paths):
    """Improved document ordering with robust hybrid approach."""
    # First sort images by filename for initial order
    try:
        image_paths = sorted(image_paths, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
    except:
        print("Warning: Could not sort by filename numbers, using original order")
    
    texts = []
    page_nums = []
    filename_nums = []
    
    # Extract all text and page numbers first
    for path in image_paths:
        try:
            filename_num = int(re.search(r'\d+', os.path.basename(path)).group())
            filename_nums.append(filename_num)
        except:
            filename_nums.append(None)
            
        text = perform_ocr(path)
        texts.append(text)
        page_num = extract_page_number(text, path)
        page_nums.append(page_num)
        
        print(f"File: {os.path.basename(path)}, Detected page: {page_num}")
    
    # First attempt: Use detected page numbers if available for all pages
    all_pages_numbered = all(num is not None for num in page_nums)
    consistent_with_filenames = True
    
    # Check if page numbers are consistent with filename numbers
    if all_pages_numbered and all(fn is not None for fn in filename_nums):
        # Check if page numbers and filename numbers have the same order
        page_order = [p for p in sorted(range(len(page_nums)), key=lambda i: page_nums[i])]
        file_order = [f for f in sorted(range(len(filename_nums)), key=lambda i: filename_nums[i])]
        consistent_with_filenames = page_order == file_order
    
    # Decide on ordering strategy
    if all_pages_numbered and consistent_with_filenames:
        print("Using detected page numbers for ordering")
        # Simple sort by page number
        ordered_indices = sorted(range(len(texts)), key=lambda i: page_nums[i])
        ordered_texts = [texts[i] for i in ordered_indices]
    else:
        print("Page number detection unreliable, applying semantic analysis")
        # Apply semantic ordering with page number hints
        ordered_texts = analyze_semantic_flow(texts, page_nums)
    
    # Final verification - check if the ordered text makes narrative sense
    print("\nVerifying narrative flow with semantic analysis...")
    verification_order = analyze_semantic_flow(ordered_texts)
    
    # If semantic analysis suggests a different order, warn the user
    if verification_order != ordered_texts:
        print("WARNING: Semantic verification suggests alternate ordering.")
        print("Consider reviewing the output manually.")
    
    return ordered_texts

# Example usage with properly named files
image_paths = [
    "C:/Users/sravi/Downloads/3.jpg",
    "C:/Users/sravi/Downloads/2.jpg",
    "C:/Users/sravi/Downloads/1.jpg"
]

# Reordering
reordered_texts = reorder_texts(image_paths)

# Print final order
print("\nFinal Document Order:\n")
for i, text in enumerate(reordered_texts, start=1):
    preview = text[:50].replace('\n', ' ')  # Show preview of each page
    print(f"Page {i} - Preview: {preview}...\n{'-'*40}\n{text}\n")