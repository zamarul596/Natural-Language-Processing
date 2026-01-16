"""
English Fake News Detection System - Web Application
University Project - Semester 7

FEATURES:
- Text input (paste news article)
- URL input (fetch from news websites)
- File upload (txt files)
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# ============================================================
# Load Model at Startup
# ============================================================
print("\nLoading model...")

with open('model/fake_news_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
vectorizer = model_data['vectorizer']
accuracy = model_data['accuracy']

print(f"Model loaded! Accuracy: {accuracy*100:.2f}%")
print(f"Vectorizer fitted: {hasattr(vectorizer, 'idf_')}")

# ============================================================
# Text Cleaning Function (matches training)
# ============================================================
def clean_text(text):
    text = str(text)
    
    # Remove news agency markers (same as training)
    text = re.sub(r'\(Reuters\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Reuters', '', text, flags=re.IGNORECASE)
    text = re.sub(r'WASHINGTON \(Reuters\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[A-Z]+\s*\(Reuters\)', '', text)
    text = re.sub(r'\(AP\)|\(AFP\)|\(CNN\)|\(BBC\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[A-Z\s,]+\s*[\(\)A-Za-z]*\s*[-–—]\s*', '', text)
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z0-9\s\.\,\!\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ============================================================
# URL Content Extraction
# ============================================================
def extract_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Try to find article content
        article = None
        
        # Common article selectors
        selectors = [
            'article',
            '.article-body',
            '.story-body',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.content-body',
            '[itemprop="articleBody"]',
            '.zn-body__paragraph',  # CNN
            '.article__body-text',  # BBC
        ]
        
        for selector in selectors:
            article = soup.select_one(selector)
            if article:
                break
        
        if article:
            paragraphs = article.find_all('p')
        else:
            paragraphs = soup.find_all('p')
        
        # Get title
        title = ""
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Extract text from paragraphs
        text_parts = [title]
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 30:  # Skip short paragraphs
                text_parts.append(text)
        
        content = ' '.join(text_parts)
        
        if len(content) < 100:
            return None, "Could not extract enough content from the URL"
        
        return content, None
        
    except requests.exceptions.Timeout:
        return None, "URL request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Could not fetch URL: {str(e)}"
    except Exception as e:
        return None, f"Error processing URL: {str(e)}"

# ============================================================
# Prediction Function
# ============================================================
def predict_news(text):
    if len(text.strip()) < 50:
        return None, None, None, "Please provide at least 50 characters of text"
    
    clean = clean_text(text)
    
    if len(clean) < 30:
        return None, None, None, "Not enough valid text content after processing"
    
    tfidf = vectorizer.transform([clean])
    prediction = model.predict(tfidf)[0]
    probability = model.predict_proba(tfidf)[0]
    
    result = "REAL" if prediction == 1 else "FAKE"
    confidence = max(probability) * 100
    
    return result, probability[0] * 100, probability[1] * 100, None

# ============================================================
# Routes
# ============================================================
@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy*100)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_type = data.get('type', 'text')
        
        text = ""
        extracted_title = ""
        
        # Handle different input types
        if input_type == 'url':
            url = data.get('url', '')
            if not url:
                return jsonify({'success': False, 'error': 'Please enter a URL'})
            
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            content, error = extract_from_url(url)
            if error:
                return jsonify({'success': False, 'error': error})
            
            text = content
            extracted_title = text[:100] + "..." if len(text) > 100 else text
            
        elif input_type == 'file':
            content = data.get('content', '')
            if not content:
                return jsonify({'success': False, 'error': 'No file content received'})
            text = content
            
        else:  # text input
            text = data.get('text', '')
        
        if not text:
            return jsonify({'success': False, 'error': 'No content to analyze'})
        
        # Predict
        result, fake_prob, real_prob, error = predict_news(text)
        
        if error:
            return jsonify({'success': False, 'error': error})
        
        response = {
            'success': True,
            'result': result,
            'confidence': round(max(fake_prob, real_prob), 1),
            'fake_prob': round(fake_prob, 1),
            'real_prob': round(real_prob, 1),
            'text_length': len(text),
            'input_type': input_type
        }
        
        if extracted_title:
            response['extracted_preview'] = extracted_title
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})

# ============================================================
# Run Server
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("FAKE NEWS DETECTION WEB APP")
    print("=" * 50)
    print(f"Model Accuracy: {accuracy*100:.2f}%")
    print(f"Open: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000)
