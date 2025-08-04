#!/bin/bash

# Complete Implementation Script for Multimodal RAG Assistant
# This script replaces all placeholder files with full implementations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Adding Complete Implementation to Multimodal RAG Assistant${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "templates" ] || [ ! -d "static" ]; then
    echo -e "${RED}❌ Error: This doesn't appear to be a multimodal RAG project directory.${NC}"
    echo -e "${RED}Please run this script from inside the project directory created by setup.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Project directory confirmed${NC}"

# Backup existing files
echo -e "\n${BLUE}💾 Creating backups of existing files...${NC}"
mkdir -p backups
cp app.py backups/app.py.backup 2>/dev/null || true
cp templates/chat_template.html backups/chat_template.html.backup 2>/dev/null || true
cp static/css/styles.css backups/styles.css.backup 2>/dev/null || true
cp static/js/app.js backups/app.js.backup 2>/dev/null || true

echo -e "${GREEN}✅ Backups created in ./backups/${NC}"

# Replace app.py with complete implementation
echo -e "\n${BLUE}🐍 Implementing complete Flask application...${NC}"

cat > app.py << 'EOF'
#!/usr/bin/env python3
"""
Multimodal RAG Service Manual Assistant
Advanced Flask application with support for text, tables, and diagrams
"""

from flask import Flask, request, jsonify, session, render_template, send_from_directory
from openai import OpenAI
import os
import uuid
import chromadb
from chromadb.config import Settings
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
from io import BytesIO
import re
from datetime import datetime
import hashlib
import dotenv
import requests
import time
from urllib.parse import urlparse, urljoin
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import base64
import json
import logging
from werkzeug.utils import secure_filename
import threading

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE_MB', 50)) * 1024 * 1024

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB
script_dir = os.path.dirname(os.path.abspath(__file__))
chroma_db_path = os.path.join(script_dir, os.getenv('CHROMA_DB_PATH', 'chroma_db'))
web_downloads_path = os.path.join(script_dir, os.getenv('WEB_DOWNLOADS_PATH', 'web_downloads'))

# Create directories if they don't exist
os.makedirs(chroma_db_path, exist_ok=True)
os.makedirs(web_downloads_path, exist_ok=True)
os.makedirs('logs', exist_ok=True)

chroma_client = chromadb.PersistentClient(path=chroma_db_path)

@dataclass
class DocumentElement:
    """Represents different types of content elements"""
    element_id: str
    element_type: str  # 'text', 'table', 'figure', 'diagram', 'heading'
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    content: str  # text content or description
    metadata: Dict[str, Any]
    image_data: Optional[bytes] = None  # for figures/diagrams
    table_data: Optional[Dict] = None   # structured table data
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None

class MultimodalServiceManualProcessor:
    """Advanced processor for service manuals with multimodal content"""
    
    def __init__(self, openai_client: OpenAI, chroma_client: chromadb.Client):
        self.openai_client = openai_client
        self.chroma_client = chroma_client
        
        # Create separate collections for different content types
        self.text_collection = self._get_or_create_collection("service_manual_text")
        self.table_collection = self._get_or_create_collection("service_manual_tables") 
        self.figure_collection = self._get_or_create_collection("service_manual_figures")
        
        logger.info("Multimodal processor initialized with separate collections")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(name)
        except ValueError:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def process_service_manual(self, pdf_path: str, manual_name: str) -> Tuple[List[DocumentElement], Dict[str, int]]:
        """Process a service manual PDF into structured elements"""
        logger.info(f"Processing service manual: {manual_name}")
        elements = []
        stats = {"text": 0, "tables": 0, "figures": 0, "pages": 0}
        
        try:
            # Open PDF with PyMuPDF for better layout detection
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            max_pages = int(os.getenv('MAX_PAGES_PROCESS', 100))
            
            if total_pages > max_pages:
                logger.warning(f"PDF has {total_pages} pages, processing first {max_pages}")
                total_pages = max_pages
            
            for page_num in range(total_pages):
                try:
                    page = pdf_doc[page_num]
                    page_elements = self._process_page(page, page_num, manual_name, pdf_path)
                    elements.extend(page_elements)
                    stats["pages"] += 1
                    
                    # Count element types
                    for element in page_elements:
                        if element.element_type in ["text", "heading"]:
                            stats["text"] += 1
                        elif element.element_type == "table":
                            stats["tables"] += 1
                        elif element.element_type in ["figure", "diagram"]:
                            stats["figures"] += 1
                            
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    continue
            
            pdf_doc.close()
            logger.info(f"Processed {len(elements)} elements from {stats['pages']} pages")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
        
        return elements, stats
    
    def _process_page(self, page: fitz.Page, page_num: int, manual_name: str, pdf_path: str) -> List[DocumentElement]:
        """Process a single page to extract text, tables, and figures"""
        elements = []
        
        try:
            # Get page layout information
            page_dict = page.get_text("dict")
            
            # 1. Extract text blocks with proper structure
            text_elements = self._extract_text_blocks(page_dict, page_num, manual_name)
            elements.extend(text_elements)
            
            # 2. Extract tables using pdfplumber
            if os.getenv('TABLE_EXTRACTION_ENABLED', 'true').lower() == 'true':
                table_elements = self._extract_tables_from_page(pdf_path, page_num, manual_name)
                elements.extend(table_elements)
            
            # 3. Extract figures and diagrams
            if os.getenv('OCR_ENABLED', 'true').lower() == 'true':
                figure_elements = self._extract_figures_from_page(page, page_num, manual_name)
                elements.extend(figure_elements)
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
        
        return elements
    
    def _extract_text_blocks(self, page_dict: Dict, page_num: int, manual_name: str) -> List[DocumentElement]:
        """Extract structured text blocks maintaining hierarchy"""
        elements = []
        
        for block_idx, block in enumerate(page_dict.get("blocks", [])):
            if "lines" not in block:
                continue
                
            # Combine lines in block
            block_text = ""
            font_sizes = []
            is_bold = False
            
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"] + " "
                    font_sizes.append(span["size"])
                    if span.get("flags", 0) & 2**4:  # Bold flag
                        is_bold = True
            
            block_text = block_text.strip()
            if len(block_text) < 10:  # Skip very short blocks
                continue
                
            # Determine if this is a heading based on font size/style
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
            is_heading = avg_font_size > 14 or is_bold
            
            element = DocumentElement(
                element_id=f"{manual_name}_p{page_num}_text_{block_idx}",
                element_type="heading" if is_heading else "text",
                page_number=page_num,
                bbox=(block["bbox"][0], block["bbox"][1], block["bbox"][2], block["bbox"][3]),
                content=block_text,
                metadata={
                    "manual_name": manual_name,
                    "avg_font_size": avg_font_size,
                    "is_heading": is_heading,
                    "is_bold": is_bold,
                    "block_type": "text",
                    "upload_time": datetime.now().isoformat()
                }
            )
            elements.append(element)
            
        return elements
    
    def _extract_tables_from_page(self, pdf_path: str, page_num: int, manual_name: str) -> List[DocumentElement]:
        """Extract tables using pdfplumber with structure preservation"""
        elements = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    tables = page.extract_tables()
                    
                    for i, table in enumerate(tables):
                        if not table or len(table) < 2:
                            continue
                            
                        # Convert table to structured format
                        try:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            df = df.dropna(how='all').fillna('')  # Clean empty rows/cells
                            
                            if len(df) == 0:
                                continue
                            
                            # Create linearized text version for embedding
                            linearized = self._linearize_table(df)
                            
                            # Generate table summary using LLM
                            table_summary = self._generate_table_summary(df, manual_name)
                            
                            element = DocumentElement(
                                element_id=f"{manual_name}_p{page_num}_table_{i}",
                                element_type="table",
                                page_number=page_num,
                                bbox=(0, 0, page.width, page.height),  # Approximate bbox
                                content=f"{table_summary}\n\n{linearized}",
                                metadata={
                                    "manual_name": manual_name,
                                    "table_rows": len(df),
                                    "table_cols": len(df.columns),
                                    "table_summary": table_summary,
                                    "block_type": "table",
                                    "upload_time": datetime.now().isoformat()
                                },
                                table_data={
                                    "dataframe": df.to_dict(),
                                    "csv": df.to_csv(index=False),
                                    "html": df.to_html(index=False, classes="table table-striped")
                                }
                            )
                            elements.append(element)
                            
                        except Exception as e:
                            logger.error(f"Error processing table {i} on page {page_num}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num}: {e}")
            
        return elements
    
    def _extract_figures_from_page(self, page: fitz.Page, page_num: int, manual_name: str) -> List[DocumentElement]:
        """Extract figures, diagrams, and images"""
        elements = []
        
        # Get images from page
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Extract image
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # Ensure it's not CMYK
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(BytesIO(img_data))
                    
                    # Skip very small images (likely decorative)
                    if pil_image.width < 100 or pil_image.height < 100:
                        continue
                    
                    # Perform OCR on the image to extract text labels
                    ocr_text = self._extract_text_from_image(pil_image)
                    
                    # Generate image description using vision model (if enabled)
                    if os.getenv('VISION_API_ENABLED', 'true').lower() == 'true':
                        image_description = self._generate_image_description(pil_image, manual_name, ocr_text)
                    else:
                        image_description = f"Technical diagram from {manual_name}"
                        if ocr_text.strip():
                            image_description += f" with text labels: {ocr_text[:100]}"
                    
                    # Combine OCR text and description for embedding
                    combined_text = f"{image_description}\n\nOCR Text: {ocr_text}".strip()
                    
                    element = DocumentElement(
                        element_id=f"{manual_name}_p{page_num}_fig_{img_index}",
                        element_type="figure",
                        page_number=page_num,
                        bbox=(0, 0, pil_image.width, pil_image.height),
                        content=combined_text,
                        metadata={
                            "manual_name": manual_name,
                            "image_width": pil_image.width,
                            "image_height": pil_image.height,
                            "has_ocr_text": len(ocr_text.strip()) > 0,
                            "block_type": "figure",
                            "ocr_text": ocr_text,
                            "description": image_description,
                            "upload_time": datetime.now().isoformat()
                        },
                        image_data=img_data
                    )
                    elements.append(element)
                    
                pix = None
                
            except Exception as e:
                logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                continue
                
        return elements
    
    def _linearize_table(self, df: pd.DataFrame) -> str:
        """Convert table to searchable text format"""
        linearized = []
        
        # Add column headers
        headers = " | ".join(str(col) for col in df.columns)
        linearized.append(f"Columns: {headers}")
        
        # Add rows with key-value pairs
        for _, row in df.iterrows():
            row_parts = []
            for col, val in row.items():
                if pd.notna(val) and str(val).strip():
                    row_parts.append(f"{col}: {val}")
            if row_parts:
                linearized.append(" | ".join(row_parts))
                
        return "\n".join(linearized)
    
    def _generate_table_summary(self, df: pd.DataFrame, manual_name: str) -> str:
        """Generate a descriptive summary of the table using LLM"""
        try:
            # Create a sample of the table for the LLM
            sample_rows = min(3, len(df))
            table_sample = df.head(sample_rows).to_string(index=False)
            
            prompt = f"""Analyze this table from a service manual and provide a brief, descriptive summary:

Manual: {manual_name}
Table data (first {sample_rows} rows):
{table_sample}

Provide a 1-2 sentence summary that describes:
1. What type of data this table contains
2. Key specifications, measurements, or parameters shown
3. Any units of measurement

Summary:"""

            response = self.openai_client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating table summary: {e}")
            return f"Table with {len(df)} rows and {len(df.columns)} columns from {manual_name}"
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        try:
            # Configure tesseract for technical diagrams
            config = os.getenv('TESSERACT_CONFIG', '--oem 3 --psm 6')
            ocr_text = pytesseract.image_to_string(image, config=config)
            return ocr_text.strip()
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def _generate_image_description(self, image: Image.Image, manual_name: str, ocr_text: str) -> str:
        """Generate description of image using vision model"""
        try:
            # Convert image to base64 for OpenAI Vision API
            buffered = BytesIO()
            # Resize large images to reduce API costs
            if image.width > 1024 or image.height > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            prompt = f"""Describe this technical diagram/image from a service manual. Focus on:
1. Type of diagram (wiring, schematic, exploded view, etc.)
2. Key components visible
3. Part numbers, labels, or callouts
4. Technical relationships shown

Manual: {manual_name}
OCR Text found: {ocr_text}

Provide a detailed but concise description:"""

            response = self.openai_client.chat.completions.create(
                model=os.getenv('OPENAI_VISION_MODEL', 'gpt-4-vision-preview'),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            return f"Technical diagram from {manual_name} (description unavailable)"
    
    def embed_and_store_elements(self, elements: List[DocumentElement]) -> Dict[str, int]:
        """Generate embeddings and store elements in appropriate collections"""
        stored_counts = {"text": 0, "tables": 0, "figures": 0, "errors": 0}
        
        for element in elements:
            try:
                # Generate text embedding for all elements
                if element.content:
                    text_embedding = self._get_text_embedding(element.content)
                    element.text_embedding = text_embedding
                
                # Store in appropriate collection
                self._store_element(element)
                
                # Update counts
                if element.element_type in ["text", "heading"]:
                    stored_counts["text"] += 1
                elif element.element_type == "table":
                    stored_counts["tables"] += 1
                elif element.element_type in ["figure", "diagram"]:
                    stored_counts["figures"] += 1
                
            except Exception as e:
                logger.error(f"Error processing element {element.element_id}: {e}")
                stored_counts["errors"] += 1
        
        return stored_counts
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def _store_element(self, element: DocumentElement):
        """Store element in appropriate ChromaDB collection"""
        
        # Prepare metadata
        metadata = element.metadata.copy()
        metadata.update({
            "element_type": element.element_type,
            "page_number": element.page_number,
            "bbox": str(element.bbox),
            "element_id": element.element_id
        })
        
        # Choose collection based on element type
        if element.element_type in ["text", "heading"]:
            collection = self.text_collection
        elif element.element_type == "table":
            collection = self.table_collection
            # Add table-specific metadata
            if element.table_data:
                metadata["table_html"] = element.table_data.get("html", "")
                metadata["table_csv"] = element.table_data.get("csv", "")
        elif element.element_type in ["figure", "diagram"]:
            collection = self.figure_collection
        else:
            collection = self.text_collection
        
        # Store with text embedding
        collection.add(
            ids=[element.element_id],
            documents=[element.content],
            embeddings=[element.text_embedding],
            metadatas=[metadata]
        )
    
    def hybrid_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Perform hybrid search across all element types"""
        logger.info(f"Performing hybrid search for: {query}")
        
        # Route query to appropriate collections based on content
        results = []
        query_lower = query.lower()
        
        # Search text collection for general queries
        if not any(keyword in query_lower for keyword in ["diagram", "figure", "table", "chart", "spec"]):
            text_results = self._search_collection(self.text_collection, query, n_results//2)
            results.extend(text_results)
        
        # Search table collection for specification queries
        if any(keyword in query_lower for keyword in 
               ['spec', 'specification', 'table', 'rating', 'dimension', 'measurement', 'value', 'torque']):
            table_results = self._search_collection(self.table_collection, query, n_results//2)
            results.extend(table_results)
        
        # Search figure collection for visual queries  
        if any(keyword in query_lower for keyword in
               ['diagram', 'figure', 'wiring', 'schematic', 'drawing', 'layout', 'connection', 'part']):
            figure_results = self._search_collection(self.figure_collection, query, n_results//2)
            results.extend(figure_results)
        
        # If no specific type detected, search all collections
        if not results:
            text_results = self._search_collection(self.text_collection, query, n_results//3)
            table_results = self._search_collection(self.table_collection, query, n_results//3)
            figure_results = self._search_collection(self.figure_collection, query, n_results//3)
            results.extend(text_results + table_results + figure_results)
        
        # Sort by relevance score and return top results
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return results[:n_results]
    
    def _search_collection(self, collection, query: str, n_results: int) -> List[Dict]:
        """Search a specific collection"""
        try:
            if collection.count() == 0:
                return []
            
            query_embedding = self._get_text_embedding(query)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count()),
                include=["documents", "distances", "metadatas"]
            )
            
            formatted_results = []
            if results["documents"]:
                for i, (doc, distance, metadata) in enumerate(zip(
                    results["documents"][0],
                    results["distances"][0], 
                    results["metadatas"][0]
                )):
                    formatted_results.append({
                        "content": doc,
                        "similarity": 1 - distance,
                        "metadata": metadata,
                        "element_type": metadata.get("element_type", "text")
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# Initialize the multimodal processor
try:
    multimodal_processor = MultimodalServiceManualProcessor(openai_client, chroma_client)
    logger.info("Multimodal processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize multimodal processor: {e}")
    multimodal_processor = None

# Legacy collection for backward compatibility
collection_name = "pdf_documents"
try:
    legacy_collection = chroma_client.get_collection(collection_name)
except:
    legacy_collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

# Helper functions for legacy text processing
def extract_text_from_pdf(pdf_file):
    """Legacy PDF text extraction function"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def chunk_text(text, chunk_size=1000, overlap=200):
    """Legacy text chunking function"""
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            last_space = text.rfind(' ', start, end)
            
            boundary = max(last_period, last_newline, last_space)
            if boundary > start:
                end = boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def get_embeddings(texts):
    """Get embeddings from OpenAI"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        raise Exception(f"Error getting embeddings: {str(e)}")

# Routes
@app.route('/')
def index():
    """Main page"""
    if 'conversation' not in session:
        session['conversation'] = []
    return render_template('chat_template.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        stats = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'collections': {
                'legacy': legacy_collection.count() if legacy_collection else 0,
            }
        }
        
        if multimodal_processor:
            stats['collections'].update({
                'text': multimodal_processor.text_collection.count(),
                'tables': multimodal_processor.table_collection.count(),
                'figures': multimodal_processor.figure_collection.count(),
            })
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get database statistics"""
    try:
        if multimodal_processor:
            # Count unique documents across collections
            all_metadata = []
            
            try:
                text_data = multimodal_processor.text_collection.get(include=['metadatas'])
                all_metadata.extend(text_data['metadatas'])
            except:
                pass
            
            try:
                table_data = multimodal_processor.table_collection.get(include=['metadatas'])
                all_metadata.extend(table_data['metadatas'])
            except:
                pass
            
            try:
                figure_data = multimodal_processor.figure_collection.get(include=['metadatas'])
                all_metadata.extend(figure_data['metadatas'])
            except:
                pass
            
            unique_docs = len(set(meta.get('manual_name', 'unknown') for meta in all_metadata))
            
            stats = {
                'documents': unique_docs,
                'text_chunks': multimodal_processor.text_collection.count(),
                'tables': multimodal_processor.table_collection.count(),
                'figures': multimodal_processor.figure_collection.count(),
                'total_chunks': (
                    multimodal_processor.text_collection.count() +
                    multimodal_processor.table_collection.count() +
                    multimodal_processor.figure_collection.count()
                ),
                'service_manuals': unique_docs  # For legacy compatibility
            }
        else:
            # Fallback to legacy collection
            total_chunks = legacy_collection.count() if legacy_collection else 0
            stats = {
                'documents': 0,
                'text_chunks': total_chunks,
                'tables': 0,
                'figures': 0,
                'total_chunks': total_chunks,
                'service_manuals': 0
            }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'documents': 0,
            'text_chunks': 0,
            'tables': 0,
            'figures': 0,
            'total_chunks': 0,
            'service_manuals': 0
        })

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF with multimodal support"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['pdf']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed'})
        
        processing_steps = []
        filename = secure_filename(file.filename)
        
        # Save uploaded file temporarily
        temp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        try:
            if multimodal_processor:
                # Use multimodal processing
                processing_steps.append({
                    'title': '🔍 Analyzing PDF structure',
                    'status': 'processing',
                    'details': f'Processing: {filename}'
                })
                
                # Process the PDF
                elements, stats = multimodal_processor.process_service_manual(temp_path, filename)
                
                processing_steps[-1]['status'] = 'complete'
                processing_steps[-1]['details'] = f'Found {len(elements)} elements across {stats["pages"]} pages'
                
                # Add content breakdown
                processing_steps.append({
                    'title': '📊 Content Analysis',
                    'status': 'complete',
                    'details': f'Text blocks: {stats["text"]}, Tables: {stats["tables"]}, Figures: {stats["figures"]}'
                })
                
                # Generate embeddings and store
                processing_steps.append({
                    'title': '🧠 Generating embeddings',
                    'status': 'processing',
                    'details': 'Converting content to vector embeddings...'
                })
                
                stored_counts = multimodal_processor.embed_and_store_elements(elements)
                
                processing_steps[-1]['status'] = 'complete'
                processing_steps[-1]['details'] = f'Stored {sum(stored_counts.values()) - stored_counts["errors"]} elements'
                
                if stored_counts["errors"] > 0:
                    processing_steps.append({
                        'title': '⚠️ Processing warnings',
                        'status': 'complete',
                        'details': f'{stored_counts["errors"]} elements had processing errors'
                    })
                
                return jsonify({
                    'message': f'Successfully processed "{filename}" with multimodal analysis',
                    'chunks_added': sum(stored_counts.values()) - stored_counts["errors"],
                    'content_breakdown': stored_counts,
                    'filename': filename,
                    'processing_steps': processing_steps
                })
                
            else:
                # Fallback to legacy processing
                processing_steps.append({
                    'title': '📝 Extracting text (legacy mode)',
                    'status': 'processing',
                    'details': 'Using basic text extraction...'
                })
                
                with open(temp_path, 'rb') as pdf_file:
                    text = extract_text_from_pdf(pdf_file)
                
                if not text.strip():
                    return jsonify({
                        'error': 'No text could be extracted from the PDF',
                        'processing_steps': processing_steps
                    })
                
                processing_steps[-1]['status'] = 'complete'
                processing_steps[-1]['details'] = f'Extracted {len(text)} characters'
                
                # Chunk the text
                processing_steps.append({
                    'title': '✂️ Chunking text',
                    'status': 'processing',
                    'details': 'Breaking text into manageable pieces...'
                })
                
                chunks = chunk_text(text)
                processing_steps[-1]['status'] = 'complete'
                processing_steps[-1]['details'] = f'Created {len(chunks)} text chunks'
                
                # Generate embeddings
                processing_steps.append({
                    'title': '🧠 Generating embeddings',
                    'status': 'processing',
                    'details': 'Converting text to vector embeddings...'
                })
                
                embeddings = get_embeddings(chunks)
                processing_steps[-1]['status'] = 'complete'
                processing_steps[-1]['details'] = f'Generated {len(embeddings)} embeddings'
                
                # Store in legacy collection
                processing_steps.append({
                    'title': '💾 Storing in database',
                    'status': 'processing',
                    'details': 'Adding chunks to vector database...'
                })
                
                # Add to legacy collection
                doc_id = hashlib.md5(filename.encode()).hexdigest()
                ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
                
                metadatas = [
                    {
                        "filename": filename,
                        "chunk_index": i,
                        "upload_time": datetime.now().isoformat(),
                        "source_url": "manual_upload",
                        "document_type": "user_upload"
                    }
                    for i in range(len(chunks))
                ]
                
                legacy_collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
                
                processing_steps[-1]['status'] = 'complete'
                processing_steps[-1]['details'] = f'Successfully stored {len(chunks)} chunks'
                
                return jsonify({
                    'message': f'Successfully uploaded "{filename}" (legacy mode)',
                    'chunks_added': len(chunks),
                    'filename': filename,
                    'processing_steps': processing_steps
                })
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint with multimodal search support"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        auto_search = data.get('auto_search', True)
        
        logger.info(f"Processing chat message: {user_message}")
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'})
        
        if not openai_client.api_key:
            return jsonify({'error': 'OpenAI API key not configured'})
        
        # Search for relevant content
        if multimodal_processor:
            search_results = multimodal_processor.hybrid_search(user_message, n_results=5)
        else:
            # Fallback to legacy search
            search_results = legacy_search(user_message, n_results=5)
        
        used_rag = len(search_results) > 0
        
        # Prepare context for LLM
        processing_info = {
            'found_relevant_docs': used_rag,
            'relevant_docs_count': len(search_results),
            'retrieved_chunks': search_results[:3] if search_results else []
        }
        
        # Build conversation context
        if 'conversation' not in session:
            session['conversation'] = []
        
        if search_results:
            # Create context from search results
            context_parts = []
            for result in search_results:
                element_type = result.get('element_type', 'text')
                content = result['content']
                
                if element_type == 'table':
                    context_parts.append(f"TABLE DATA:\n{content}")
                elif element_type == 'figure':
                    context_parts.append(f"DIAGRAM/FIGURE:\n{content}")
                else:
                    context_parts.append(f"TEXT:\n{content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            system_message = f"""You are a helpful technical assistant specializing in service manuals and repair documentation. Use the following context from documents to answer the user's question. The context may include text, tables with specifications, and descriptions of diagrams/figures.

If the context contains table data, present it clearly. If it references diagrams or figures, explain what they show. If the context doesn't fully answer the question, supplement with your general knowledge but clearly indicate what comes from the documents vs. your general knowledge.

CONTEXT:
{context}

Provide a helpful, detailed response based on the context and your expertise."""
            
            messages = [{"role": "system", "content": system_message}]
            messages.extend(session['conversation'][-6:])  # Keep recent conversation
            messages.append({"role": "user", "content": user_message})
            
        else:
            # No relevant documents found
            messages = session['conversation'][-8:]
            messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = openai_client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        bot_response = response.choices[0].message.content
        
        # Update conversation history
        session['conversation'].append({'role': 'user', 'content': user_message})
        session['conversation'].append({'role': 'assistant', 'content': bot_response})
        
        # Keep conversation history manageable
        if len(session['conversation']) > 12:
            session['conversation'] = session['conversation'][-12:]
        
        return jsonify({
            'response': bot_response,
            'used_rag': used_rag,
            'processing_info': processing_info
        })
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'})

def legacy_search(query: str, n_results: int = 5) -> List[Dict]:
    """Legacy search function for backward compatibility"""
    try:
        if legacy_collection.count() == 0:
            return []
        
        query_embedding = get_embeddings([query])[0]
        
        results = legacy_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, legacy_collection.count()),
            include=['documents', 'distances', 'metadatas']
        )
        
        formatted_results = []
        if results['documents']:
            for doc, distance, metadata in zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            ):
                formatted_results.append({
                    'content': doc,
                    'similarity': 1 - distance,
                    'metadata': metadata,
                    'element_type': 'text'
                })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Legacy search error: {e}")
        return []

@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    session['conversation'] = []
    return jsonify({'message': 'Conversation cleared'})

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error occurred.'}), 500

if __name__ == '__main__':
    # Check configuration
    config_warnings = []
    
    if not openai_client.api_key:
        config_warnings.append("OpenAI API key not configured (OPENAI_API_KEY)")
    
    if not os.getenv('GOOGLE_SEARCH_API_KEY') or not os.getenv('GOOGLE_SEARCH_CX'):
        config_warnings.append("Google Custom Search API not configured (optional)")
    
    if config_warnings:
        print("\n" + "="*70)
        print("⚠️  CONFIGURATION WARNINGS:")
        for warning in config_warnings:
            print(f"   • {warning}")
        print("Please check your .env file for missing configuration.")
        print("="*70 + "\n")
    
    # Display startup information
    print(f"🚀 Starting Multimodal RAG Service Manual Assistant...")
    print(f"📊 Database Statistics:")
    
    if multimodal_processor:
        print(f"   📄 Text elements: {multimodal_processor.text_collection.count()}")
        print(f"   📊 Tables: {multimodal_processor.table_collection.count()}")
        print(f"   🖼️  Figures: {multimodal_processor.figure_collection.count()}")
    else:
        print(f"   ⚠️  Multimodal processor not available")
    
    print(f"   📚 Legacy chunks: {legacy_collection.count() if legacy_collection else 0}")
    print(f"🌐 Server starting at: http://localhost:5000")
    print(f"📁 Downloads saved to: {web_downloads_path}")
    print(f"📝 Logs saved to: logs/app.log")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
EOF

echo -e "${GREEN}✅ Flask application implementation complete${NC}"

# Replace HTML template with enhanced version
echo -e "\n${BLUE}🎨 Implementing enhanced HTML template...${NC}"

cat > templates/chat_template.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal RAG Service Manual Assistant</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔧</text></svg>">
</head>
<body>
    <div class="main-container">
        <div class="chat-container">
            <div class="chat-header">
                <h1>🔧 Multimodal RAG Assistant</h1>
                <p>Advanced service manual analysis with text, tables, and diagrams</p>
                <div class="feature-badges">
                    <span class="badge">📄 Text</span>
                    <span class="badge">📊 Tables</span>
                    <span class="badge">🖼️ Diagrams</span>
                    <span class="badge">🔍 Auto-Search</span>
                </div>
            </div>
            
            <div class="upload-section">
                <div class="file-input-wrapper">
                    <input type="file" id="pdfFile" class="file-input" accept=".pdf" aria-label="Choose PDF file">
                    <label for="pdfFile" class="file-input-label">📄 Choose PDF</label>
                </div>
                <button id="uploadButton" class="upload-button" aria-label="Upload PDF" disabled>Upload</button>
                <div class="file-info" id="fileInfo">No file selected</div>
                <button id="autoSearchButton" class="auto-search-button" aria-label="Toggle auto-search">
                    🔍 Auto-Search: <span id="autoSearchStatus">ON</span>
                </button>
            </div>
            
            <div class="chat-messages" id="chatMessages" role="log" aria-live="polite">
                <div class="message system-message">
                    <div class="system-icon">🎉</div>
                    <div class="system-content">
                        <strong>Welcome to Multimodal RAG Assistant!</strong><br>
                        Upload service manuals with complex tables and diagrams, or ask technical questions. 
                        I can analyze text, extract table data, read diagrams with OCR, and search Google for missing manuals automatically.
                        <div class="example-queries">
                            <div class="example-title">Try asking:</div>
                            <div class="example-chips">
                                <span class="example-chip" onclick="setExampleQuery(this)">Show me the wiring diagram for the compressor</span>
                                <span class="example-chip" onclick="setExampleQuery(this)">What are the torque specifications?</span>
                                <span class="example-chip" onclick="setExampleQuery(this)">Parts breakdown for model XYZ</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator" role="status" aria-live="polite">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span>AI is analyzing multimodal content...</span>
            </div>
            
            <div class="chat-input">
                <input type="text" id="messageInput" aria-label="Chat message" 
                       placeholder="Ask about specifications, diagrams, procedures, or upload a manual..." maxlength="1000">
                <button id="sendButton" aria-label="Send message" disabled>
                    <span class="send-icon">📤</span>
                    Send
                </button>
            </div>
        </div>
        
        <div class="processing-panel">
            <div class="processing-header">
                <div class="header-title">
                    <span class="header-icon">🔄</span>
                    <span>Multimodal Processing</span>
                </div>
                <div class="header-subtitle">Real-time content analysis</div>
            </div>
            
            <div class="processing-content" id="processingContent">
                <div class="stats-section">
                    <div class="stats-title">📊 Content Database</div>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-icon">📄</div>
                            <div class="stat-details">
                                <div class="stat-label">Text Blocks</div>
                                <div class="stat-value" id="textCount">0</div>
                            </div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-icon">📊</div>
                            <div class="stat-details">
                                <div class="stat-label">Tables</div>
                                <div class="stat-value" id="tableCount">0</div>
                            </div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-icon">🖼️</div>
                            <div class="stat-details">
                                <div class="stat-label">Figures</div>
                                <div class="stat-value" id="figureCount">0</div>
                            </div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-icon">📚</div>
                            <div class="stat-details">
                                <div class="stat-label">Total</div>
                                <div class="stat-value" id="totalCount">0</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div id="processingSteps"></div>
                
                <div class="retrieved-chunks" id="retrievedChunks" style="display: none;">
                    <div class="chunks-title">🔍 Retrieved Content</div>
                    <div class="chunks-container"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Loading overlay -->
    <div class="loading-overlay" id="loadingOverlay" style="display: none;">
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <div class="loading-text">Processing multimodal content...</div>
        </div>
    </div>

    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>
EOF

echo -e "${GREEN}✅ HTML template implementation complete${NC}"

# Replace CSS with complete styles
echo -e "\n${BLUE}🎨 Implementing complete CSS styles...${NC}"

cat > static/css/styles.css << 'EOF'
/* Multimodal RAG Assistant - Complete Styles */

/* CSS Variables for theming */
:root {
    --primary-color: #4299e1;
    --primary-dark: #3182ce;
    --secondary-color: #9f7aea;
    --secondary-dark: #805ad5;
    --success-color: #38a169;
    --warning-color: #f6ad55;
    --error-color: #e53e3e;
    --text-color: #2d3748;
    --text-light: #4a5568;
    --text-muted: #718096;
    --bg-light: #f7fafc;
    --bg-white: #ffffff;
    --border-color: #e2e8f0;
    --border-light: #cbd5e0;
    --shadow: 0 10px 30px rgba(0,0,0,0.2);
    --shadow-light: 0 2px 8px rgba(0,0,0,0.1);
    --border-radius: 10px;
    --border-radius-small: 6px;
    --transition: all 0.2s ease;
    --font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Reset and base styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: var(--font-family);
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    line-height: 1.6;
    color: var(--text-color);
}

/* Main layout */
.main-container {
    width: 95%;
    max-width: 1400px;
    height: 90vh;
    display: flex;
    gap: 15px;
    padding: 10px;
}

.chat-container {
    flex: 2;
    background: var(--bg-white);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 600px;
}

.processing-panel {
    flex: 1;
    background: var(--bg-white);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 380px;
    max-width: 500px;
}

/* Chat header */
.chat-header {
    background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
    color: white;
    padding: 25px 30px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.chat-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="rgba(255,255,255,0.1)"/></svg>') repeat;
    opacity: 0.3;
}

.chat-header > * {
    position: relative;
    z-index: 1;
}

.chat-header h1 {
    margin-bottom: 8px;
    font-size: 1.8rem;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.chat-header p {
    margin-bottom: 15px;
    opacity: 0.9;
    font-size: 1rem;
}

.feature-badges {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 4px 10px;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: 500;
    transition: var(--transition);
}

.badge:hover {
    background: rgba(255, 255, 255, 0.25);
    transform: translateY(-1px);
}

/* Upload section */
.upload-section {
    padding: 20px 25px;
    background: var(--bg-light);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}

.file-input-wrapper {
    position: relative;
    display: inline-block;
}

.file-input {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
}

.file-input-label {
    padding: 10px 18px;
    background: var(--primary-color);
    color: white;
    border-radius: var(--border-radius-small);
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: var(--transition);
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: var(--shadow-light);
}

.file-input-label:hover {
    background: var(--primary-dark);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3);
}

.upload-button {
    padding: 10px 18px;
    background: var(--success-color);
    color: white;
    border: none;
    border-radius: var(--border-radius-small);
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: var(--transition);
    box-shadow: var(--shadow-light);
}

.upload-button:hover:not(:disabled) {
    background: #2f855a;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(56, 161, 105, 0.3);
}

.upload-button:disabled {
    background: var(--text-muted);
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}

.auto-search-button {
    padding: 10px 16px;
    background: var(--secondary-color);
    color: white;
    border: none;
    border-radius: var(--border-radius-small);
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    transition: var(--transition);
    margin-left: auto;
    box-shadow: var(--shadow-light);
}

.auto-search-button:hover:not(:disabled) {
    background: var(--secondary-dark);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(159, 122, 234, 0.3);
}

.file-info {
    font-size: 14px;
    color: var(--text-light);
    flex: 1;
    min-width: 200px;
    font-weight: 500;
}

/* Chat messages */
.chat-messages {
    flex: 1;
    padding: 25px;
    overflow-y: auto;
    background: var(--bg-light);
    scroll-behavior: smooth;
}

.chat-messages::-webkit-scrollbar {
    width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
    background: transparent;
}

.chat-messages::-webkit-scrollbar-thumb {
    background: var(--border-light);
    border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

.message {
    margin-bottom: 20px;
    padding: 16px 20px;
    border-radius: 16px;
    max-width: 85%;
    word-wrap: break-word;
    position: relative;
    animation: messageSlideIn 0.3s ease-out;
    box-shadow: var(--shadow-light);
}

@keyframes messageSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.user-message {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    color: white;
    margin-left: auto;
    text-align: right;
    border-bottom-right-radius: 6px;
}

.bot-message {
    background: var(--bg-white);
    border: 1px solid var(--border-color);
    margin-right: auto;
    border-bottom-left-radius: 6px;
}

.system-message {
    background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%);
    border: 1px solid #81e6d9;
    color: #234e52;
    margin: 0 auto;
    max-width: 70%;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    text-align: left;
}

.system-icon {
    font-size: 1.5rem;
    flex-shrink: 0;
}

.system-content {
    flex: 1;
}

.example-queries {
    margin-top: 15px;
    padding-top: 12px;
    border-top: 1px solid rgba(129, 230, 217, 0.3);
}

.example-title {
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 8px;
    color: #1a202c;
}

.example-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.example-chip {
    background: var(--bg-white);
    border: 1px solid #81e6d9;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: var(--transition);
    color: #2d3748;
}

.example-chip:hover {
    background: #81e6d9;
    color: white;
    transform: translateY(-1px);
}

/* Content type indicators */
.content-type-indicator {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 4px;
}

.content-type-text { 
    background: #e6f3ff; 
    color: #0066cc;
    border: 1px solid #b3d9ff;
}

.content-type-table { 
    background: #e6ffe6; 
    color: #008800;
    border: 1px solid #b3ffb3;
}

.content-type-figure { 
    background: #ffe6f3; 
    color: #cc0066;
    border: 1px solid #ffb3d9;
}

.content-type-heading {
    background: #fff5e6;
    color: #cc6600;
    border: 1px solid #ffd9b3;
}

/* Typing indicator */
.typing-indicator {
    display: none;
    margin: 0 25px 20px 25px;
    padding: 16px 20px;
    background: var(--bg-white);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    max-width: 85%;
    font-style: italic;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: var(--shadow-light);
}

.typing-dots {
    display: flex;
    gap: 4px;
}

.typing-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--primary-color);
    animation: typingBounce 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes typingBounce {
    0%, 80%, 100% { 
        transform: scale(0.8);
        opacity: 0.5;
    }
    40% { 
        transform: scale(1);
        opacity: 1;
    }
}

/* Chat input */
.chat-input {
    display: flex;
    padding: 25px;
    background: var(--bg-white);
    border-top: 1px solid var(--border-color);
    gap: 12px;
}

.chat-input input {
    flex: 1;
    padding: 14px 18px;
    border: 2px solid var(--border-light);
    border-radius: 25px;
    outline: none;
    font-size: 15px;
    transition: var(--transition);
    background: var(--bg-light);
}

.chat-input input:focus {
    border-color: var(--primary-color);
    background: var(--bg-white);
    box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.1);
}

.chat-input input::placeholder {
    color: var(--text-muted);
}

.chat-input button {
    padding: 14px 24px;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    color: white;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    font-size: 15px;
    font-weight: 600;
    transition: var(--transition);
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: var(--shadow-light);
}

.chat-input button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(66, 153, 225, 0.4);
}

.chat-input button:disabled {
    background: var(--text-muted);
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}

.send-icon {
    font-size: 1rem;
}

/* Processing panel */
.processing-header {
    background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
    color: white;
    padding: 20px;
    position: relative;
    overflow: hidden;
}

.processing-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="1" fill="rgba(255,255,255,0.1)"/></svg>') repeat;
    opacity: 0.3;
}

.processing-header > * {
    position: relative;
    z-index: 1;
}

.header-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.header-icon {
    font-size: 1.2rem;
    animation: spin 2s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.header-subtitle {
    font-size: 0.9rem;
    opacity: 0.8;
}

.processing-content {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    background: var(--bg-light);
    font-size: 13px;
}

.processing-content::-webkit-scrollbar {
    width: 6px;
}

.processing-content::-webkit-scrollbar-track {
    background: transparent;
}

.processing-content::-webkit-scrollbar-thumb {
    background: var(--border-light);
    border-radius: 3px;
}

/* Stats section */
.stats-section {
    background: var(--bg-white);
    border: 1px solid var(--border-color);
    padding: 16px;
    margin-bottom: 20px;
    border-radius: var(--border-radius-small);
    box-shadow: var(--shadow-light);
}

.stats-title {
    font-weight: 700;
    color: var(--text-color);
    margin-bottom: 15px;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

.stat-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px;
    background: var(--bg-light);
    border-radius: var(--border-radius-small);
    transition: var(--transition);
}

.stat-item:hover {
    background: #edf2f7;
    transform: translateY(-1px);
}

.stat-icon {
    font-size: 1.2rem;
    flex-shrink: 0;
}

.stat-details {
    flex: 1;
}

.stat-label {
    font-size: 11px;
    color: var(--text-muted);
    font-weight: 500;
}

.stat-value {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-color);
}

/* Processing steps */
.process-step {
    margin-bottom: 15px;
    padding: 12px 16px;
    border-radius: var(--border-radius-small);
    border-left: 4px solid var(--border-color);
    background: var(--bg-white);
    transition: var(--transition);
    box-shadow: var(--shadow-light);
}

.step-pending {
    border-left-color: var(--border-light);
    color: var(--text-muted);
}

.step-processing {
    border-left-color: var(--primary-color);
    background: #ebf8ff;
    color: #2b6cb0;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.8; }
    100% { opacity: 1; }
}

.step-complete {
    border-left-color: var(--success-color);
    background: #f0fff4;
    color: #22543d;
}

.step-error {
    border-left-color: var(--error-color);
    background: #fed7d7;
    color: #c53030;
}

.step-title {
    font-weight: 600;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.step-details {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
    line-height: 1.4;
}

.processing-spinner {
    display: inline-block;
    width: 12px;
    height: 12px;
    border: 2px solid var(--border-color);
    border-radius: 50%;
    border-top-color: var(--primary-color);
    animation: spin 1s ease-in-out infinite;
    margin-right: 6px;
}

/* Retrieved chunks */
.retrieved-chunks {
    margin-top: 20px;
    background: var(--bg-white);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-small);
    overflow: hidden;
    box-shadow: var(--shadow-light);
}

.chunks-title {
    background: var(--bg-light);
    padding: 12px 16px;
    font-weight: 600;
    color: var(--text-color);
    border-bottom: 1px solid var(--border-color);
    font-size: 14px;
}

.chunks-container {
    max-height: 300px;
    overflow-y: auto;
}

.chunk {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    transition: var(--transition);
}

.chunk:last-child {
    border-bottom: none;
}

.chunk:hover {
    background: var(--bg-light);
}

.chunk-header {
    font-weight: 600;
    color: var(--text-light);
    margin-bottom: 6px;
    font-size: 11px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chunk-content {
    color: var(--text-color);
    line-height: 1.4;
    font-size: 12px;
}

/* Loading overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(4px);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.loading-content {
    background: var(--bg-white);
    padding: 40px;
    border-radius: var(--border-radius);
    text-align: center;
    box-shadow: var(--shadow);
}

.loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid var(--border-color);
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px;
}

.loading-text {
    color: var(--text-color);
    font-weight: 600;
}

/* Error and success states */
.error-message {
    background: #fed7d7;
    border: 1px solid #feb2b2;
    color: #c53030;
}

.success-message {
    background: #c6f6d5;
    border: 1px solid #9ae6b4;
    color: #22543d;
}

/* Responsive design */
@media (max-width: 1200px) {
    .main-container {
        flex-direction: column;
        height: auto;
        max-height: 95vh;
        gap: 10px;
    }
    
    .chat-container {
        flex: none;
        height: 65vh;
        min-width: auto;
    }
    
    .processing-panel {
        flex: none;
        height: 25vh;
        min-width: auto;
        max-width: none;
    }
    
    .stats-grid {
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
    }
    
    .stat-item {
        flex-direction: column;
        text-align: center;
        gap: 4px;
        padding: 8px;
    }
    
    .stat-details {
        text-align: center;
    }
}

@media (max-width: 768px) {
    .main-container {
        width: 98%;
        padding: 5px;
        height: 95vh;
    }
    
    .upload-section {
        flex-direction: column;
        align-items: stretch;
        gap: 8px;
    }
    
    .auto-search-button {
        margin-left: 0;
    }
    
    .chat-header {
        padding: 20px;
    }
    
    .chat-header h1 {
        font-size: 1.5rem;
    }
    
    .feature-badges {
        gap: 4px;
    }
    
    .badge {
        font-size: 0.7rem;
        padding: 3px 8px;
    }
    
    .message {
        max-width: 95%;
        padding: 12px 16px;
    }
    
    .system-message {
        max-width: 90%;
    }
    
    .stats-grid {
        grid-template-columns: 1fr 1fr;
    }
    
    .chat-input {
        padding: 15px;
        flex-direction: column;
        gap: 8px;
    }
    
    .chat-input input {
        border-radius: 20px;
    }
    
    .chat-input button {
        border-radius: 20px;
        align-self: stretch;
        justify-content: center;
    }
}

/* Print styles */
@media print {
    .main-container {
        height: auto;
        background: white;
        flex-direction: column;
    }
    
    .processing-panel {
        display: none;
    }
    
    .chat-header {
        background: white;
        color: black;
        border-bottom: 2px solid black;
    }
    
    .upload-section,
    .chat-input {
        display: none;
    }
    
    .message {
        break-inside: avoid;
        margin-bottom: 10px;
    }
}

/* Focus styles for accessibility */
.file-input-label:focus,
.upload-button:focus,
.auto-search-button:focus,
.chat-input button:focus,
.example-chip:focus {
    outline: 2px solid var(--primary-color);
    outline-offset: 2px;
}

.chat-input input:focus {
    outline: none; /* Custom focus style already applied */
}

/* High contrast mode support */
@media (prefers-contrast: high) {
    :root {
        --border-color: #000000;
        --border-light: #333333;
        --text-muted: #000000;
    }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}
EOF

echo -e "${GREEN}✅ CSS implementation complete${NC}"

# Replace JavaScript with complete implementation
echo -e "\n${BLUE}🎯 Implementing complete JavaScript functionality...${NC}"

cat > static/js/app.js << 'EOF'
// Multimodal RAG Assistant - Complete Client-side JavaScript

// Global state
let autoSearchEnabled = true;
let currentStepId = 0;
let isProcessing = false;

// DOM elements - initialized after DOMContentLoaded
let elements = {};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Multimodal RAG Assistant initialized');
    
    // Get DOM element references
    elements = {
        chatMessages: document.getElementById('chatMessages'),
        messageInput: document.getElementById('messageInput'),
        sendButton: document.getElementById('sendButton'),
        typingIndicator: document.getElementById('typingIndicator'),
        pdfFile: document.getElementById('pdfFile'),
        uploadButton: document.getElementById('uploadButton'),
        fileInfo: document.getElementById('fileInfo'),
        processingContent: document.getElementById('processingContent'),
        processingSteps: document.getElementById('processingSteps'),
        retrievedChunks: document.getElementById('retrievedChunks'),
        textCount: document.getElementById('textCount'),
        tableCount: document.getElementById('tableCount'),
        figureCount: document.getElementById('figureCount'),
        totalCount: document.getElementById('totalCount'),
        autoSearchButton: document.getElementById('autoSearchButton'),
        autoSearchStatus: document.getElementById('autoSearchStatus'),
        loadingOverlay: document.getElementById('loadingOverlay')
    };
    
    // Initialize the app
    updateStats();
    setupEventListeners();
    elements.messageInput.focus();
    
    // Enable send button when input has content
    elements.messageInput.addEventListener('input', function() {
        elements.sendButton.disabled = !this.value.trim() || isProcessing;
    });
});

// Event listeners setup
function setupEventListeners() {
    // Send message on button click
    elements.sendButton.addEventListener('click', sendMessage);
    
    // Upload PDF on button click
    elements.uploadButton.addEventListener('click', uploadPDF);
    
    // Toggle auto-search
    elements.autoSearchButton.addEventListener('click', toggleAutoSearch);
    
    // Send message on Enter key
    elements.messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey && !isProcessing) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // File selection handling
    elements.pdfFile.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const sizeMB = (file.size / 1024 / 1024).toFixed(2);
            elements.fileInfo.textContent = `Selected: ${file.name} (${sizeMB} MB)`;
            elements.uploadButton.disabled = false;
            
            // Validate file size (50MB limit)
            if (file.size > 50 * 1024 * 1024) {
                elements.fileInfo.textContent = `⚠️ File too large: ${file.name} (${sizeMB} MB) - Max 50MB`;
                elements.fileInfo.style.color = '#e53e3e';
                elements.uploadButton.disabled = true;
            } else {
                elements.fileInfo.style.color = '#4a5568';
            }
        } else {
            elements.fileInfo.textContent = 'No file selected';
            elements.fileInfo.style.color = '#4a5568';
            elements.uploadButton.disabled = true;
        }
    });
    
    // Drag and drop for PDF files
    const chatContainer = elements.chatMessages.parentElement;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        chatContainer.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        chatContainer.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        chatContainer.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight(e) {
        chatContainer.style.background = '#f0f8ff';
        chatContainer.style.border = '2px dashed #4299e1';
    }
    
    function unhighlight(e) {
        chatContainer.style.background = '';
        chatContainer.style.border = '';
    }
    
    chatContainer.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            const file = files[0];
            if (file.type === 'application/pdf') {
                elements.pdfFile.files = files;
                elements.pdfFile.dispatchEvent(new Event('change', { bubbles: true }));
            } else {
                addMessage('❌ Please drop a PDF file only.', 'system', true);
            }
        }
    }
}

// Update statistics display
function updateStats() {
    fetch('/stats')
        .then(response => response.json())
        .then(data => {
            elements.textCount.textContent = data.text_chunks || 0;
            elements.tableCount.textContent = data.tables || 0;
            elements.figureCount.textContent = data.figures || 0;
            elements.totalCount.textContent = data.total_chunks || 0;
            
            // Add animation to updated values
            [elements.textCount, elements.tableCount, elements.figureCount, elements.totalCount].forEach(el => {
                el.style.transform = 'scale(1.1)';
                setTimeout(() => {
                    el.style.transform = 'scale(1)';
                }, 200);
            });
        })
        .catch(error => {
            console.log('Could not load stats:', error);
        });
}

// Toggle auto-search functionality
function toggleAutoSearch() {
    autoSearchEnabled = !autoSearchEnabled;
    elements.autoSearchStatus.textContent = autoSearchEnabled ? 'ON' : 'OFF';
    elements.autoSearchButton.style.background = autoSearchEnabled ? '#9f7aea' : '#a0aec0';
    
    const message = `Auto service manual search ${autoSearchEnabled ? 'enabled' : 'disabled'}. ` +
                   `${autoSearchEnabled ? 'I\'ll automatically search and download service manuals when relevant.' : 'I won\'t search for manuals automatically.'}`;
    
    addMessage(message, 'system');
}

// Processing steps management
function addProcessingStep(title, status = 'pending', details = '') {
    const stepId = `step-${currentStepId++}`;
    const stepDiv = document.createElement('div');
    stepDiv.className = `process-step step-${status}`;
    stepDiv.id = stepId;
    
    stepDiv.innerHTML = `
        <div class="step-title">
            ${status === 'processing' ? '<span class="processing-spinner"></span>' : ''}
            ${title}
        </div>
        ${details ? `<div class="step-details">${details}</div>` : ''}
    `;
    
    elements.processingSteps.appendChild(stepDiv);
    elements.processingContent.scrollTop = elements.processingContent.scrollHeight;
    
    return stepId;
}

function updateProcessingStep(stepId, status, details = '') {
    const stepDiv = document.getElementById(stepId);
    if (!stepDiv) return;
    
    stepDiv.className = `process-step step-${status}`;
    
    const titleElement = stepDiv.querySelector('.step-title');
    const titleText = titleElement.textContent.replace(/^\s*/, '');
    titleElement.innerHTML = `
        ${status === 'processing' ? '<span class="processing-spinner"></span>' : ''}
        ${titleText}
    `;
    
    if (details) {
        let detailsDiv = stepDiv.querySelector('.step-details');
        if (!detailsDiv) {
            detailsDiv = document.createElement('div');
            detailsDiv.className = 'step-details';
            stepDiv.appendChild(detailsDiv);
        }
        detailsDiv.textContent = details;
    }
    
    elements.processingContent.scrollTop = elements.processingContent.scrollHeight;
}

function clearProcessingSteps() {
    elements.processingSteps.innerHTML = '';
    elements.retrievedChunks.style.display = 'none';
    currentStepId = 0;
}

// Enhanced display for multimodal retrieved content
function showEnhancedRetrievedContent(searchResults) {
    elements.retrievedChunks.style.display = 'block';
    
    const container = elements.retrievedChunks.querySelector('.chunks-container') || 
                     (() => {
                         const div = document.createElement('div');
                         div.className = 'chunks-container';
                         elements.retrievedChunks.appendChild(div);
                         return div;
                     })();
    
    // Clear previous content
    container.innerHTML = '';
    
    if (!searchResults || searchResults.length === 0) {
        container.innerHTML = '<div class="chunk"><div class="chunk-content">No relevant content found</div></div>';
        return;
    }
    
    // Add content items
    searchResults.forEach((result, index) => {
        const chunkDiv = document.createElement('div');
        chunkDiv.className = 'chunk';
        
        const contentType = result.metadata?.element_type || 'text';
        const typeLabel = getContentTypeLabel(contentType);
        const similarity = (result.similarity || 0).toFixed(3);
        const pageNum = result.metadata?.page_number || 'Unknown';
        
        chunkDiv.innerHTML = `
            <div class="chunk-header">
                <span class="content-type-indicator content-type-${contentType}">
                    ${typeLabel}
                </span>
                <span>Page ${pageNum} • Similarity: ${similarity}</span>
            </div>
            <div class="chunk-content">
                ${formatContentPreview(result, contentType)}
            </div>
        `;
        
        container.appendChild(chunkDiv);
    });
    
    elements.processingContent.scrollTop = elements.processingContent.scrollHeight;
}

function getContentTypeLabel(type) {
    const labels = {
        'text': '📄 Text',
        'heading': '📋 Heading',
        'table': '📊 Table',
        'figure': '🖼️ Figure',
        'diagram': '📐 Diagram'
    };
    return labels[type] || '📄 Text';
}

function formatContentPreview(result, contentType) {
    let preview = result.content || '';
    
    // Truncate very long content
    if (preview.length > 300) {
        preview = preview.substring(0, 300) + '...';
    }
    
    // Add special formatting for different content types
    if (contentType === 'table') {
        // Try to show table in a more structured way
        const tableHtml = result.metadata?.table_html;
        if (tableHtml && tableHtml.length < 500) {
            return `<div class="table-preview">${tableHtml}</div>`;
        }
    }
    
    if (contentType === 'figure') {
        const ocr = result.metadata?.ocr_text || '';
        const description = result.metadata?.description || '';
        
        let figureInfo = '';
        if (result.metadata?.image_width && result.metadata?.image_height) {
            figureInfo = `<div class="figure-info">
                📏 ${result.metadata.image_width}×${result.metadata.image_height}px
                ${ocr ? '• 📝 Contains text labels' : '• 🖼️ Visual content'}
            </div>`;
        }
        
        return `${preview}${figureInfo}`;
    }
    
    return preview;
}

// Message handling
function addMessage(text, sender, isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    if (isError) {
        messageDiv.className += ' error-message';
    } else if (sender === 'system' && !isError) {
        messageDiv.className = 'message system-message success-message';
    }
    
    // Handle HTML content for system messages
    if (sender === 'system' && text.includes('<')) {
        messageDiv.innerHTML = text;
    } else {
        messageDiv.textContent = text;
    }
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    
    // Auto-scroll with smooth animation
    messageDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Set example query from welcome message
function setExampleQuery(element) {
    elements.messageInput.value = element.textContent;
    elements.messageInput.focus();
    elements.sendButton.disabled = false;
}

// Make it globally available
window.setExampleQuery = setExampleQuery;

// Upload PDF functionality
async function uploadPDF() {
    const file = elements.pdfFile.files[0];
    if (!file) return;

    // Disable upload controls
    elements.uploadButton.disabled = true;
    elements.uploadButton.textContent = 'Uploading...';
    isProcessing = true;
    
    clearProcessingSteps();
    
    // Show loading overlay
    elements.loadingOverlay.style.display = 'flex';

    const formData = new FormData();
    formData.append('pdf', file);

    const step1 = addProcessingStep('📄 Uploading PDF file', 'processing', 
                                   `File: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        // Hide loading overlay
        elements.loadingOverlay.style.display = 'none';
        
        if (data.error) {
            updateProcessingStep(step1, 'error', `Upload failed: ${data.error}`);
            addMessage(`❌ Upload failed: ${data.error}`, 'system', true);
        } else {
            updateProcessingStep(step1, 'complete', `Successfully uploaded ${file.name}`);
            
            // Add processing steps from server
            if (data.processing_steps) {
                data.processing_steps.forEach(step => {
                    addProcessingStep(step.title, step.status, step.details);
                });
            }
            
            // Show success message with content breakdown
            let successMsg = `✅ Successfully uploaded "${file.name}"`;
            if (data.content_breakdown) {
                const breakdown = data.content_breakdown;
                successMsg += `\n📊 Content processed: ${breakdown.text || 0} text blocks, ${breakdown.tables || 0} tables, ${breakdown.figures || 0} figures`;
            } else {
                successMsg += ` - ${data.chunks_added} chunks added to knowledge base`;
            }
            
            addMessage(successMsg, 'system');
            
            // Clear file selection
            elements.pdfFile.value = '';
            elements.fileInfo.textContent = 'No file selected';
            elements.fileInfo.style.color = '#4a5568';
            
            // Update stats
            updateStats();
        }
    } catch (error) {
        elements.loadingOverlay.style.display = 'none';
        updateProcessingStep(step1, 'error', 'Network error during upload');
        addMessage('❌ Upload failed: Network error', 'system', true);
        console.error('Upload error:', error);
    } finally {
        // Re-enable upload controls
        elements.uploadButton.disabled = false;
        elements.uploadButton.textContent = 'Upload';
        isProcessing = false;
    }
}

// Send message functionality
async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message || isProcessing) return;

    // Disable input controls
    elements.messageInput.disabled = true;
    elements.sendButton.disabled = true;
    isProcessing = true;
    
    // Add user message
    addMessage(message, 'user');
    elements.messageInput.value = '';
    
    clearProcessingSteps();

    // Show typing indicator
    elements.typingIndicator.style.display = 'flex';
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;

    const step1 = addProcessingStep('🔍 Searching vector database', 'processing', 
                                   'Looking for relevant content across text, tables, and diagrams...');

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message: message,
                auto_search: autoSearchEnabled
            })
        });

        const data = await response.json();
        
        // Hide typing indicator
        elements.typingIndicator.style.display = 'none';

        if (data.error) {
            updateProcessingStep(step1, 'error', `Query failed: ${data.error}`);
            addMessage(`❌ Error: ${data.error}`, 'bot', true);
        } else {
            // Update processing steps based on response
            if (data.processing_info) {
                const info = data.processing_info;
                
                if (info.found_relevant_docs) {
                    updateProcessingStep(step1, 'complete', 
                                       `Found ${info.relevant_docs_count} relevant content pieces`);
                    
                    if (info.retrieved_chunks && info.retrieved_chunks.length > 0) {
                        showEnhancedRetrievedContent(info.retrieved_chunks);
                    }
                    
                    const step2 = addProcessingStep('🧠 Generating response with context', 'complete', 
                                                   'Using retrieved documents + general knowledge');
                } else {
                    updateProcessingStep(step1, 'complete', 
                                       'No relevant documents found - using general knowledge');
                    const step2 = addProcessingStep('🧠 Generating response', 'complete', 
                                                   'Using general knowledge only');
                }
            }
            
            // Show context indicator if RAG was used
            if (data.used_rag) {
                const contextDiv = document.createElement('div');
                contextDiv.className = 'context-indicator';
                contextDiv.innerHTML = '📄 <em>Answer based on uploaded documents</em>';
                contextDiv.style.fontSize = '12px';
                contextDiv.style.color = '#718096';
                contextDiv.style.fontStyle = 'italic';
                contextDiv.style.marginBottom = '10px';
                contextDiv.style.textAlign = 'center';
                elements.chatMessages.appendChild(contextDiv);
            }
            
            // Add bot response
            addMessage(data.response, 'bot');
        }
    } catch (error) {
        elements.typingIndicator.style.display = 'none';
        updateProcessingStep(step1, 'error', 'Network error during chat request');
        addMessage('❌ Sorry, there was an error connecting to the server.', 'bot', true);
        console.error('Chat error:', error);
    } finally {
        // Re-enable input controls
        elements.messageInput.disabled = false;
        elements.sendButton.disabled = false;
        isProcessing = false;
        elements.messageInput.focus();
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Health check
function checkHealth() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            console.log('Health check:', data);
        })
        .catch(error => {
            console.error('Health check failed:', error);
        });
}

// Initialize health checking
setInterval(checkHealth, 300000); // Every 5 minutes

// Error handling for unhandled promise rejections
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    addMessage('❌ An unexpected error occurred. Please try again.', 'system', true);
});

// Export functions for debugging (development only)
if (window.location.hostname === 'localhost') {
    window.RAGDebug = {
        updateStats,
        clearProcessingSteps,
        addMessage,
        sendMessage,
        uploadPDF,
        toggleAutoSearch
    };
}

console.log('🎨 Multimodal RAG Assistant UI fully loaded');
EOF

echo -e "${GREEN}✅ JavaScript implementation complete${NC}"

# Create additional utility files
echo -e "\n${BLUE}📄 Creating additional utility files...${NC}"

# Create requirements.txt for reference
cat > requirements.txt << 'EOF'
# Multimodal RAG Assistant Dependencies
# Generated from pyproject.toml for reference

flask>=3.0.0
openai>=1.50.0
chromadb>=0.4.22
PyMuPDF>=1.23.8
pdfplumber>=0.10.3
pytesseract>=0.3.10
Pillow>=10.1.0
pandas>=2.1.4
camelot-py[cv]>=0.10.1
requests>=2.31.0
python-dotenv>=1.0.0
sentence-transformers>=2.2.2
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
opencv-python>=4.8.0
easyocr>=1.7.0
tabulate>=0.9.0
tqdm>=4.66.0
python-multipart>=0.0.6

# Development dependencies
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.4.0
EOF

# Create configuration validation script
cat > validate_config.py << 'EOF'
#!/usr/bin/env python3
"""
Configuration validation script for Multimodal RAG Assistant
"""

import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Check environment configuration"""
    load_dotenv()
    
    print("🔍 Validating Configuration...")
    print("=" * 50)
    
    issues = []
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        issues.append("❌ OPENAI_API_KEY not set")
    elif openai_key.startswith('sk-'):
        print("✅ OpenAI API key configured")
    else:
        issues.append("⚠️  OpenAI API key format looks incorrect")
    
    # Check Google Search API (optional)
    google_key = os.getenv('GOOGLE_SEARCH_API_KEY')
    google_cx = os.getenv('GOOGLE_SEARCH_CX')
    
    if google_key and google_cx:
        print("✅ Google Custom Search API configured")
    else:
        print("⚠️  Google Custom Search API not configured (optional)")
    
    # Check Flask secret key
    flask_secret = os.getenv('FLASK_SECRET_KEY')
    if not flask_secret or flask_secret == 'your-secure-secret-key-change-this':
        issues.append("⚠️  Flask secret key should be changed for production")
    else:
        print("✅ Flask secret key configured")
    
    # Check system dependencies
    try:
        import tesseract
        print("✅ Tesseract OCR available")
    except ImportError:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            print("✅ Tesseract OCR available via pytesseract")
        except:
            issues.append("❌ Tesseract OCR not available")
    
    # Check Python dependencies
    required_packages = [
        'flask', 'openai', 'chromadb', 'fitz', 'pdfplumber', 
        'PIL', 'pandas', 'numpy', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        issues.append(f"❌ Missing packages: {', '.join(missing_packages)}")
    else:
        print("✅ All required Python packages available")
    
    print("\n" + "=" * 50)
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease resolve these issues before running the application.")
        return False
    else:
        print("🎉 Configuration looks good!")
        return True

if __name__ == '__main__':
    success = check_environment()
    sys.exit(0 if success else 1)
EOF

chmod +x validate_config.py

# Update the README with complete implementation details
cat > README.md << 'EOF'
# Multimodal RAG Service Manual Assistant

A sophisticated Retrieval-Augmented Generation (RAG) system specifically engineered for service manuals, featuring advanced support for text analysis, table extraction, diagram interpretation, and automated content discovery.

## 🌟 Features

### **Advanced Content Processing**
- 📄 **Smart Text Extraction**: Structure-aware PDF parsing with heading detection
- 📊 **Table Intelligence**: Automatic table detection, extraction, and semantic understanding
- 🖼️ **Visual Analysis**: OCR processing and AI-powered diagram interpretation
- 🧠 **Multimodal Embeddings**: Separate vector stores for different content types

### **Intelligent Search & Retrieval**
- 🎯 **Query Routing**: Automatic routing to appropriate content types based on query intent
- 🔍 **Hybrid Search**: Combines semantic similarity with content-type awareness
- 📈 **Relevance Scoring**: Advanced scoring considering content type and semantic similarity
- 🔄 **Context Assembly**: Smart context building from multiple content sources

### **Automated Content Discovery**
- 🌐 **Google Integration**: Custom Search API integration for automatic manual discovery
- 🚗 **Vehicle-Specific Matching**: Intelligent matching for specific vehicle models and years
- ⬇️ **Automatic Download**: PDF download and processing pipeline
- 📚 **Duplicate Detection**: Prevents redundant manual processing

## 🚀 Quick Start

### **1. System Requirements**

**Operating System**: Linux, macOS, or Windows with WSL
**Python**: 3.11 or higher
**System Dependencies**: tesseract-ocr, poppler-utils

### **2. Installation**

```bash
# Clone or download the project
git clone <repository-url>
cd multimodal-rag-assistant

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
nano .env  # Add your API keys
```

### **3. Configuration**

Edit `.env` file with your API keys:

```bash
# Required
OPENAI_API_KEY=your-openai-api-key-here

# Optional (for auto-search)
GOOGLE_SEARCH_API_KEY=your-google-api-key
GOOGLE_SEARCH_CX=your-search-engine-id

# Flask configuration
FLASK_SECRET_KEY=your-secure-secret-key
```

### **4. Validation & Launch**

```bash
# Validate configuration
uv run python validate_config.py

# Start the application
./run.sh

# Or run directly
uv run python app.py
```

### **5. Access**

Open your browser to `http://localhost:5000`

## 📚 Usage Guide

### **PDF Upload & Processing**

1. **Select PDF**: Click "Choose PDF" or drag & drop
2. **Upload**: System automatically:
   - Extracts text with structure preservation
   - Detects and processes tables
   - Performs OCR on diagrams/images
   - Generates embeddings for all content types
   - Stores in appropriate vector collections

### **Query Types & Examples**

**Text-based Queries**:
```
"How do I replace the alternator?"
"What is the maintenance schedule?"
"Safety precautions for electrical work"
```

**Table/Specification Queries**:
```
"What are the torque specifications?"
"Show me the bolt sizes table"
"Electrical specifications for 2018 model"
```

**Diagram/Visual Queries**:
```
"Show me the wiring diagram for the compressor"
"Parts breakdown for the transmission"
"Exploded view of the engine assembly"
```

**Vehicle-Specific Queries**:
```
"2018 Honda Civic alternator replacement"
"1994 Ford Bronco speaker wiring diagram"
"2006 Chevy 1500 torque specifications"
```

### **Auto-Search Functionality**

When enabled, the system automatically:
1. Detects queries requiring service manuals
2. Searches Google Custom Search for relevant PDFs
3. Downloads and processes matching manuals
4. Provides intelligent vehicle-specific matching

## 🏗️ Architecture

### **Directory Structure**

```
multimodal-rag-assistant/
├── app.py                      # Main Flask application
├── templates/
│   └── chat_template.html      # Enhanced UI template
├── static/
│   ├── css/styles.css          # Complete styling
│   └── js/app.js              # Full client-side functionality
├── chroma_db/                  # Vector database storage
│   ├── service_manual_text/    # Text content collection
│   ├── service_manual_tables/  # Table data collection
│   └── service_manual_figures/ # Figure/diagram collection
├── web_downloads/              # Downloaded service manuals
├── logs/                       # Application logs
├── tests/                      # Test suite
├── backups/                    # Configuration backups
├── .env                        # Environment configuration
├── pyproject.toml             # Dependency management
├── requirements.txt           # Dependencies reference
├── validate_config.py         # Configuration validator
├── run.sh                     # Quick start script
└── dev.sh                     # Development helper
```

### **Processing Pipeline**

```
PDF Input → Structure Analysis → Content Extraction
    ↓
┌─ Text Blocks ─┐  ┌─ Tables ─┐  ┌─ Figures ─┐
│   • Headings  │  │ • Extract │  │ • OCR     │
│   • Paragraphs│  │ • Structure│  │ • Vision  │
│   • Metadata  │  │ • Summarize│  │ • Describe│
└───────────────┘  └───────────┘  └───────────┘
    ↓                  ↓              ↓
┌─ Text Embeddings ──┐ ┌─ Table Embeddings ─┐ ┌─ Figure Embeddings ─┐
│ OpenAI Ada-002     │ │ Structured +        │ │ OCR + Description   │
│ text-embedding     │ │ Summary Embeddings  │ │ Combined Embeddings │
└────────────────────┘ └────────────────────┘ └─────────────────────┘
    ↓                  ↓              ↓
┌─ Vector Storage (ChromaDB Collections) ─┐
│ • service_manual_text                    │
│ • service_manual_tables                 │
│ • service_manual_figures                │
└──────────────────────────────────────────┘
```

### **Search & Retrieval Flow**

```
User Query → Query Analysis → Content Type Detection
    ↓
┌─ Route to Collections ─┐
│ • Text keywords        │
│ • Table indicators     │
│ • Visual terms         │
└────────────────────────┘
    ↓
┌─ Parallel Search ─┐
│ • Text Collection │
│ • Table Collection│
│ • Figure Collection│
└───────────────────┘
    ↓
┌─ Result Fusion ─┐
│ • Similarity    │
│ • Content Type  │
│ • Page Context  │
└─────────────────┘
    ↓
┌─ Context Assembly ─┐
│ • Format by type   │
│ • Add metadata     │
│ • Provide to LLM   │
└───────────────────┘
```

## 🔧 Development

### **Development Commands**

```bash
# Install development dependencies
./dev.sh install

# Run tests
./dev.sh test

# Code formatting
./dev.sh format

# Linting
./dev.sh lint

# Clean up
./dev.sh clean
```

### **Testing**

```bash
# Run full test suite
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=app --cov-report=html

# Test specific functionality
uv run pytest tests/test_multimodal.py -v
```

### **Configuration Options**

Environment variables in `.env`:

```bash
# Core Configuration
OPENAI_API_KEY=sk-...           # Required
OPENAI_MODEL=gpt-3.5-turbo      # LLM model
OPENAI_VISION_MODEL=gpt-4-vision-preview  # Vision model

# Google Search (Optional)
GOOGLE_SEARCH_API_KEY=...       # Custom Search API key
GOOGLE_SEARCH_CX=...           # Search engine ID

# Processing Limits
MAX_FILE_SIZE_MB=50            # Upload limit
MAX_PAGES_PROCESS=100          # Pages per document
OCR_ENABLED=true               # Enable OCR processing
VISION_API_ENABLED=true        # Enable vision descriptions
TABLE_EXTRACTION_ENABLED=true  # Enable table processing

# OCR Configuration
TESSERACT_CMD=/usr/bin/tesseract
TESSERACT_CONFIG=--oem 3 --psm 6

# Storage
CHROMA_DB_PATH=./chroma_db
WEB_DOWNLOADS_PATH=./web_downloads

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## 🐛 Troubleshooting

### **Common Issues**

**1. Tesseract OCR Not Found**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/tesseract-ocr/tesseract
```

**2. PDF Processing Errors**
```bash
# Install poppler utilities
sudo apt-get install poppler-utils  # Linux
brew install poppler                # macOS
```

**3. ChromaDB Permission Issues**
```bash
# Fix directory permissions
sudo chown -R $USER:$USER chroma_db/
chmod -R 755 chroma_db/
```

**4. Memory Issues with Large PDFs**
- Reduce `MAX_PAGES_PROCESS` in `.env`
- Disable vision API for large files
- Process files in smaller batches

### **Performance Optimization**

**For Large Documents**:
- Set `MAX_PAGES_PROCESS=50` for faster processing
- Disable vision API: `VISION_API_ENABLED=false`
- Use `gpt-3.5-turbo` instead of `gpt-4` for table summaries

**For Production**:
- Set `LOG_LEVEL=WARNING`
- Enable caching for embeddings
- Use load balancer for multiple instances

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`./dev.sh test`)
5. Format code (`./dev.sh format`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include configuration and log files (remove sensitive data)

---

**Built with ❤️ for the service manual community**
EOF

echo -e "${GREEN}✅ All implementation files created successfully!${NC}"

# Update run script with validation
cat > run.sh << 'EOF'
#!/bin/bash

# Multimodal RAG Assistant - Enhanced Run Script

echo "🚀 Starting Multimodal RAG Service Manual Assistant..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env with your API keys:"
    echo "   nano .env"
    echo ""
    echo "Required: OPENAI_API_KEY"
    echo "Optional: GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_CX"
    exit 1
fi

# Run configuration validation
echo "🔍 Validating configuration..."
if ! uv run python validate_config.py; then
    echo ""
    echo "❌ Configuration validation failed."
    echo "Please fix the issues above before starting the application."
    exit 1
fi

echo ""
echo "✅ Configuration validated successfully!"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Virtual environment not found. Creating..."
    uv sync
fi

# Create necessary directories
mkdir -p logs chroma_db web_downloads

# Start the application
echo ""
echo "🌐 Starting Flask application..."
echo "   Local:    http://localhost:5000"
echo "   Network:  http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "📊 Features enabled:"
echo "   • Multimodal content processing"
echo "   • OCR for diagrams and images"
echo "   • Table extraction and analysis"
echo "   • Intelligent query routing"
echo "   • Auto-search for service manuals"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uv run python app.py
EOF

echo -e "\n${BLUE}🎯 Creating final deployment checks...${NC}"

# Create simple deployment test
cat > test_deployment.py << 'EOF'
#!/usr/bin/env python3
"""
Simple deployment test for Multimodal RAG Assistant
"""

import requests
import time
import sys

def test_deployment():
    """Test basic deployment functionality"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing deployment...")
    
    try:
        # Test health endpoint
        print("1. Testing health check...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("   ✅ Health check passed")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test main page
        print("2. Testing main page...")
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200 and "Multimodal RAG" in response.text:
            print("   ✅ Main page loaded")
        else:
            print(f"   ❌ Main page failed: {response.status_code}")
            return False
        
        # Test stats endpoint
        print("3. Testing stats endpoint...")
        response = requests.get(f"{base_url}/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Stats: {data.get('total_chunks', 0)} total chunks")
        else:
            print(f"   ❌ Stats failed: {response.status_code}")
            return False
        
        print("\n🎉 All tests passed! Deployment looks good.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the application.")
        print("   Make sure the server is running: ./run.sh")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == '__main__':
    success = test_deployment()
    sys.exit(0 if success else 1)
EOF

chmod +x test_deployment.py

echo -e "\n${GREEN}🎉 Complete implementation finished!${NC}\n"

# Final status and instructions
echo -e "${BLUE}📋 Implementation Summary:${NC}"
echo -e "${GREEN}✅ Complete Flask application with multimodal processing${NC}"
echo -e "${GREEN}✅ Enhanced HTML template with modern UI${NC}"
echo -e "${GREEN}✅ Complete CSS with responsive design${NC}"
echo -e "${GREEN}✅ Full JavaScript with multimodal features${NC}"
echo -e "${GREEN}✅ Configuration validation tools${NC}"
echo -e "${GREEN}✅ Comprehensive documentation${NC}"
echo -e "${GREEN}✅ Development and deployment scripts${NC}"

echo -e "\n${BLUE}🚀 Next Steps:${NC}"
echo -e "${YELLOW}1.${NC} Configure your API keys:"
echo -e "   ${BLUE}nano .env${NC}"
echo -e ""
echo -e "${YELLOW}2.${NC} Validate your configuration:"
echo -e "   ${BLUE}uv run python validate_config.py${NC}"
echo -e ""
echo -e "${YELLOW}3.${NC} Start the application:"
echo -e "   ${BLUE}./run.sh${NC}"
echo -e ""
echo -e "${YELLOW}4.${NC} Test the deployment:"
echo -e "   ${BLUE}# In another terminal:${NC}"
echo -e "   ${BLUE}uv run python test_deployment.py${NC}"
echo -e ""
echo -e "${YELLOW}5.${NC} Open in browser:"
echo -e "   ${BLUE}http://localhost:5000${NC}"