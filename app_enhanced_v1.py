#!/usr/bin/env python3
"""
Enhanced Multimodal RAG Service Manual Assistant with Agent Integration
Fixed ChromaDB collection issues and intelligent query routing
"""

from flask import Flask, request, jsonify, session, render_template, send_from_directory, Response
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
import asyncio
import uuid

# Import the agent system
from agents.agent_pipeline import AgentPipeline
from config.agent_config import load_agent_config

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

# Initialize ChromaDB with better error handling
script_dir = os.path.dirname(os.path.abspath(__file__))
chroma_db_path = os.path.join(script_dir, os.getenv('CHROMA_DB_PATH', 'chroma_db'))
web_downloads_path = os.path.join(script_dir, os.getenv('WEB_DOWNLOADS_PATH', 'web_downloads'))

# Create directories if they don't exist
os.makedirs(chroma_db_path, exist_ok=True)
os.makedirs(web_downloads_path, exist_ok=True)
os.makedirs('logs', exist_ok=True)

try:
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    logger.info(f"ChromaDB initialized successfully at {chroma_db_path}")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    chroma_client = None

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

class LegacyMultimodalProcessor:
    """Legacy processor for service manuals with multimodal content"""
    
    def __init__(self, openai_client: OpenAI, chroma_client: chromadb.Client):
        self.openai_client = openai_client
        self.chroma_client = chroma_client
        
        # Create separate collections for different content types
        self.text_collection = self._get_or_create_collection("service_manual_text")
        self.table_collection = self._get_or_create_collection("service_manual_tables") 
        self.figure_collection = self._get_or_create_collection("service_manual_figures")
        
        logger.info("Legacy multimodal processor initialized with separate collections")
    
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
                    
                    # Save image as file instead of storing base64 in metadata
                    element_id = f"{manual_name}_p{page_num}_fig_{img_index}"
                    images_dir = os.path.join(os.path.dirname(__file__), "static", "images")
                    os.makedirs(images_dir, exist_ok=True)
                    
                    image_filename = f"{element_id}.png"
                    image_file_path = os.path.join(images_dir, image_filename)
                    
                    # Save image file
                    with open(image_file_path, 'wb') as f:
                        f.write(img_data)
                    
                    logger.info(f"✅ Saved image file: {image_file_path} ({len(img_data)} bytes)")
                    
                    element = DocumentElement(
                        element_id=element_id,
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
                            "image_file_path": image_file_path,
                            "image_filename": image_filename,
                            "image_format": "png",
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
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
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
                model=os.getenv('OPENAI_VISION_MODEL', 'gpt-4o-mini'),
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

class EnhancedMultimodalProcessor:
    """Enhanced processor with better collection management and error handling"""
    
    def __init__(self, openai_client: OpenAI, chroma_client: chromadb.Client, web_search_agent=None):
        self.openai_client = openai_client
        self.chroma_client = chroma_client
        self.web_search_agent = web_search_agent
        self.collections = {}
        
        # Initialize optimized processor for batch operations
        try:
            from utils.optimized_processor import OptimizedMultimodalProcessor
            self.optimized_processor = OptimizedMultimodalProcessor(openai_client, chroma_client, web_search_agent)
            logger.info("Optimized batch processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize optimized processor: {e}")
            self.optimized_processor = None
        
        if chroma_client:
            self._initialize_collections()
        else:
            logger.error("ChromaDB client not available - multimodal processing disabled")
    
    def _initialize_collections(self):
        """Initialize collections with proper error handling"""
        collection_names = ["service_manual_text", "service_manual_tables", "service_manual_figures"]
        
        for name in collection_names:
            try:
                # Try to get existing collection first
                collection = self.chroma_client.get_collection(name)
                logger.info(f"Found existing collection: {name} with {collection.count()} items")
            except Exception:
                # Create new collection if it doesn't exist
                try:
                    collection = self.chroma_client.create_collection(
                        name=name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Created new collection: {name}")
                except Exception as e:
                    logger.error(f"Failed to create collection {name}: {e}")
                    collection = None
            
            self.collections[name] = collection
        
        # Set collection references for backward compatibility
        self.text_collection = self.collections.get("service_manual_text")
        self.table_collection = self.collections.get("service_manual_tables")
        self.figure_collection = self.collections.get("service_manual_figures")
        
        # Validate collections
        valid_collections = sum(1 for c in self.collections.values() if c is not None)
        logger.info(f"Initialized {valid_collections}/{len(collection_names)} collections successfully")
    
    def clear_irrelevant_data(self, manual_type: str = "automotive") -> Dict[str, int]:
        """Clear data that doesn't match the expected manual type"""
        cleared_counts = {"text": 0, "tables": 0, "figures": 0}
        
        try:
            irrelevant_keywords = {
                "automotive": ["refrigeration", "cooling unit", "compressor assembly", "door replacement", "evaporator"],
                "hvac": ["alternator", "transmission", "engine oil", "carburetor"],
                "general": []
            }
            
            keywords_to_remove = irrelevant_keywords.get(manual_type, [])
            if not keywords_to_remove:
                logger.info("No irrelevant keywords specified for manual type")
                return cleared_counts
            
            for collection_name, collection in self.collections.items():
                if not collection:
                    continue
                
                try:
                    # Get all documents to check for irrelevant content
                    all_docs = collection.get(include=['documents', 'metadatas'])
                    ids_to_remove = []
                    
                    for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                        if any(keyword.lower() in doc.lower() for keyword in keywords_to_remove):
                            ids_to_remove.append(all_docs['ids'][i])
                    
                    if ids_to_remove:
                        collection.delete(ids=ids_to_remove)
                        cleared_counts[collection_name.split('_')[-1]] = len(ids_to_remove)
                        logger.info(f"Removed {len(ids_to_remove)} irrelevant documents from {collection_name}")
                
                except Exception as e:
                    logger.error(f"Error clearing irrelevant data from {collection_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error during irrelevant data clearing: {e}")
        
        return cleared_counts
    
    def intelligent_search(self, query: str, search_strategy: Dict[str, Any], n_results: int = 5) -> List[Dict]:
        """Enhanced search using agent-provided strategy"""
        search_scope = search_strategy.get('search_scope', 'local_first')
        logger.info(f"Performing intelligent search with strategy: {search_scope}")
        
        local_results = []
        web_results = []
        
        # Step 1: Search local collections first (unless external_only)
        if search_scope != "external_only":
            local_results = self._search_local_collections(query, search_strategy, n_results)
            logger.info(f"Local search returned {len(local_results)} results")
        
        # Step 2: Determine if we need external search
        needs_external_search = self._should_search_externally(search_strategy, local_results)
        
        if needs_external_search:
            logger.info("Triggering external web search")
            web_results = self._search_external_sources(query, search_strategy, n_results)
            logger.info(f"Web search returned {len(web_results)} results")
        
        # Step 3: Combine and rank all results
        all_results = self._merge_and_rank_results(local_results, web_results, search_strategy)
        
        # Step 4: Apply final filtering
        confidence_threshold = search_strategy.get("confidence_threshold", 0.7)
        filtered_results = [r for r in all_results if r.get("similarity", 0) >= confidence_threshold]
        
        # Sort by relevance and return top results
        filtered_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        final_results = filtered_results[:n_results]
        
        logger.info(f"Intelligent search returned {len(final_results)} results")
        return final_results
    
    def _search_collection(self, collection, query: str, n_results: int) -> List[Dict]:
        """Search a specific collection with error handling"""
        try:
            if not collection or collection.count() == 0:
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
            logger.error(f"Search error in collection: {e}")
            return []
    
    def _search_local_collections(self, query: str, search_strategy: Dict[str, Any], n_results: int) -> List[Dict]:
        """Search local ChromaDB collections"""
        results = []
        content_priorities = search_strategy.get("content_type_priority", ["text", "tables", "diagrams"])
        primary_keywords = search_strategy.get("primary_keywords", [])
        
        # Enhanced query with keywords from agent analysis
        enhanced_query = query
        if primary_keywords:
            enhanced_query = f"{query} {' '.join(primary_keywords[:3])}"
        
        # Search each content type based on priority
        results_per_type = search_strategy.get("max_results_per_type", 3)
        
        for content_type in content_priorities:
            if content_type == "text" and self.text_collection:
                text_results = self._search_collection(self.text_collection, enhanced_query, results_per_type)
                results.extend(text_results)
            
            elif content_type == "tables" and self.table_collection:
                table_results = self._search_collection(self.table_collection, enhanced_query, results_per_type)
                results.extend(table_results)
            
            elif content_type in ["diagrams", "figures"] and self.figure_collection:
                figure_results = self._search_collection(self.figure_collection, enhanced_query, results_per_type)
                results.extend(figure_results)
        
        # If no results found, try fallback search
        if not results and search_strategy.get("enable_hybrid_search", True):
            logger.info("No results from targeted local search, trying broader local search")
            results = self._fallback_search(query, n_results)
        
        return results
    
    def _should_search_externally(self, search_strategy: Dict[str, Any], local_results: List[Dict]) -> bool:
        """Determine if external search is needed based on strategy and local results"""
        search_scope = search_strategy.get("search_scope", "local_first")
        
        # Always search externally for these scopes
        if search_scope in ["external_preferred", "external_only"]:
            return True
        
        # Search externally if no good local results
        if search_scope == "hybrid" and len(local_results) < 2:
            return True
        
        # Check if strategy explicitly requests external search
        if search_strategy.get("needs_external_search", False):
            return True
        
        # Search externally for specific equipment with high confidence
        search_metadata = search_strategy.get("search_metadata", {})
        if (search_metadata.get("has_specific_equipment", False) and 
            search_metadata.get("equipment_confidence", 0) > 0.8):
            return True
        
        return False
    
    def _search_external_sources(self, query: str, search_strategy: Dict[str, Any], n_results: int) -> List[Dict]:
        """Search external sources using WebSearchAgent"""
        if not self.web_search_agent:
            logger.warning("WebSearchAgent not available for external search")
            return []
        
        try:
            # Create search strategy object for web search agent
            from models.agent_models import SearchStrategy, SearchScope, ContentType, IntentCategory, EquipmentCategory, SearchMetadata
            
            search_metadata = search_strategy.get("search_metadata", {})
            
            # Convert string values to enum values
            intent_map = {
                "installation": IntentCategory.INSTALLATION,
                "troubleshooting": IntentCategory.TROUBLESHOOTING,
                "specifications": IntentCategory.SPECIFICATIONS,
                "diagram_request": IntentCategory.DIAGRAM_REQUEST,
                "general_inquiry": IntentCategory.GENERAL_INQUIRY
            }
            
            equipment_map = {
                "automotive": EquipmentCategory.AUTOMOTIVE,
                "hvac": EquipmentCategory.HVAC,
                "electronics": EquipmentCategory.ELECTRONICS,
                "appliance": EquipmentCategory.APPLIANCE,
                "industrial": EquipmentCategory.INDUSTRIAL,
                "unknown": EquipmentCategory.UNKNOWN
            }
            
            scope_map = {
                "external_preferred": SearchScope.EXTERNAL_PREFERRED,
                "external_only": SearchScope.EXTERNAL_ONLY,
                "hybrid": SearchScope.HYBRID,
                "local_first": SearchScope.LOCAL_FIRST
            }
            
            content_type_map = {
                "text": ContentType.TEXT,
                "tables": ContentType.TABLES,
                "diagrams": ContentType.DIAGRAMS,
                "videos": ContentType.VIDEOS
            }
            
            # Create proper search metadata
            web_search_metadata = SearchMetadata(
                intent_primary=intent_map.get(search_metadata.get("intent_primary", "general_inquiry"), IntentCategory.GENERAL_INQUIRY),
                intent_confidence=search_metadata.get("intent_confidence", 0.7),
                equipment_category=equipment_map.get(search_metadata.get("equipment_category", "unknown"), EquipmentCategory.UNKNOWN),
                equipment_confidence=search_metadata.get("equipment_confidence", 0.5),
                has_specific_equipment=search_metadata.get("has_specific_equipment", False),
            )
            
            # Create search strategy for web agent
            web_search_strategy = SearchStrategy(
                search_scope=scope_map.get(search_strategy.get("search_scope", "external_preferred"), SearchScope.EXTERNAL_PREFERRED),
                primary_keywords=search_strategy.get("primary_keywords", []),
                secondary_keywords=search_strategy.get("secondary_keywords", []),
                content_type_priority=[content_type_map.get(ct, ContentType.TEXT) for ct in search_strategy.get("content_type_priority", ["text"])],
                confidence_threshold=search_strategy.get("confidence_threshold", 0.7),
                max_results_per_type=search_strategy.get("max_results_per_type", 3),
                enable_hybrid_search=search_strategy.get("enable_hybrid_search", True),
                search_metadata=web_search_metadata
            )
            
            # Call web search agent
            response = asyncio.run(self.web_search_agent.process_query(query, web_search_strategy))
            
            if response.success:
                search_result = response.data.get('search_result', {})
                web_resources = search_result.get('resources', [])
                
                # Convert web resources to expected format
                return self._convert_web_resources_to_results(web_resources)
            else:
                logger.error(f"Web search failed: {response.errors}")
                return []
                
        except Exception as e:
            logger.error(f"Error in external search: {e}")
            return []
    
    def _convert_web_resources_to_results(self, web_resources: List) -> List[Dict]:
        """Convert WebResource objects to the expected result format"""
        results = []
        
        for resource in web_resources:
            if isinstance(resource, dict):
                # Resource is already a dictionary
                resource_dict = resource
            else:
                # Convert object to dictionary
                resource_dict = resource if hasattr(resource, '__dict__') else {}
            
            # Extract content - prefer extracted_content, fallback to preview
            content = resource_dict.get('extracted_content', '') or resource_dict.get('content_preview', '')
            
            # Calculate similarity score based on relevance
            similarity = resource_dict.get('relevance_score', 0.8)
            
            # Determine element type based on content type
            content_type = resource_dict.get('content_type', 'article')
            element_type_map = {
                'pdf': 'manual',
                'forum_post': 'discussion',
                'video': 'tutorial',
                'article': 'web_resource',
                'manual': 'manual'
            }
            element_type = element_type_map.get(content_type, 'web_resource')
            
            # Create metadata
            metadata = {
                'source': resource_dict.get('source', 'web'),
                'url': resource_dict.get('url', ''),
                'title': resource_dict.get('title', ''),
                'content_type': content_type,
                'download_path': resource_dict.get('download_path', ''),
                **resource_dict.get('metadata', {})
            }
            
            # Create result in expected format
            result = {
                'content': content,
                'similarity': similarity,
                'metadata': metadata,
                'element_type': element_type
            }
            
            results.append(result)
        
        return results
    
    def _merge_and_rank_results(self, local_results: List[Dict], web_results: List[Dict], search_strategy: Dict[str, Any]) -> List[Dict]:
        """Merge and rank results from local and web sources"""
        all_results = []
        
        # Add local results with slight boost for being local
        for result in local_results:
            result_copy = result.copy()
            result_copy['similarity'] = min(result_copy.get('similarity', 0) * 1.05, 1.0)  # 5% boost
            result_copy['source_type'] = 'local'
            all_results.append(result_copy)
        
        # Add web results
        for result in web_results:
            result_copy = result.copy()
            result_copy['source_type'] = 'web'
            # Boost manufacturer sources
            if result_copy.get('metadata', {}).get('source') == 'manufacturer':
                result_copy['similarity'] = min(result_copy.get('similarity', 0) * 1.15, 1.0)  # 15% boost
            all_results.append(result_copy)
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return all_results
    
    def _fallback_search(self, query: str, n_results: int) -> List[Dict]:
        """Fallback search across all collections"""
        all_results = []
        
        for collection in [self.text_collection, self.table_collection, self.figure_collection]:
            if collection:
                results = self._search_collection(collection, query, n_results // 3)
                all_results.extend(results)
        
        return all_results
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text with error handling - DEPRECATED: Use batch processing instead"""
        logger.warning("Individual embedding call detected - consider using batch processing")
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using batch processing"""
        if self.optimized_processor:
            return self.optimized_processor.embedding_optimizer.get_embeddings_batch(texts)
        else:
            # Fallback to individual calls (not recommended)
            logger.warning(f"Using fallback individual embedding calls for {len(texts)} texts")
            return [self._get_text_embedding(text) for text in texts]

# Initialize the agent pipeline first
try:
    agent_config = load_agent_config()
    agent_pipeline = AgentPipeline(openai_client, agent_config)
    logger.info("Agent pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent pipeline: {e}")
    agent_pipeline = None

# Initialize the enhanced multimodal processor with web search agent
try:
    web_search_agent = agent_pipeline.web_search if agent_pipeline else None
    multimodal_processor = EnhancedMultimodalProcessor(openai_client, chroma_client, web_search_agent)
    logger.info("Enhanced multimodal processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize enhanced multimodal processor: {e}")
    multimodal_processor = None

# Legacy collection for backward compatibility
collection_name = "pdf_documents"
legacy_collection = None

if chroma_client:
    try:
        legacy_collection = chroma_client.get_collection(collection_name)
        logger.info(f"Found legacy collection with {legacy_collection.count()} items")
    except:
        try:
            legacy_collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created new legacy collection")
        except Exception as e:
            logger.error(f"Failed to create legacy collection: {e}")

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
    """Enhanced health check endpoint"""
    try:
        stats = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'chromadb': bool(chroma_client),
                'multimodal_processor': bool(multimodal_processor),
                'agent_pipeline': bool(agent_pipeline),
                'openai_client': bool(openai_client.api_key)
            },
            'collections': {}
        }
        
        if legacy_collection:
            stats['collections']['legacy'] = legacy_collection.count()
        
        if multimodal_processor and multimodal_processor.collections:
            for name, collection in multimodal_processor.collections.items():
                if collection:
                    try:
                        stats['collections'][name] = collection.count()
                    except:
                        stats['collections'][name] = 'error'
        
        # Check agent pipeline status if available
        if agent_pipeline:
            try:
                pipeline_status = asyncio.run(agent_pipeline.get_pipeline_status())
                stats['agent_pipeline_status'] = pipeline_status
            except Exception as e:
                stats['agent_pipeline_status'] = {'error': str(e)}
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/chat', methods=['POST'])
def enhanced_chat():
    """Enhanced chat endpoint with agent pipeline integration"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        auto_search = data.get('auto_search', True)
        
        logger.info(f"Processing enhanced chat message: {user_message}")
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'})
        
        if not openai_client.api_key:
            return jsonify({'error': 'OpenAI API key not configured'})
        
        # Step 1: Use agent pipeline for query analysis (if available)
        processing_info = {
            'agent_analysis_used': False,
            'found_relevant_docs': False,
            'relevant_docs_count': 0,
            'retrieved_chunks': [],
            'processing_summary': 'Basic processing',
            'search_strategy': None
        }
        
        search_strategy = None
        
        if agent_pipeline:
            try:
                # Validate query first
                validation = agent_pipeline.validate_query(user_message)
                if not validation['valid']:
                    return jsonify({
                        'error': f"Query validation failed: {'; '.join(validation['issues'])}",
                        'recommendations': validation.get('recommendations', [])
                    })
                
                # Process query through agent pipeline
                pipeline_result = asyncio.run(agent_pipeline.process_query(user_message))
                
                if pipeline_result['success']:
                    processing_info['agent_analysis_used'] = True
                    processing_info['processing_summary'] = agent_pipeline.get_processing_summary(pipeline_result)
                    search_strategy = pipeline_result.get('search_strategy')
                    
                    logger.info(f"Agent analysis complete: {processing_info['processing_summary']}")
                else:
                    logger.warning(f"Agent pipeline failed: {pipeline_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"Agent pipeline error: {e}")
        
        # Step 2: Search for relevant content using enhanced search
        search_results = []
        
        if multimodal_processor:
            if search_strategy:
                # Use intelligent search with agent strategy
                search_results = multimodal_processor.intelligent_search(user_message, search_strategy, n_results=5)
                logger.info(f"Intelligent search returned {len(search_results)} results")
            else:
                # Fallback to basic search
                search_results = multimodal_processor._fallback_search(user_message, n_results=5)
                logger.info(f"Fallback search returned {len(search_results)} results")
        elif legacy_collection:
            # Legacy search
            search_results = legacy_search(user_message, n_results=5)
            logger.info(f"Legacy search returned {len(search_results)} results")
        
        used_rag = len(search_results) > 0
        processing_info.update({
            'found_relevant_docs': used_rag,
            'relevant_docs_count': len(search_results),
            'retrieved_chunks': search_results[:3] if search_results else [],
            'search_strategy': search_strategy
        })
        
        # Step 3: Build conversation context
        if 'conversation' not in session:
            session['conversation'] = []
        
        if search_results:
            # Create enhanced context from search results
            context_parts = []
            for result in search_results:
                element_type = result.get('element_type', 'text')
                content = result['content']
                similarity = result.get('similarity', 0)
                
                if element_type == 'web_resource':
                    source = result.get('metadata', {}).get('source', 'web')
                    url = result.get('metadata', {}).get('url', '')
                    context_parts.append(f"🌐 WEB RESOURCE from {source.upper()} (relevance: {similarity:.2f}):\n{content}\nSource: {url}")
                elif element_type == 'table':
                    context_parts.append(f"📊 TABLE DATA (relevance: {similarity:.2f}):\n{content}")
                elif element_type == 'figure':
                    context_parts.append(f"🖼️ DIAGRAM/FIGURE (relevance: {similarity:.2f}):\n{content}")
                else:
                    context_parts.append(f"📄 TEXT (relevance: {similarity:.2f}):\n{content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Enhanced system message with agent insights
            system_parts = [
                "You are a helpful technical assistant specializing in service manuals and repair documentation.",
                "Use the following context from documents to answer the user's question."
            ]
            
            if processing_info['agent_analysis_used']:
                system_parts.append("The query has been analyzed by an intelligent agent system for optimal processing.")
                if search_strategy:
                    intent = search_strategy.get('search_metadata', {}).get('intent_primary', 'unknown')
                    system_parts.append(f"Query intent: {intent}")
            
            context_sources = "uploaded documents"
            
            system_parts.extend([
                "The context may include text, tables with specifications, descriptions of diagrams/figures, and external web resources.",
                "Present table data clearly. Explain what diagrams show. Clearly indicate the source of information.",
                "If context doesn't fully answer the question, supplement with general knowledge but clearly indicate what comes from documents vs. web resources vs. general knowledge.",
                f"\n📄 Answer based on {context_sources}\n\nCONTEXT:\n{context}",
                "Provide a helpful, detailed response based on the context and your expertise."
            ])
            
            system_message = " ".join(system_parts)
            
            messages = [{"role": "system", "content": system_message}]
            messages.extend(session['conversation'][-6:])  # Keep recent conversation
            messages.append({"role": "user", "content": user_message})
            
        else:
            # No relevant documents found - use general knowledge
            logger.info("No relevant documents found, using general knowledge")
            messages = session['conversation'][-8:]
            messages.append({"role": "user", "content": user_message})
        
        # Step 4: Generate response
        response = openai_client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
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
        logger.error(f"Enhanced chat error: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'})

def legacy_search(query: str, n_results: int = 5) -> List[Dict]:
    """Legacy search function for backward compatibility"""
    try:
        if not legacy_collection or legacy_collection.count() == 0:
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

@app.route('/clear-irrelevant', methods=['POST'])
def clear_irrelevant_data():
    """Clear irrelevant data from collections"""
    try:
        data = request.get_json()
        manual_type = data.get('manual_type', 'automotive')
        
        if not multimodal_processor:
            return jsonify({'error': 'Multimodal processor not available'})
        
        cleared_counts = multimodal_processor.clear_irrelevant_data(manual_type)
        
        return jsonify({
            'message': f'Cleared irrelevant {manual_type} data',
            'cleared_counts': cleared_counts,
            'total_cleared': sum(cleared_counts.values())
        })
        
    except Exception as e:
        logger.error(f"Clear irrelevant data error: {e}")
        return jsonify({'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF with enhanced multimodal support"""
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
                # Use enhanced multimodal processing
                processing_steps.append({
                    'title': '🔍 Analyzing PDF structure',
                    'status': 'processing',
                    'details': f'Processing: {filename}'
                })
                
                # Use the built-in multimodal processor (no import needed)
                legacy_processor = LegacyMultimodalProcessor(openai_client, chroma_client)
                
                # Process the PDF using the original method
                elements, stats = legacy_processor.process_service_manual(temp_path, filename)
                
                processing_steps[-1]['status'] = 'complete'
                processing_steps[-1]['details'] = f'Found {len(elements)} elements across {stats["pages"]} pages'
                
                # Add content breakdown
                processing_steps.append({
                    'title': '📊 Content Analysis',
                    'status': 'complete',
                    'details': f'Text blocks: {stats["text"]}, Tables: {stats["tables"]}, Figures: {stats["figures"]}'
                })
                
                # Generate embeddings and store in enhanced collections using BATCH PROCESSING
                processing_steps.append({
                    'title': '🧠 Generating embeddings (batch optimized)',
                    'status': 'processing',
                    'details': 'Converting content to vector embeddings using batch processing...'
                })
                
                # Use optimized batch processing if available
                if multimodal_processor.optimized_processor:
                    processing_stats = multimodal_processor.optimized_processor.process_and_store_elements_optimized(
                        elements, filename
                    )
                    
                    stored_counts = {
                        "text": processing_stats.text_blocks,
                        "tables": processing_stats.tables,
                        "figures": processing_stats.figures,
                        "errors": processing_stats.total_elements - processing_stats.processed_elements
                    }
                    
                    # Update processing step with detailed info
                    processing_steps[-1]['details'] = (
                        f'Batch processed {processing_stats.processed_elements} elements. '
                        f'Generated {processing_stats.embeddings_generated} new embeddings, '
                        f'used {processing_stats.embeddings_cached} cached. '
                        f'Cost: ${processing_stats.cost_estimate:.4f}'
                    )
                    
                else:
                    # Fallback to individual processing (LEGACY - NOT RECOMMENDED)
                    logger.warning("Using legacy individual embedding processing - this is expensive!")
                    stored_counts = {"text": 0, "tables": 0, "figures": 0, "errors": 0}
                    
                    # Collect all texts for batch processing
                    texts_to_embed = []
                    text_element_map = {}
                    
                    for i, element in enumerate(elements):
                        if element.content:
                            texts_to_embed.append(element.content)
                            text_element_map[len(texts_to_embed) - 1] = i
                    
                    # Get embeddings in batch
                    if texts_to_embed:
                        embeddings = multimodal_processor.get_embeddings_batch(texts_to_embed)
                        
                        # Assign embeddings back to elements
                        for text_idx, embedding in enumerate(embeddings):
                            if text_idx in text_element_map:
                                element_idx = text_element_map[text_idx]
                                elements[element_idx].text_embedding = embedding
                    
                    # Store elements
                    for element in elements:
                        try:
                            if not element.text_embedding or not element.content:
                                stored_counts["errors"] += 1
                                continue
                            
                            # Store in appropriate enhanced collection
                            metadata = element.metadata.copy()
                            metadata.update({
                                "element_type": element.element_type,
                                "page_number": element.page_number,
                                "bbox": str(element.bbox),
                                "element_id": element.element_id
                            })
                            
                            # Choose collection based on element type
                            if element.element_type in ["text", "heading"]:
                                collection = multimodal_processor.text_collection
                                stored_counts["text"] += 1
                            elif element.element_type == "table":
                                collection = multimodal_processor.table_collection
                                # Add table-specific metadata
                                if element.table_data:
                                    metadata["table_html"] = element.table_data.get("html", "")
                                    metadata["table_csv"] = element.table_data.get("csv", "")
                                stored_counts["tables"] += 1
                            elif element.element_type in ["figure", "diagram"]:
                                collection = multimodal_processor.figure_collection
                                # Ensure image data is included in metadata for figures
                                if element.metadata.get("image_base64"):
                                    metadata["image_base64"] = element.metadata["image_base64"]
                                    metadata["image_format"] = element.metadata.get("image_format", "png")
                                stored_counts["figures"] += 1
                            else:
                                collection = multimodal_processor.text_collection
                                stored_counts["text"] += 1
                            
                            # Store with text embedding
                            collection.add(
                                ids=[element.element_id],
                                documents=[element.content],
                                embeddings=[element.text_embedding],
                                metadatas=[metadata]
                            )
                            
                        except Exception as e:
                            logger.error(f"Error processing element {element.element_id}: {e}")
                            stored_counts["errors"] += 1
                
                processing_steps[-1]['status'] = 'complete'
                processing_steps[-1]['details'] = f'Stored {sum(stored_counts.values()) - stored_counts["errors"]} elements'
                
                if stored_counts["errors"] > 0:
                    processing_steps.append({
                        'title': '⚠️ Processing warnings',
                        'status': 'complete',
                        'details': f'{stored_counts["errors"]} elements had processing errors'
                    })
                
                return jsonify({
                    'message': f'Successfully processed "{filename}" with enhanced multimodal analysis',
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

@app.route('/agent-status')
def agent_status():
    """Get agent system status"""
    try:
        status = {
            'agent_pipeline_available': bool(agent_pipeline),
            'multimodal_processor_available': bool(multimodal_processor),
            'optimized_processor_available': bool(multimodal_processor and multimodal_processor.optimized_processor),
            'timestamp': datetime.now().isoformat()
        }
        
        if agent_pipeline:
            try:
                pipeline_status = asyncio.run(agent_pipeline.get_pipeline_status())
                status['pipeline_status'] = pipeline_status
            except Exception as e:
                status['pipeline_error'] = str(e)
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/optimization-stats')
def optimization_stats():
    """Get embedding optimization statistics"""
    try:
        if not multimodal_processor or not multimodal_processor.optimized_processor:
            return jsonify({'error': 'Optimized processor not available'})
        
        stats = multimodal_processor.optimized_processor.get_optimization_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting optimization stats: {e}")
        return jsonify({'error': str(e)})

@app.route('/estimate-cost', methods=['POST'])
def estimate_embedding_cost():
    """Estimate cost for embedding operation"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'No texts provided'})
        
        if not multimodal_processor or not multimodal_processor.optimized_processor:
            # Rough estimate without optimizer
            total_tokens = sum(len(text.split()) for text in texts)
            estimated_cost = (total_tokens / 1000) * 0.0001
            
            return jsonify({
                'total_texts': len(texts),
                'estimated_tokens': total_tokens,
                'estimated_cost': estimated_cost,
                'cache_hits': 0,
                'optimization_available': False
            })
        
        # Use optimizer for accurate estimate
        estimate = multimodal_processor.optimized_processor.embedding_optimizer.estimate_cost(texts)
        
        return jsonify({
            'total_texts': estimate.total_texts,
            'estimated_tokens': estimate.total_tokens,
            'estimated_cost': estimate.estimated_cost,
            'batch_count': estimate.batch_count,
            'cache_hits': estimate.cache_hits,
            'cache_savings': estimate.cache_savings,
            'optimization_available': True
        })
        
    except Exception as e:
        logger.error(f"Error estimating cost: {e}")
        return jsonify({'error': str(e)})

def normalize_content_type(raw_type):
    """Normalize content type from collection names to standard types"""
    type_map = {
        'figures': 'figure',
        'tables': 'table', 
        'text': 'text'
    }
    return type_map.get(raw_type, 'text')

@app.route('/api/content/<element_id>')
def get_content_details(element_id):
    """Get detailed content information for an element"""
    try:
        if not multimodal_processor:
            return jsonify({'error': 'Multimodal processor not available'}), 404
        
        # Search across all collections to find the element
        element_data = None
        collection_type = None
        
        for coll_name, collection in multimodal_processor.collections.items():
            if not collection:
                continue
                
            try:
                # Get all documents and find the matching element_id
                results = collection.get(
                    where={"element_id": element_id},
                    include=['documents', 'metadatas']
                )
                
                if results['documents']:
                    element_data = {
                        'content': results['documents'][0],
                        'metadata': results['metadatas'][0],
                        'collection': coll_name
                    }
                    # Normalize the content type (figures -> figure, tables -> table)
                    raw_type = coll_name.split('_')[-1]  # Extract 'text', 'tables', or 'figures'
                    collection_type = normalize_content_type(raw_type)
                    logger.info(f"Found element {element_id} in {coll_name}, normalized type: {raw_type} -> {collection_type}")
                    break
                    
            except Exception as e:
                logger.error(f"Error searching collection {coll_name}: {e}")
                continue
        
        if not element_data:
            logger.warning(f"Element {element_id} not found in any collection")
            return jsonify({'error': 'Content not found'}), 404
        
        # Enhanced response with better type detection
        response_data = {
            'element_id': element_id,
            'content_type': collection_type,
            'content': element_data['content'],
            'metadata': element_data['metadata'],
            'has_image': element_data['metadata'].get('block_type') == 'figure' or collection_type == 'figure',
            'has_table_data': element_data['metadata'].get('block_type') == 'table' or collection_type == 'table'
        }
        
        # Add specific data based on content type
        if collection_type == 'table':
            response_data.update({
                'table_html': element_data['metadata'].get('table_html', ''),
                'table_csv': element_data['metadata'].get('table_csv', ''),
                'table_rows': element_data['metadata'].get('table_rows', 0),
                'table_cols': element_data['metadata'].get('table_cols', 0),
                'table_summary': element_data['metadata'].get('table_summary', '')
            })
        elif collection_type == 'figure':
            response_data.update({
                'image_width': element_data['metadata'].get('image_width', 0),
                'image_height': element_data['metadata'].get('image_height', 0),
                'has_ocr_text': element_data['metadata'].get('has_ocr_text', False),
                'ocr_text': element_data['metadata'].get('ocr_text', ''),
                'description': element_data['metadata'].get('description', ''),
                'has_stored_image': 'image_base64' in element_data['metadata']
            })
        
        logger.info(f"Returning content details for {element_id}: type={collection_type}, has_image={response_data.get('has_stored_image', False)}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting content details for {element_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<element_id>')
def get_element_image(element_id):
    """Serve image data for a figure element with enhanced debugging"""
    try:
        logger.info(f"🖼️ Image request for element: {element_id}")
        
        if not multimodal_processor:
            logger.error("Multimodal processor not available")
            return jsonify({'error': 'Multimodal processor not available'}), 404
        
        # Search figure collection for the element
        if not multimodal_processor.figure_collection:
            logger.error("Figure collection not available")
            return jsonify({'error': 'Figure collection not available'}), 404
        
        try:
            results = multimodal_processor.figure_collection.get(
                where={"element_id": element_id},
                include=['documents', 'metadatas']
            )
            
            logger.info(f"Database search results: {len(results.get('documents', []))} documents found")
            
            if not results['documents'] or not results['metadatas']:
                logger.warning(f"No image data found for element: {element_id}")
                return jsonify({
                    'error': 'Image not found in database',
                    'details': f'Element {element_id} not found in figure collection'
                }), 404
            
            metadata = results['metadatas'][0]
            logger.info(f"Found element metadata: manual={metadata.get('manual_name')}, page={metadata.get('page_number')}, has_base64={'image_base64' in metadata}")
            
            # Check if we have stored image file
            if 'image_file_path' in metadata:
                image_file_path = metadata['image_file_path']
                logger.info(f"✅ Found stored image file path for {element_id}: {image_file_path}")
                
                try:
                    if os.path.exists(image_file_path):
                        with open(image_file_path, 'rb') as f:
                            image_data = f.read()
                        
                        image_format = metadata.get('image_format', 'png')
                        logger.info(f"Loaded image file: {len(image_data)} bytes, format: {image_format}")
                        
                        return Response(
                            image_data,
                            mimetype=f'image/{image_format}',
                            headers={
                                'Content-Disposition': f'inline; filename="{element_id}.{image_format}"',
                                'Cache-Control': 'public, max-age=3600',
                                'X-Image-Source': 'file_storage'
                            }
                        )
                    else:
                        logger.warning(f"⚠️ Image file not found: {image_file_path}")
                        return jsonify({
                            'error': 'Image file not found',
                            'details': f'File path in database but file missing: {image_file_path}'
                        }), 404
                        
                except Exception as e:
                    logger.error(f"❌ Error reading image file for {element_id}: {e}")
                    return jsonify({
                        'error': 'Failed to read image file',
                        'details': str(e)
                    }), 500
            
            # Fallback: Check for legacy base64 data  
            elif 'image_base64' in metadata:
                logger.info(f"✅ Found legacy base64 image data for {element_id}")
                # This won't work for existing data since ChromaDB dropped it, but kept for future compatibility
                import base64
                
                try:
                    image_base64 = metadata['image_base64']
                    if not image_base64:
                        logger.warning(f"⚠️ Empty base64 data for {element_id}")
                        return jsonify({
                            'error': 'Empty image data',
                            'details': 'Base64 field exists but is empty - data was likely dropped by ChromaDB'
                        }), 404
                    
                    image_data = base64.b64decode(image_base64)
                    image_format = metadata.get('image_format', 'png')
                    
                    logger.info(f"Decoded legacy image: {len(image_data)} bytes, format: {image_format}")
                    
                    return Response(
                        image_data,
                        mimetype=f'image/{image_format}',
                        headers={
                            'Content-Disposition': f'inline; filename="{element_id}.{image_format}"',
                            'Cache-Control': 'public, max-age=3600',
                            'X-Image-Source': 'legacy_base64'
                        }
                    )
                except Exception as e:
                    logger.error(f"❌ Error decoding legacy base64 data for {element_id}: {e}")
                    return jsonify({
                        'error': 'Failed to decode legacy base64 data',
                        'details': str(e)
                    }), 500
            
            # Fallback: Try to re-extract from PDF if we have the necessary metadata
            manual_name = metadata.get('manual_name')
            page_number = metadata.get('page_number')
            
            logger.info(f"⚠️ No stored image data, attempting re-extraction from {manual_name}, page {page_number}")
            
            if manual_name and page_number is not None:
                # Try to find the PDF file and re-extract
                pdf_path = find_pdf_file(manual_name)
                logger.info(f"PDF search result: {pdf_path}")
                
                if pdf_path and os.path.exists(pdf_path):
                    try:
                        image_data = re_extract_image_from_pdf(pdf_path, page_number, element_id)
                        if image_data:
                            logger.info(f"✅ Successfully re-extracted {len(image_data)} bytes from PDF")
                            return Response(
                                image_data,
                                mimetype='image/png',
                                headers={
                                    'Content-Disposition': f'inline; filename="{element_id}.png"',
                                    'Cache-Control': 'public, max-age=3600',
                                    'X-Image-Source': 'pdf_reextraction'
                                }
                            )
                        else:
                            logger.warning(f"Re-extraction returned no data for {element_id}")
                    except Exception as e:
                        logger.error(f"❌ Error re-extracting image from PDF: {e}")
                else:
                    logger.warning(f"PDF file not found: {pdf_path}")
            
            # If all else fails, return detailed error information
            error_response = {
                'error': 'Image data not available',
                'details': 'Image was processed but binary data is not accessible',
                'debug_info': {
                    'element_id': element_id,
                    'manual_name': manual_name,
                    'page_number': page_number,
                    'has_stored_data': 'image_base64' in metadata,
                    'pdf_path_found': find_pdf_file(manual_name) if manual_name else None,
                    'metadata_keys': list(metadata.keys())
                },
                'recommendations': [
                    'Upload the PDF again to store image data properly',
                    'Check if the original PDF file is still available',
                    'Verify image extraction settings are enabled'
                ]
            }
            
            logger.error(f"❌ All image serving methods failed for {element_id}: {error_response}")
            return jsonify(error_response), 404
            
        except Exception as e:
            logger.error(f"❌ Error accessing figure collection for {element_id}: {e}")
            return jsonify({'error': f'Database error: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"❌ Critical error serving image for {element_id}: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def find_pdf_file(manual_name):
    """Try to find the original PDF file for re-extraction"""
    # Look in common upload locations
    possible_paths = [
        os.path.join(web_downloads_path, manual_name),
        os.path.join('/tmp', manual_name),
        manual_name  # If it's an absolute path
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def re_extract_image_from_pdf(pdf_path, page_number, element_id):
    """Re-extract image from PDF for serving"""
    try:
        import fitz  # PyMuPDF
        pdf_doc = fitz.open(pdf_path)
        
        if page_number < len(pdf_doc):
            page = pdf_doc[page_number]
            image_list = page.get_images()
            
            # Try to find the specific image (this is approximate)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(pdf_doc, xref)
                
                if pix.n - pix.alpha < 4:  # Ensure it's not CMYK
                    img_data = pix.tobytes("png")
                    pix = None
                    pdf_doc.close()
                    return img_data
        
        pdf_doc.close()
        return None
        
    except Exception as e:
        logger.error(f"Error re-extracting image: {e}")
        return None

@app.route('/api/table/<element_id>')
def get_table_data(element_id):
    """Get formatted table data for a table element"""
    try:
        if not multimodal_processor:
            return jsonify({'error': 'Multimodal processor not available'}), 404
        
        # Search table collection for the element
        if not multimodal_processor.table_collection:
            return jsonify({'error': 'Table collection not available'}), 404
        
        try:
            results = multimodal_processor.table_collection.get(
                where={"element_id": element_id},
                include=['documents', 'metadatas']
            )
            
            if not results['documents']:
                return jsonify({'error': 'Table not found'}), 404
            
            metadata = results['metadatas'][0]
            
            return jsonify({
                'element_id': element_id,
                'content': results['documents'][0],
                'table_html': metadata.get('table_html', ''),
                'table_csv': metadata.get('table_csv', ''),
                'table_rows': metadata.get('table_rows', 0),
                'table_cols': metadata.get('table_cols', 0),
                'table_summary': metadata.get('table_summary', ''),
                'metadata': metadata
            })
            
        except Exception as e:
            logger.error(f"Error accessing table collection: {e}")
            return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        logger.error(f"Error serving table data for {element_id}: {e}")
        return jsonify({'error': str(e)}), 500

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
    
    if not chroma_client:
        config_warnings.append("ChromaDB initialization failed")
    
    if not agent_pipeline:
        config_warnings.append("Agent pipeline initialization failed")
    
    if config_warnings:
        print("\n" + "="*70)
        print("⚠️  CONFIGURATION WARNINGS:")
        for warning in config_warnings:
            print(f"   • {warning}")
        print("Please check your configuration for missing components.")
        print("="*70 + "\n")
    
    # Display startup information
    print(f"🚀 Starting Enhanced Multimodal RAG Service Manual Assistant...")
    print(f"📊 System Status:")
    print(f"   🧠 Agent Pipeline: {'✅ Ready' if agent_pipeline else '❌ Failed'}")
    print(f"   📚 ChromaDB: {'✅ Connected' if chroma_client else '❌ Failed'}")
    print(f"   🔧 Multimodal Processor: {'✅ Ready' if multimodal_processor else '❌ Failed'}")
    
    # Show optimization status
    optimized_available = multimodal_processor and hasattr(multimodal_processor, 'optimized_processor') and multimodal_processor.optimized_processor
    print(f"   ⚡ Batch Optimization: {'✅ Enabled' if optimized_available else '❌ Disabled'}")
    
    if multimodal_processor and multimodal_processor.collections:
        print(f"   📄 Text elements: {multimodal_processor.text_collection.count() if multimodal_processor.text_collection else 0}")
        print(f"   📊 Tables: {multimodal_processor.table_collection.count() if multimodal_processor.table_collection else 0}")
        print(f"   🖼️  Figures: {multimodal_processor.figure_collection.count() if multimodal_processor.figure_collection else 0}")
    
    print(f"   📚 Legacy chunks: {legacy_collection.count() if legacy_collection else 0}")
    print(f"🌐 Server starting at: http://localhost:5000")
    print(f"📁 Downloads saved to: {web_downloads_path}")
    print(f"📝 Logs saved to: logs/app.log")
    print(f"🤖 Agent system: {'Enabled' if agent_pipeline else 'Disabled'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
