# Multimodal RAG Assistant: Comprehensive Agentic Workflow Analysis

Based on my deep analysis of the codebase, here's a detailed breakdown of this sophisticated multi-agent system designed for technical service manual assistance:

## 🏗️ **System Architecture Overview**

This is a **multi-agent RAG (Retrieval-Augmented Generation) system** that combines intelligent query processing, multimodal content understanding, and external resource discovery to provide expert-level technical assistance for service manuals.

### **Core Philosophy: "Intelligent Specialization with Orchestrated Collaboration"**
- Each agent has a specific domain of expertise with clearly defined responsibilities
- Agents collaborate through a centralized pipeline with standardized data models
- The system adaptively routes queries to the most appropriate processing strategy
- Multimodal content (text, tables, diagrams) is treated as first-class citizens with specialized processing

## 🤖 **Multi-Agent Architecture**

### **1. Agent Pipeline (Central Orchestrator)**
**File:** `agents/agent_pipeline.py`
**Role:** Master coordinator and intelligent query router

**Key Responsibilities:**
- **Query Validation:** Ensures incoming queries meet processing criteria
- **Multi-Stage Processing:** Orchestrates a 3-step pipeline (Intent → Equipment → Strategy)
- **Agent Coordination:** Routes work to appropriate specialized agents
- **Performance Monitoring:** Tracks processing time and success rates
- **Confidence Aggregation:** Combines confidence scores from multiple agents

**Pipeline Flow:**
```
User Query → Intent Router → Equipment Classifier → Search Strategy → Results
```

**Strategic Value:** Provides intelligent preprocessing that dramatically improves downstream agent performance by giving them rich context and clear objectives.

### **2. Intent Router Agent**
**Role:** Query understanding and intent classification

**Intent Categories (from models):**
- `manual_search_needed`: User needs a specific manual
- `troubleshooting`: Diagnostic/repair guidance  
- `specifications`: Technical specs/measurements
- `diagram_request`: Visual/schematic information
- `maintenance`: Routine maintenance procedures
- `installation`: Installation guidance
- `parts_diagram`: Exploded views and part identification

**Strategic Approach:** Uses semantic analysis to understand not just what the user is asking, but what they're trying to accomplish, enabling proactive assistance.

### **3. Equipment Classifier Agent**
**Role:** Equipment identification and contextual understanding

**Equipment Categories:**
- **Automotive:** Cars, trucks, motorcycles (with year/make/model extraction)
- **HVAC:** Heating, cooling, refrigeration systems
- **Appliance:** Household/commercial appliances
- **Electronics:** Circuit boards, consumer electronics
- **Industrial:** Heavy machinery, manufacturing equipment

**Entity Extraction Capabilities:**
- Years, brands, models, components, part numbers
- Alternative interpretations for ambiguous queries
- Confidence scoring for equipment matches

**Strategic Value:** Prevents cross-contamination between equipment types and enables precise manual discovery.

### **4. Web Search Agent**
**File:** `agents/web_search_agent.py`
**Role:** External resource discovery and acquisition

**Multi-Source Search Strategy:**
- **Google Custom Search:** PDF manuals with targeted queries
- **Reddit Forums:** Community discussions and troubleshooting
- **YouTube:** Video tutorials and demonstrations
- **Manufacturer Sites:** Official documentation

**Advanced Capabilities:**
- **Model-Specific Targeting:** Uses regex to extract exact model names (e.g., "True T-49-HC")
- **Content Validation:** Ensures downloaded PDFs match requested equipment
- **Multi-Format Processing:** PDFs, forum posts, video descriptions
- **Source Reliability Scoring:** Manufacturer sites > Forums > General web

**Key Innovation:** **Direct Model Matching** - Instead of complex filtering, uses simple regex patterns to extract exact model names and searches specifically for those models, dramatically improving relevance.

## 🧠 **Intelligent Query Processing Workflow**

### **Phase 1: Query Analysis & Validation**
```
Raw Query → Validation → Intent Classification → Confidence Scoring
```

**Validation Rules:**
- Minimum 5 characters, maximum 1000 characters
- Must contain meaningful content (not just numbers)
- Provides recommendations for query improvement

### **Phase 2: Equipment Intelligence**
```
Validated Query → Entity Extraction → Category Classification → Confidence Assessment
```

**Entity Extraction:**
- **Years:** "1998", "late 1990s" → standardized format
- **Brands:** "Toyota", "Honda" → normalized brand names
- **Models:** "Supra", "Civic" → exact model identification
- **Components:** "alternator", "compressor" → component mapping

### **Phase 3: Search Strategy Formation**
```
Intent + Equipment + Context → Search Strategy → Resource Prioritization
```

**Search Scopes:**
- `local_first`: Check internal database, then external if needed
- `external_preferred`: Prioritize web search for latest information
- `hybrid`: Intelligently combine local and external results
- `external_only`: Only search external sources

## 🎯 **Multimodal Content Processing Strategy**

### **Content Type Specialization**

#### **Text Processing:**
- **Structure-Aware:** Preserves document hierarchy (headings, sections, procedures)
- **Semantic Chunking:** Intelligent text segmentation maintaining context
- **Technical Language:** Optimized for technical terminology and step-by-step procedures
- **Cross-References:** Maintains links between related sections

#### **Table Processing:**
- **Structure Preservation:** Maintains relationships between data points
- **Semantic Summarization:** AI-generated descriptions of table contents
- **Multiple Formats:** HTML display, CSV download, searchable text
- **Context Integration:** Tables linked to surrounding explanatory text

#### **Image/Diagram Processing:**
- **Dual Analysis:** OCR for text extraction + Vision API for visual understanding
- **Technical Diagram Recognition:** Specialized for schematics, wiring diagrams, exploded views
- **Scale-Aware Processing:** Intelligent image resizing for optimal OCR
- **Metadata Enrichment:** Comprehensive tagging for searchability

### **Three-Collection Vector Database Architecture**
```
service_manual_text     →  Text content with semantic embeddings
service_manual_tables   →  Structured data with relationship embeddings  
service_manual_figures  →  Visual content with multimodal embeddings
```

**Strategic Benefits:**
- **Type-Specific Optimization:** Each collection optimized for its content type
- **Efficient Retrieval:** Targeted search within appropriate content domains
- **Scalable Architecture:** Independent scaling and tuning per content type

## 🔄 **Problem-Solving Evolution**

### **Original Problems (Solved):**
1. **ChromaDB Collection Failures:** Collections weren't being initialized properly
2. **Cross-Equipment Contamination:** Automotive queries matching refrigeration content
3. **Random PDF Downloads:** Web search downloading irrelevant manuals

### **Solution Strategy:**
```
Before: Query → Generic Search → Wrong Matches → LLM Confusion
After: Query → Agent Analysis → Targeted Search → Relevant Results → Expert Response
```

**Example Transformation:**
```
Query: "how do i replace the alternator on my 2019 Infiniti Q50?"

OLD SYSTEM:
→ Generic embeddings search
→ Matches refrigeration content (high similarity due to technical terms)
→ Falls back to general knowledge

NEW SYSTEM:
→ Intent Router: "equipment_specific" + "troubleshooting"
→ Equipment Classifier: automotive, Infiniti, Q50, alternator, replacement
→ Search Strategy: automotive-focused, diagrams priority, specific keywords
→ Enhanced Search: filters out refrigeration content, finds automotive data
→ Accurate automotive repair guidance
```

## 📊 **Type-Safe Data Architecture**

### **Pydantic Models for Reliability**
**File:** `models/agent_models.py`

**Key Model Categories:**
- **Intent Models:** `IntentAnalysis`, `QueryComplexity`
- **Equipment Models:** `EquipmentAnalysis`, `ExtractedEntities`
- **Search Models:** `SearchStrategy`, `SearchMetadata`
- **Pipeline Models:** `PipelineResult`, `AgentStepSummary`
- **Configuration Models:** `AgentConfig`, `PipelineConfig`

**Strategic Value:** 
- **Type Safety:** Prevents runtime errors with comprehensive validation
- **API Consistency:** Standardized data exchange between agents
- **Configuration Management:** Centralized, validated configuration system
- **Monitoring & Debugging:** Rich metadata for system observability

## 🎨 **User Experience Strategy**

### **Progressive Disclosure Interface**
- **Summary Level:** Quick overview with processing information
- **Detail Level:** Expandable content cards with full context
- **Raw Data Level:** Direct access to source documents

### **Interactive Content Exploration**
- **Clickable Context Cards:** Each retrieved chunk is explorable
- **Content Type Indicators:** Visual cues for text, tables, figures
- **Cross-Reference Navigation:** Jump between related content pieces
- **Source Attribution:** Clear tracking of information sources

### **Real-Time Processing Feedback**
```
Query Received → Intent Analysis → Equipment Detection → Search Execution → Results Assembly
```

**User Visibility:**
- Step-by-step processing updates like: "🎯 Intent: equipment_specific (95% confidence) | 🔧 Equipment: automotive - Infiniti Q50 | ⚡ Processed in 1.2s using 2 agents"
- Confidence indicators for each stage
- Performance metrics (processing time, agents used)
- Error handling with graceful degradation

## 🛠️ **Technical Implementation Highlights**

### **Performance Optimizations**
- **Async Processing:** Non-blocking external searches
- **Batch Embedding:** Reduces API costs and latency
- **Intelligent Caching:** Prevents redundant processing with configurable TTL
- **Connection Pooling:** Efficient API client management

### **Reliability & Resilience**
- **Graceful Degradation:** System continues with partial agent failures
- **Comprehensive Error Handling:** Each agent has isolated error boundaries
- **Fallback Strategies:** Multiple approaches for each processing step
- **Resource Management:** Automatic cleanup and memory management

### **Modern Development Practices**
- **UV Package Management:** Fast, reliable dependency management
- **Type Safety:** Comprehensive Pydantic validation
- **Comprehensive Testing:** Test suite covering all components
- **Detailed Logging:** Step-by-step processing traces

## 🎯 **Strategic Advantages**

### **1. Domain-Specific Intelligence**
Unlike generic RAG systems, this is purpose-built for technical service manuals with deep understanding of:
- Technical document structure and conventions
- Equipment-specific terminology and relationships
- Visual technical content interpretation
- Troubleshooting and maintenance workflows

### **2. Proactive Resource Discovery**
- **Intelligent Manual Finding:** Automatically discovers missing manuals based on equipment identification
- **Multi-Source Aggregation:** Combines manufacturer docs, community knowledge, and video content
- **Quality-Assured Downloads:** Validates content relevance before integration

### **3. Multimodal Content Mastery**
- **Beyond Text Understanding:** Comprehensive processing of tables, diagrams, and images
- **Context Preservation:** Maintains relationships between different content types
- **Technical Visual Analysis:** Specialized understanding of schematics and technical drawings

### **4. Adaptive Processing Intelligence**
- **Context-Aware Strategy:** Adjusts processing approach based on query characteristics
- **Confidence-Based Decision Making:** Uses uncertainty to guide processing decisions
- **Equipment-Specific Optimization:** Tailors search and processing for equipment categories

### **5. Enterprise-Grade Reliability**
- **Type-Safe Architecture:** Comprehensive data validation prevents runtime errors
- **Comprehensive Monitoring:** Full observability into system performance
- **Scalable Design:** Modular architecture supports adding new agents and capabilities

## 🚀 **System Capabilities Summary**

This agentic workflow represents a sophisticated application of multi-agent AI to a specific technical domain, combining:

- **Intelligent Query Understanding** (Intent Router + Equipment Classifier)
- **Adaptive Search Strategy** (Pipeline orchestration with confidence-based decision making)  
- **Multimodal Content Processing** (Text, tables, diagrams with specialized pipelines)
- **Proactive Resource Discovery** (Web search with equipment-aware filtering)
- **Interactive Knowledge Exploration** (Progressive disclosure with cross-references)
- **Enterprise Reliability** (Type safety, monitoring, error handling)

The system goes far beyond simple retrieval to provide **expert-level technical assistance** that understands context, discovers missing information, and presents complex technical content in an accessible, explorable format.

This represents a practical implementation of agentic AI that demonstrates how multiple specialized agents can collaborate to solve complex, domain-specific problems that would be challenging for any single AI system to handle effectively.

The key insight is that by having each agent focus on what it does best (intent understanding, equipment classification, resource discovery, content processing), the overall system achieves much higher accuracy and user satisfaction than a monolithic approach.

////////  DESIGN OVERVIEW FOR MULTIMODAL EMBEDDING, STORAGE AND RETRIEVAL
# 📊🖼️ **Detailed Multimodal Content Processing Pipeline**

## **Complete Flow: Tables and Images - From PDF to Interactive Chat UI**

Based on my analysis of the codebase, here's the comprehensive breakdown of how tables and images are interpreted, embedded, saved, and retrieved in your multimodal RAG system:

---

## 🏗️ **PHASE 1: Content Extraction & Interpretation**

### **📊 Table Processing Pipeline**

#### **1.1 Table Detection & Extraction**
```python
# Uses pdfplumber for precise table extraction
with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[page_num]
    tables = page.extract_tables()  # Detects table structures
```

#### **1.2 Structure Preservation**
```python
# Convert to pandas DataFrame maintaining relationships
df = pd.DataFrame(table[1:], columns=table[0])  # Headers + data
df = df.dropna(how='all').fillna('')  # Clean empty rows/cells

# Multiple format creation:
linearized_text = self._linearize_table(df)      # For embedding/search
html_format = df.to_html(classes="table table-striped")  # For display
csv_format = df.to_csv(index=False)              # For download
```

#### **1.3 AI-Enhanced Table Understanding**
```python
# Generate semantic summary using LLM
table_summary = self._generate_table_summary(df, manual_name)

# Example LLM prompt:
"""Analyze this table from a service manual:
Table data: [first 3 rows shown]
Provide a 1-2 sentence summary describing:
1. What type of data this table contains
2. Key specifications, measurements, or parameters shown
3. Any units of measurement"""
```

### **🖼️ Image/Diagram Processing Pipeline**

#### **2.1 Enhanced Image Extraction**
```python
# DUAL APPROACH: Individual images + Full-page diagrams

# Method 1: Individual images from PDF
image_list = page.get_images()

# Method 2: Full-page technical diagrams
if len(visual_elements) > 2:  # Likely a diagram page
    # Create high-resolution full-page image
    target_width = min(1024, int(page_width * 2))  # Optimal resolution
    mat = fitz.Matrix(scaling_factor)
    full_page_pix = page.get_pixmap(matrix=mat)
```

#### **2.2 Intelligent Image Enhancement**
```python
# Smart scaling for optimal processing
if original_width < 200:
    # Scale up small images for better OCR
    scale_factor = max(2.0, 300 / max(original_width, original_height))
    enhanced_image = image.resize((new_width, new_height), Image.LANCZOS)
elif original_width > 1024:
    # Scale down large images for API efficiency
    image.thumbnail((1024, 1024), Image.LANCZOS)
```

#### **2.3 Multimodal Analysis**
```python
# DUAL CONTENT EXTRACTION:

# OCR Text Extraction
ocr_text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')

# AI Vision Analysis  
image_description = openai_client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this technical diagram..."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]
    }]
)

# Combined content for embedding
combined_text = f"{image_description}\n\nOCR Text: {ocr_text}"
```

---

## 🧠 **PHASE 2: Embedding Generation & Storage**

### **📈 Batch-Optimized Embedding Process**
```python
# INTELLIGENT BATCH PROCESSING (Cost Optimized)
texts_to_embed = []
for element in elements:
    if element.content:
        texts_to_embed.append(element.content)

# Batch embedding generation (much cheaper than individual calls)
embeddings = multimodal_processor.get_embeddings_batch(texts_to_embed)

# Cost savings: ~90% reduction vs individual API calls
```

### **🗄️ Three-Collection Storage Architecture**

#### **Collection 1: `service_manual_text`**
```python
# Text content with hierarchical metadata
collection.add(
    ids=[element.element_id],
    documents=[element.content],
    embeddings=[text_embedding],
    metadatas=[{
        "element_type": "text|heading",
        "page_number": page_num,
        "manual_name": manual_name,
        "avg_font_size": font_size,
        "is_heading": True/False
    }]
)
```

#### **Collection 2: `service_manual_tables`**
```python
# Structured table data with multiple formats
collection.add(
    ids=[table_element_id],
    documents=[f"{table_summary}\n\n{linearized_text}"],
    embeddings=[table_embedding],
    metadatas=[{
        "element_type": "table",
        "table_rows": len(df),
        "table_cols": len(df.columns),
        "table_summary": ai_generated_summary,
        "table_html": df.to_html(),        # For rich display
        "table_csv": df.to_csv(),          # For download
        "manual_name": manual_name
    }]
)
```

#### **Collection 3: `service_manual_figures`**
```python
# Visual content with multimodal metadata
collection.add(
    ids=[figure_element_id],
    documents=[f"{vision_description}\n\nOCR: {ocr_text}"],
    embeddings=[image_embedding],
    metadatas=[{
        "element_type": "figure|diagram",
        "image_width": width,
        "image_height": height,
        "has_ocr_text": len(ocr_text) > 0,
        "ocr_text": extracted_text,
        "description": vision_analysis,
        "image_filename": "element_id.png",       # File reference
        "image_file_path": "/path/to/image.png",  # Full path
        "processing_method": "enhanced_full_page|individual"
    }]
)

# Images saved as files (not in database)
with open(f"static/images/{element_id}.png", 'wb') as f:
    f.write(image_data)
```

---

## 🔍 **PHASE 3: Intelligent Retrieval**

### **🎯 Equipment-Aware Search**
```python
# Enhanced search with agent intelligence
search_strategy = {
    "equipment_category": "automotive",  # From agent analysis
    "content_type_priority": ["diagrams", "tables", "text"],
    "confidence_threshold": 0.7,
    "cross_equipment_filtering": True
}

results = multimodal_processor.intelligent_search(query, search_strategy)
```

### **📊 Multi-Collection Query**
```python
# Search appropriate collections based on content type priority
for content_type in ["diagrams", "tables", "text"]:
    if content_type == "tables":
        collection_results = table_collection.query(
            query_embeddings=[enhanced_query_embedding],
            n_results=3,
            include=["documents", "distances", "metadatas"]
        )
    # Similar for other collections...
```

---

## 🖥️ **PHASE 4: Frontend Display System**

### **🎨 Interactive Content Cards**

#### **JavaScript Content Rendering**
```javascript
// Handle clickable content chunks
function handleChunkClick(chunkData) {
    const elementId = chunkData.metadata.element_id;
    const elementType = chunkData.element_type;
    
    // Create dynamic content panel based on type
    if (elementType === 'table') {
        showTableContent(elementId);
    } else if (elementType === 'figure' || elementType === 'diagram') {
        showImageContent(elementId);
    } else {
        showTextContent(elementId);
    }
}
```

#### **📊 Table Display Implementation**
```javascript
async function showTableContent(elementId) {
    // Fetch rich table data
    const response = await fetch(`/api/table/${elementId}`);
    const tableData = await response.json();
    
    // Create interactive table display
    const contentHtml = `
        <div class="table-container">
            <div class="table-header">
                <h4>📊 ${tableData.table_summary}</h4>
                <div class="table-stats">
                    ${tableData.table_rows} rows × ${tableData.table_cols} columns
                </div>
            </div>
            
            <!-- Rich HTML table display -->
            <div class="table-wrapper">
                ${tableData.table_html}
            </div>
            
            <!-- Download options -->
            <div class="table-actions">
                <button onclick="downloadTableCSV('${elementId}')">
                    📥 Download CSV
                </button>
                <button onclick="copyTableData('${elementId}')">
                    📋 Copy Data
                </button>
            </div>
        </div>
    `;
    
    showContentPanel(contentHtml);
}

function downloadTableCSV(elementId) {
    // Create and trigger CSV download
    const csvContent = tableData.table_csv;
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${elementId}_table.csv`;
    a.click();
}
```

#### **🖼️ Image Display Implementation**
```javascript
async function showImageContent(elementId) {
    // Fetch image metadata
    const response = await fetch(`/api/content/${elementId}`);
    const imageData = await response.json();
    
    // Create rich image display
    const contentHtml = `
        <div class="image-container">
            <div class="image-header">
                <h4>🖼️ ${imageData.description}</h4>
                <div class="image-stats">
                    ${imageData.image_width} × ${imageData.image_height} pixels
                </div>
            </div>
            
            <!-- Zoomable image display -->
            <div class="image-wrapper">
                <img src="/api/image/${elementId}" 
                     alt="Technical diagram"
                     class="zoomable-image"
                     onclick="openImageModal('${elementId}')"
                     onerror="handleImageError(this)">
            </div>
            
            <!-- OCR text if available -->
            ${imageData.has_ocr_text ? `
                <div class="ocr-section">
                    <h5>🔤 Extracted Text:</h5>
                    <pre class="ocr-text">${imageData.ocr_text}</pre>
                </div>
            ` : ''}
            
            <!-- AI description -->
            <div class="description-section">
                <h5>🤖 AI Analysis:</h5>
                <p>${imageData.description}</p>
            </div>
        </div>
    `;
    
    showContentPanel(contentHtml);
}

function openImageModal(elementId) {
    // Full-screen image viewer with zoom controls
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <img src="/api/image/${elementId}" class="modal-image">
            <div class="zoom-controls">
                <button onclick="zoomIn()">🔍+</button>
                <button onclick="zoomOut()">🔍-</button>
                <button onclick="resetZoom()">↻</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}
```

### **🎯 Smart Content Indicators**
```javascript
// Visual content type indicators in chat
function formatRetrievedChunk(chunk) {
    const typeIcons = {
        'table': '📊',
        'figure': '🖼️',
        'diagram': '📐',
        'text': '📄'
    };
    
    const similarity = (chunk.similarity * 100).toFixed(0);
    
    return `
        <div class="content-chunk ${chunk.element_type}" 
             onclick="handleChunkClick(${JSON.stringify(chunk)})">
            <div class="chunk-header">
                ${typeIcons[chunk.element_type]} 
                <span class="content-type">${chunk.element_type.toUpperCase()}</span>
                <span class="similarity-score">${similarity}% match</span>
            </div>
            <div class="chunk-preview">
                ${chunk.content.substring(0, 150)}...
            </div>
            <div class="chunk-footer">
                📄 ${chunk.metadata.manual_name} • Page ${chunk.metadata.page_number}
                <span class="click-hint">Click to expand</span>
            </div>
        </div>
    `;
}
```

---

## 🔧 **PHASE 5: Advanced Features**

### **⚡ Performance Optimizations**
- **Batch Embedding**: ~90% cost reduction vs individual API calls
- **Image Caching**: Static file serving with cache headers
- **Smart Scaling**: Optimal image sizes for processing vs display
- **Progressive Loading**: Content panels load on-demand

### **🎨 Rich User Experience**
- **Content Type Indicators**: Visual cues for different content types
- **Interactive Tables**: Sortable, downloadable, copyable
- **Zoomable Images**: Full-screen modal with zoom controls
- **Context Preservation**: Maintains relationships between content types
- **Error Handling**: Graceful fallbacks when images aren't available

### **🔍 Search Intelligence**
- **Equipment-Aware Filtering**: Prevents automotive queries from matching HVAC content
- **Content Type Prioritization**: Searches diagrams first for visual queries
- **Cross-Reference Navigation**: Jump between related content pieces
- **Confidence Scoring**: Shows relevance percentage for each result

---

## 📱 **Complete User Journey Example**

### **Step 1: PDF Upload & Processing**
1. User uploads "Toyota Supra Service Manual.pdf"
2. System extracts 45 text blocks, 12 tables, 23 diagrams
3. Each table gets AI summary: "Torque specifications for engine components (Nm values)"
4. Each diagram gets OCR + Vision analysis: "Wiring diagram showing alternator connections with part numbers B+ terminal, ground points, and voltage regulator circuit"

### **Step 2: Query Processing**
```
User Query: "What's the alternator wiring for 1998 Supra?"
↓
Agent Pipeline: Intent=diagram_request, Equipment=automotive-toyota-supra-alternator
↓
Search Strategy: content_priority=["diagrams","tables","text"], automotive_focused=true
```

### **Step 3: Intelligent Retrieval**
```
Search Results:
📐 Diagram (95% match): "Alternator wiring schematic showing B+ terminal..."
📊 Table (87% match): "Electrical specifications: Alternator output 14.2V @ 2000RPM..."
📄 Text (82% match): "Remove negative battery cable before alternator service..."
```

### **Step 4: Interactive Chat Display**
User sees clickable cards in chat. Clicking the diagram card:
- Loads high-res image from `/api/image/supra_p45_alternator_diagram`
- Shows OCR text: "B+, IGN, F, L terminals"
- Displays AI description: "Wiring diagram for 3.0L 2JZ-GE alternator circuit"
- Offers full-screen zoom view

### **Step 5: Table Interaction**
Clicking table card shows:
- Rich HTML table with specifications
- Download CSV button
- Interactive sorting/filtering
- Source attribution

This comprehensive pipeline transforms static PDF content into an interactive, searchable, multimodal knowledge base where users can click on any retrieved content to explore detailed tables with download options and high-resolution technical diagrams with AI-powered descriptions.
