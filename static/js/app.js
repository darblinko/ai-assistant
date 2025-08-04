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
        const elementId = result.metadata?.element_id;
        const contentType = result.metadata?.element_type || 'text';
        
        // Make chunk clickable if it has an element_id
        if (elementId) {
            chunkDiv.className = 'chunk clickable';
            chunkDiv.style.cursor = 'pointer';
            chunkDiv.addEventListener('click', () => {
                handleChunkClick(elementId, contentType, result);
            });
        } else {
            chunkDiv.className = 'chunk';
        }
        
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

// ============================================================================
// CONTENT PANEL FUNCTIONALITY - Missing piece for interactive chunks
// ============================================================================

// Handle chunk clicks to show detailed content
async function handleChunkClick(elementId, contentType, result) {
    console.log(`📱 Opening content panel for ${elementId} (${contentType})`);
    
    // Show loading state
    showContentPanel({
        elementId: elementId,
        contentType: contentType,
        loading: true
    });
    
    try {
        // Fetch detailed content from backend
        const contentData = await fetchContentDetails(elementId);
        
        // Merge with existing result data
        const enrichedData = {
            ...contentData,
            similarity: result.similarity,
            originalContent: result.content
        };
        
        // Display in content panel
        showContentPanel(enrichedData);
        
    } catch (error) {
        console.error('Error fetching content details:', error);
        showContentPanel({
            elementId: elementId,
            contentType: contentType,
            error: `Failed to load content: ${error.message}`
        });
    }
}

// Fetch detailed content from backend APIs
async function fetchContentDetails(elementId) {
    const response = await fetch(`/api/content/${elementId}`);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return await response.json();
}

// Create and show the content panel
function showContentPanel(data) {
    // Remove existing panel if any
    removeContentPanel();
    
    // Create panel elements
    const overlay = document.createElement('div');
    overlay.className = 'panel-overlay';
    overlay.id = 'contentPanelOverlay';
    
    const panel = document.createElement('div');
    panel.className = 'content-panel';
    panel.id = 'contentPanel';
    
    // Panel header with normalized content type
    const rawContentType = data.contentType || data.content_type;
    const normalizedContentType = normalizeContentType(rawContentType);
    
    const header = document.createElement('div');
    header.className = 'content-panel-header';
    header.innerHTML = `
        <div class="content-panel-title">
            ${getContentTypeLabel(normalizedContentType)} Details
        </div>
        <button class="panel-close-btn" onclick="closeContentPanel()" aria-label="Close panel">
            ×
        </button>
    `;
    
    // Panel body
    const body = document.createElement('div');
    body.className = 'content-panel-body';
    
    if (data.loading) {
        body.innerHTML = '<div class="panel-loading">Loading content...</div>';
    } else if (data.error) {
        body.innerHTML = `<div class="panel-error">${data.error}</div>`;
    } else {
        body.innerHTML = generatePanelContent(data);
    }
    
    // Assemble panel
    panel.appendChild(header);
    panel.appendChild(body);
    document.body.appendChild(overlay);
    document.body.appendChild(panel);
    
    // Show with animation
    requestAnimationFrame(() => {
        overlay.classList.add('active');
        panel.classList.add('open');
    });
    
    // Close on overlay click
    overlay.addEventListener('click', closeContentPanel);
    
    // Handle escape key
    document.addEventListener('keydown', handlePanelEscape);
}

// Normalize content type from backend to standard types
function normalizeContentType(contentType) {
    const typeMap = {
        'figures': 'figure',
        'figure': 'figure',
        'diagram': 'figure',
        'tables': 'table',
        'table': 'table',
        'text': 'text',
        'heading': 'text'
    };
    return typeMap[contentType] || 'text';
}

// Generate content for the panel based on content type
function generatePanelContent(data) {
    const rawContentType = data.content_type || data.contentType;
    const contentType = normalizeContentType(rawContentType);
    
    console.log(`🎨 Generating panel content: ${rawContentType} -> ${contentType}`, data);
    
    let html = '';
    
    // Metadata section
    html += `
        <div class="content-detail-section">
            <h3>📋 Metadata</h3>
            <div class="content-metadata">
                <div class="metadata-item">
                    <span class="metadata-label">Element ID:</span>
                    <span class="metadata-value">${data.element_id}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Content Type:</span>
                    <span class="metadata-value">${contentType}</span>
                </div>
                ${data.metadata?.page_number ? `
                <div class="metadata-item">
                    <span class="metadata-label">Page:</span>
                    <span class="metadata-value">${data.metadata.page_number}</span>
                </div>` : ''}
                ${data.similarity ? `
                <div class="metadata-item">
                    <span class="metadata-label">Similarity:</span>
                    <span class="metadata-value">${(data.similarity * 100).toFixed(1)}%</span>
                </div>` : ''}
                ${data.metadata?.manual_name ? `
                <div class="metadata-item">
                    <span class="metadata-label">Manual:</span>
                    <span class="metadata-value">${data.metadata.manual_name}</span>
                </div>` : ''}
            </div>
        </div>
    `;
    
    // Content-specific sections based on normalized type
    if (contentType === 'table') {
        console.log('📊 Generating table content');
        html += generateTableContent(data);
    } else if (contentType === 'figure') {
        console.log('🖼️ Generating image content');
        html += generateImageContent(data);
    } else {
        console.log('📄 Generating text content');
        html += generateTextContent(data);
    }
    
    return html;
}

// Generate table-specific content
function generateTableContent(data) {
    let html = `
        <div class="content-detail-section">
            <h3>📊 Table Data</h3>
    `;
    
    // Table summary if available
    if (data.table_summary) {
        html += `<div class="table-summary">${data.table_summary}</div>`;
    }
    
    // HTML table if available
    if (data.table_html) {
        html += `
            <div class="content-table-display">
                ${data.table_html}
            </div>
        `;
    }
    
    // Table statistics
    if (data.table_rows || data.table_cols) {
        html += `
            <div class="content-metadata" style="margin-top: 15px;">
                ${data.table_rows ? `
                <div class="metadata-item">
                    <span class="metadata-label">Rows:</span>
                    <span class="metadata-value">${data.table_rows}</span>
                </div>` : ''}
                ${data.table_cols ? `
                <div class="metadata-item">
                    <span class="metadata-label">Columns:</span>
                    <span class="metadata-value">${data.table_cols}</span>
                </div>` : ''}
            </div>
        `;
    }
    
    // Download options
    if (data.table_csv) {
        html += `
            <div style="margin-top: 15px;">
                <button onclick="downloadTableCSV('${data.element_id}')" 
                        style="background: #38a169; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    📥 Download as CSV
                </button>
            </div>
        `;
    }
    
    html += '</div>';
    
    // Raw content section
    html += `
        <div class="content-detail-section">
            <h3>📄 Full Content</h3>
            <div class="content-full-text">${data.content || 'No content available'}</div>
        </div>
    `;
    
    return html;
}

// Generate image-specific content
function generateImageContent(data) {
    let html = `
        <div class="content-detail-section">
            <h3>🖼️ Image Content</h3>
    `;
    
    // Try to load the actual image
    html += `
        <div class="content-image-display">
            <div id="imageContainer-${data.element_id}">
                <div class="content-image-placeholder">
                    📷 Loading image...
                </div>
            </div>
        </div>
    `;
    
    // Image metadata
    if (data.metadata?.image_width || data.metadata?.image_height) {
        html += `
            <div class="image-metadata">
                <strong>Image Properties:</strong><br>
                ${data.metadata.image_width ? `Width: ${data.metadata.image_width}px<br>` : ''}
                ${data.metadata.image_height ? `Height: ${data.metadata.image_height}px<br>` : ''}
                ${data.metadata.has_ocr_text ? 'Contains text labels' : 'Visual content only'}
            </div>
        `;
    }
    
    // OCR text if available
    if (data.metadata?.ocr_text) {
        html += `
            <div class="ocr-text-section">
                <h4>📝 OCR Text Found:</h4>
                <div class="ocr-text">${data.metadata.ocr_text}</div>
            </div>
        `;
    }
    
    // Description if available
    if (data.metadata?.description) {
        html += `
            <div style="margin-top: 15px;">
                <h4>🔍 AI Description:</h4>
                <div class="description-text">${data.metadata.description}</div>
            </div>
        `;
    }
    
    html += '</div>';
    
    // Full content section
    html += `
        <div class="content-detail-section">
            <h3>📄 Full Content</h3>
            <div class="content-full-text">${data.content || 'No content available'}</div>
        </div>
    `;
    
    // Attempt to load the image after panel is shown
    setTimeout(() => {
        loadImageForPanel(data.element_id);
    }, 100);
    
    return html;
}

// Generate text content
function generateTextContent(data) {
    return `
        <div class="content-detail-section">
            <h3>📄 Full Content</h3>
            <div class="content-full-text">${data.content || 'No content available'}</div>
        </div>
    `;
}

// Load image for the panel with enhanced error handling
async function loadImageForPanel(elementId) {
    const container = document.getElementById(`imageContainer-${elementId}`);
    if (!container) {
        console.error(`Image container not found: imageContainer-${elementId}`);
        return;
    }
    
    console.log(`🖼️ Loading image for element: ${elementId}`);
    
    // Show loading state
    container.innerHTML = `
        <div class="content-image-placeholder">
            ⏳ Loading image...<br>
            <small style="color: #6c757d;">Fetching from server...</small>
        </div>
    `;
    
    try {
        const response = await fetch(`/api/image/${elementId}`);
        console.log(`📡 Image API response for ${elementId}:`, response.status, response.statusText);
        
        if (response.ok) {
            const contentType = response.headers.get('content-type');
            console.log(`✅ Image loaded successfully: ${contentType}`);
            
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            
            container.innerHTML = `
                <img src="${imageUrl}" alt="Technical diagram for ${elementId}" 
                     style="max-width: 100%; height: auto; border: 1px solid #dee2e6; border-radius: 8px; cursor: pointer;"
                     onclick="openImageFullscreen('${imageUrl}')"
                     onload="console.log('🖼️ Image rendered successfully: ${elementId}')">
                <div style="text-align: center; margin-top: 10px; font-size: 12px; color: #6c757d;">
                    🔍 Click image to view full size
                </div>
            `;
        } else {
            // Get error details from server
            console.warn(`⚠️ Image not available: ${response.status}`);
            
            let errorData;
            try {
                errorData = await response.json();
            } catch (e) {
                errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
            }
            
            container.innerHTML = `
                <div class="content-image-placeholder">
                    📷 Image not available<br>
                    <small style="color: #6c757d;">
                        ${errorData.error || 'Image data not found'}<br>
                        Status: ${response.status}
                    </small>
                    ${errorData.details ? `<br><small style="color: #999;">${errorData.details}</small>` : ''}
                </div>
            `;
        }
    } catch (error) {
        console.error(`❌ Image loading failed for ${elementId}:`, error);
        container.innerHTML = `
            <div class="content-image-placeholder">
                ❌ Failed to load image<br>
                <small style="color: #e53e3e;">
                    ${error.message}<br>
                    Element: ${elementId}
                </small>
            </div>
        `;
    }
}

// Open image in fullscreen
function openImageFullscreen(imageUrl) {
    const fullscreenDiv = document.createElement('div');
    fullscreenDiv.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.9); z-index: 2000; display: flex;
        justify-content: center; align-items: center; cursor: pointer;
    `;
    
    const img = document.createElement('img');
    img.src = imageUrl;
    img.style.cssText = 'max-width: 95%; max-height: 95%; object-fit: contain;';
    
    fullscreenDiv.appendChild(img);
    document.body.appendChild(fullscreenDiv);
    
    fullscreenDiv.addEventListener('click', () => {
        document.body.removeChild(fullscreenDiv);
        URL.revokeObjectURL(imageUrl);
    });
}

// Close content panel
function closeContentPanel() {
    const panel = document.getElementById('contentPanel');
    const overlay = document.getElementById('contentPanelOverlay');
    
    if (panel) {
        panel.classList.remove('open');
        setTimeout(() => {
            document.body.removeChild(panel);
        }, 300);
    }
    
    if (overlay) {
        overlay.classList.remove('active');
        setTimeout(() => {
            document.body.removeChild(overlay);
        }, 300);
    }
    
    document.removeEventListener('keydown', handlePanelEscape);
}

// Remove any existing panels
function removeContentPanel() {
    const existingPanel = document.getElementById('contentPanel');
    const existingOverlay = document.getElementById('contentPanelOverlay');
    
    if (existingPanel) document.body.removeChild(existingPanel);
    if (existingOverlay) document.body.removeChild(existingOverlay);
}

// Handle escape key for panel
function handlePanelEscape(event) {
    if (event.key === 'Escape') {
        closeContentPanel();
    }
}

// Download table as CSV
function downloadTableCSV(elementId) {
    fetch(`/api/table/${elementId}`)
        .then(response => response.json())
        .then(data => {
            if (data.table_csv) {
                const blob = new Blob([data.table_csv], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `table_${elementId}.csv`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
        })
        .catch(error => {
            console.error('Error downloading CSV:', error);
            alert('Failed to download CSV file');
        });
}

// Make functions globally available
window.closeContentPanel = closeContentPanel;
window.downloadTableCSV = downloadTableCSV;
window.openImageFullscreen = openImageFullscreen;

console.log('🎨 Multimodal RAG Assistant UI fully loaded with interactive content panels');
