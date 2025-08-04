#!/usr/bin/env python3
"""
Web Search & Resource Discovery Agent
Handles external resource discovery, PDF downloads, and forum scraping
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse, quote_plus
import re
import json
import os
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import tempfile
import logging
from bs4 import BeautifulSoup
import pdfplumber
import fitz  # PyMuPDF
from openai import OpenAI

from .base_agent import BaseAgent, AgentResponse
from models.agent_models import SearchStrategy

logger = logging.getLogger(__name__)

@dataclass
class WebResource:
    """Represents a discovered web resource"""
    url: str
    title: str
    content_type: str  # 'pdf', 'forum_post', 'manual', 'video', 'article'
    source: str  # 'reddit', 'manufacturer', 'youtube', 'forum'
    relevance_score: float
    content_preview: str
    metadata: Dict[str, Any]
    download_path: Optional[str] = None
    extracted_content: Optional[str] = None

@dataclass
class SearchResult:
    """Search result with resources and processing info"""
    query: str
    resources: List[WebResource]
    search_time: float
    sources_searched: List[str]
    total_found: int
    success: bool
    error: Optional[str] = None

class WebSearchAgent(BaseAgent):
    """Agent for discovering and acquiring external resources"""
    
    def __init__(self, openai_client: OpenAI, config: Dict[str, Any]):
        super().__init__(openai_client, config)
        self.name = "WebSearchAgent"
        
        # Search configuration
        self.search_engines = config.get('search_engines', ['google', 'reddit', 'youtube'])
        self.max_results_per_engine = config.get('max_results_per_engine', 5)
        self.download_directory = config.get('download_directory', 'web_resources')
        self.cache_duration_days = config.get('cache_duration_days', 7)
        
        # API keys and endpoints
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.google_search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
        # Setup download directory
        os.makedirs(self.download_directory, exist_ok=True)
        
        # Content type priorities
        self.content_priorities = {
            'manual': 10,
            'pdf': 9,
            'forum_post': 7,
            'article': 6,
            'video': 5
        }
        
        # Source reliability scores
        self.source_reliability = {
            'manufacturer': 10,
            'reddit': 8,
            'youtube': 7,
            'automotive_forum': 8,
            'general_forum': 6
        }

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Required process method from BaseAgent"""
        query = input_data.get('query', '')
        search_strategy = input_data.get('search_strategy')
        
        if not search_strategy:
            return self._create_response(
                success=False,
                data={},
                confidence=0.0,
                reasoning="No search strategy provided",
                processing_time=0.0,
                errors=["search_strategy is required"]
            )
        
        return await self.process_query(query, search_strategy)

    async def process_query(self, query: str, search_strategy: SearchStrategy) -> AgentResponse:
        """Main entry point for web search processing"""
        start_time = datetime.now()
        
        try:
            logger.info(f"WebSearchAgent processing: {query}")
            
            # Extract search parameters from strategy
            equipment_info = {'category': search_strategy.search_metadata.equipment_category.value}
            intent = search_strategy.search_metadata.intent_primary.value
            
            # Generate targeted search queries
            search_queries = self._generate_search_queries(query, equipment_info, intent)
            
            # Perform searches across different engines/sources
            all_resources = []
            sources_searched = []
            
            for engine in self.search_engines:
                try:
                    if engine == 'google' and self.google_api_key:
                        resources = await self._search_google(search_queries)
                        all_resources.extend(resources)
                        sources_searched.append('google')
                    
                    elif engine == 'reddit':
                        resources = await self._search_reddit(search_queries, equipment_info)
                        all_resources.extend(resources)
                        sources_searched.append('reddit')
                    
                    elif engine == 'youtube':
                        resources = await self._search_youtube(search_queries, equipment_info)
                        all_resources.extend(resources)
                        sources_searched.append('youtube')
                        
                except Exception as e:
                    logger.error(f"Error searching {engine}: {e}")
            
            # Process and rank resources
            processed_resources = await self._process_resources(all_resources, query)
            
            # Download high-value resources
            downloaded_resources = await self._download_resources(processed_resources[:3])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create search result
            search_result = SearchResult(
                query=query,
                resources=downloaded_resources,
                search_time=processing_time,
                sources_searched=sources_searched,
                total_found=len(all_resources),
                success=True
            )
            
            return self._create_response(
                success=True,
                data={'search_result': asdict(search_result)},
                confidence=0.9 if downloaded_resources else 0.6,
                reasoning=f"Found {len(all_resources)} resources from {len(sources_searched)} sources",
                processing_time=processing_time,
                metadata={
                    'sources_searched': sources_searched,
                    'resources_found': len(all_resources),
                    'resources_downloaded': len(downloaded_resources)
                }
            )
            
        except Exception as e:
            logger.error(f"WebSearchAgent error: {e}")
            return self._create_response(
                success=False,
                data={},
                confidence=0.0,
                reasoning=f"Web search failed: {str(e)}",
                processing_time=(datetime.now() - start_time).total_seconds(),
                errors=[str(e)]
            )

    def _generate_search_queries(self, query: str, equipment_info: Dict, intent: str) -> List[str]:
        """Generate targeted search queries based on equipment and intent"""
        queries = []
        
        # Extract equipment details
        category = equipment_info.get('category', '')
        brand = equipment_info.get('brand', '')
        model = equipment_info.get('model', '')
        year = equipment_info.get('year', '')
        component = equipment_info.get('component', '')
        
        # Base query
        base = f"{year} {brand} {model} {component}".strip()
        
        # Intent-specific queries
        if intent == 'troubleshooting':
            queries.extend([
                f"{base} repair manual PDF",
                f"{base} service manual download",
                f"{base} replacement guide",
                f"how to replace {component} {year} {brand} {model}",
                f"{base} forum discussion reddit"
            ])
        
        elif intent == 'specifications':
            queries.extend([
                f"{base} specifications manual",
                f"{base} torque specs",
                f"{base} parts diagram",
                f"{year} {brand} {model} {component} specs PDF"
            ])
        
        elif intent == 'diagram_request':
            queries.extend([
                f"{base} wiring diagram",
                f"{base} schematic PDF",
                f"{year} {brand} {model} electrical diagram",
                f"{base} parts diagram manual"
            ])
        
        # General fallback queries
        queries.extend([
            f"{base} manual",
            f"{year} {brand} {model} service manual",
            query  # Original query as fallback
        ])
        
        # Remove duplicates and empty queries
        return list(filter(None, list(dict.fromkeys(queries))))

    async def _search_google(self, queries: List[str]) -> List[WebResource]:
        """Search Google Custom Search API"""
        resources = []
        
        if not self.google_api_key or not self.google_search_engine_id:
            logger.warning("Google Search API not configured")
            return resources
        
        try:
            async with aiohttp.ClientSession() as session:
                for query in queries[:3]:  # Limit to top 3 queries
                    url = f"https://www.googleapis.com/customsearch/v1"
                    params = {
                        'key': self.google_api_key,
                        'cx': self.google_search_engine_id,
                        'q': query,
                        'num': 5,
                        'fileType': 'pdf'  # Prioritize PDF files
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('items', []):
                                resource = WebResource(
                                    url=item['link'],
                                    title=item['title'],
                                    content_type='pdf' if item['link'].endswith('.pdf') else 'article',
                                    source='manufacturer' if self._is_manufacturer_site(item['link']) else 'general',
                                    relevance_score=0.8,
                                    content_preview=item.get('snippet', ''),
                                    metadata={
                                        'search_query': query,
                                        'search_engine': 'google'
                                    }
                                )
                                resources.append(resource)
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Google search error: {e}")
        
        return resources

    async def _search_reddit(self, queries: List[str], equipment_info: Dict) -> List[WebResource]:
        """Search Reddit for relevant discussions"""
        resources = []
        
        try:
            # Relevant subreddits for automotive queries
            automotive_subreddits = [
                'MechanicAdvice', 'Infiniti', 'Nissan', 'Cartalk', 
                'autorepair', 'AskAMechanic', 'Justrolledintotheshop'
            ]
            
            # Use Reddit search API (no auth required for public posts)
            async with aiohttp.ClientSession() as session:
                for subreddit in automotive_subreddits[:4]:  # Limit subreddits
                    for query in queries[:2]:  # Limit queries per subreddit
                        url = f"https://www.reddit.com/r/{subreddit}/search.json"
                        params = {
                            'q': query,
                            'sort': 'relevance',
                            'limit': 3,
                            'restrict_sr': 'true'
                        }
                        
                        headers = {'User-Agent': 'ServiceManualBot/1.0'}
                        
                        async with session.get(url, params=params, headers=headers) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                for post in data.get('data', {}).get('children', []):
                                    post_data = post['data']
                                    
                                    resource = WebResource(
                                        url=f"https://reddit.com{post_data['permalink']}",
                                        title=post_data['title'],
                                        content_type='forum_post',
                                        source='reddit',
                                        relevance_score=min(post_data.get('score', 0) / 100, 1.0),
                                        content_preview=post_data.get('selftext', '')[:200],
                                        metadata={
                                            'subreddit': subreddit,
                                            'score': post_data.get('score', 0),
                                            'num_comments': post_data.get('num_comments', 0),
                                            'search_query': query
                                        }
                                    )
                                    resources.append(resource)
                        
                        await asyncio.sleep(0.2)  # Rate limiting
        
        except Exception as e:
            logger.error(f"Reddit search error: {e}")
        
        return resources

    async def _search_youtube(self, queries: List[str], equipment_info: Dict) -> List[WebResource]:
        """Search YouTube for tutorial videos"""
        resources = []
        
        try:
            # Use YouTube Data API v3 if available, otherwise scrape
            if os.getenv('YOUTUBE_API_KEY'):
                resources = await self._search_youtube_api(queries)
            else:
                resources = await self._search_youtube_scrape(queries)
        
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
        
        return resources

    async def _search_youtube_scrape(self, queries: List[str]) -> List[WebResource]:
        """Scrape YouTube search results (fallback when no API key)"""
        resources = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for query in queries[:2]:
                    search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
                    
                    async with session.get(search_url) as response:
                        if response.status == 200:
                            html = await response.text()
                            
                            # Extract video info using regex
                            video_pattern = r'"videoId":"([^"]+)".*?"title":{"runs":\[{"text":"([^"]+)"}.*?"shortViewCountText":{"simpleText":"([^"]+)"}'
                            matches = re.findall(video_pattern, html)
                            
                            for video_id, title, view_count in matches[:3]:
                                resource = WebResource(
                                    url=f"https://www.youtube.com/watch?v={video_id}",
                                    title=title,
                                    content_type='video',
                                    source='youtube',
                                    relevance_score=0.7,
                                    content_preview=f"YouTube tutorial - {view_count} views",
                                    metadata={
                                        'video_id': video_id,
                                        'view_count': view_count,
                                        'search_query': query
                                    }
                                )
                                resources.append(resource)
                    
                    await asyncio.sleep(0.5)  # Rate limiting
        
        except Exception as e:
            logger.error(f"YouTube scraping error: {e}")
        
        return resources

    async def _process_resources(self, resources: List[WebResource], query: str) -> List[WebResource]:
        """Process and rank resources by relevance and quality"""
        
        for resource in resources:
            # Calculate composite score
            content_score = self.content_priorities.get(resource.content_type, 5)
            source_score = self.source_reliability.get(resource.source, 5)
            
            # Boost score for manufacturer sources
            if resource.source == 'manufacturer':
                source_score *= 1.5
            
            # Boost score for PDF manuals
            if resource.content_type == 'pdf' and 'manual' in resource.title.lower():
                content_score *= 1.3
            
            # Calculate final relevance score
            resource.relevance_score = (
                resource.relevance_score * 0.4 +
                (content_score / 10) * 0.4 +
                (source_score / 10) * 0.2
            )
        
        # Sort by relevance score
        return sorted(resources, key=lambda x: x.relevance_score, reverse=True)

    async def _download_resources(self, resources: List[WebResource]) -> List[WebResource]:
        """Download high-value resources (PDFs, etc.)"""
        downloaded = []
        
        for resource in resources:
            try:
                if resource.content_type == 'pdf':
                    success = await self._download_pdf(resource)
                    if success:
                        downloaded.append(resource)
                elif resource.content_type == 'forum_post':
                    success = await self._extract_forum_content(resource)
                    if success:
                        downloaded.append(resource)
                else:
                    # For other types, just include metadata
                    downloaded.append(resource)
            
            except Exception as e:
                logger.error(f"Error downloading {resource.url}: {e}")
        
        return downloaded

    async def _download_pdf(self, resource: WebResource) -> bool:
        """Download a PDF file"""
        try:
            # Generate filename
            url_hash = hashlib.md5(resource.url.encode()).hexdigest()[:8]
            filename = f"manual_{url_hash}.pdf"
            filepath = os.path.join(self.download_directory, filename)
            
            # Check if already downloaded
            if os.path.exists(filepath):
                resource.download_path = filepath
                await self._extract_pdf_content(resource)
                return True
            
            # Download file
            async with aiohttp.ClientSession() as session:
                async with session.get(resource.url) as response:
                    if response.status == 200 and response.content_type == 'application/pdf':
                        content = await response.read()
                        
                        with open(filepath, 'wb') as f:
                            f.write(content)
                        
                        resource.download_path = filepath
                        await self._extract_pdf_content(resource)
                        
                        logger.info(f"Downloaded PDF: {filename}")
                        return True
        
        except Exception as e:
            logger.error(f"PDF download error: {e}")
        
        return False

    async def _extract_pdf_content(self, resource: WebResource):
        """Extract text content from downloaded PDF"""
        try:
            if not resource.download_path or not os.path.exists(resource.download_path):
                return
            
            text_content = ""
            
            # Try pdfplumber first
            try:
                with pdfplumber.open(resource.download_path) as pdf:
                    for page in pdf.pages[:10]:  # First 10 pages
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
            except:
                # Fallback to PyMuPDF
                doc = fitz.open(resource.download_path)
                for page_num in range(min(10, doc.page_count)):
                    page = doc[page_num]
                    text_content += page.get_text() + "\n"
                doc.close()
            
            # Store extracted content
            resource.extracted_content = text_content[:2000]  # First 2000 chars
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")

    async def _extract_forum_content(self, resource: WebResource) -> bool:
        """Extract content from forum posts"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(resource.url + '.json') as response:  # Reddit JSON API
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract post and top comments
                        post = data[0]['data']['children'][0]['data']
                        comments = data[1]['data']['children']
                        
                        content = f"**Post**: {post.get('selftext', '')}\n\n"
                        
                        # Add top comments
                        for comment in comments[:3]:
                            comment_data = comment['data']
                            if comment_data.get('body'):
                                content += f"**Comment** ({comment_data.get('score', 0)} pts): {comment_data['body'][:300]}...\n\n"
                        
                        resource.extracted_content = content
                        return True
        
        except Exception as e:
            logger.error(f"Forum extraction error: {e}")
        
        return False

    def _is_manufacturer_site(self, url: str) -> bool:
        """Check if URL is from a manufacturer website"""
        manufacturer_domains = [
            'nissan.com', 'infiniti.com', 'nissanusa.com', 'infinitiusa.com',
            'ford.com', 'toyota.com', 'honda.com', 'gm.com', 'chrysler.com'
        ]
        
        domain = urlparse(url).netloc.lower()
        return any(mfg in domain for mfg in manufacturer_domains)

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'name': self.name,
            'ready': True,
            'search_engines_configured': {
                'google': bool(self.google_api_key and self.google_search_engine_id),
                'reddit': True,  # No auth required for public search
                'youtube': True   # Scraping fallback available
            },
            'download_directory': self.download_directory,
            'cache_info': {
                'location': self.download_directory,
                'size_mb': self._get_cache_size_mb()
            }
        }

    def _get_cache_size_mb(self) -> float:
        """Get cache directory size in MB"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.download_directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0.0
