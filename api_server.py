from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
import shutil
import os
import logging
import threading
from datetime import datetime
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
import time

# Environment variables used:
# - API_KEY: API authentication key (default: "dev-key-change-in-production")
# - ALLOWED_ORIGINS: CORS allowed origins (default: "*")
# - ENVIRONMENT: Environment setting (default: "development")

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
from quillai_llm import QuillAILLM
from retrieval import RetrievalAugmentor
from knowledgebase import KnowledgeBaseManager
from response_cache import DualResponseCache

# Security
security = HTTPBearer(auto_error=False)

# Global state management with thread safety
class AppState:
    def __init__(self):
        self._lock = threading.RLock()
        self._llm_instance: Optional[QuillAILLM] = None
        self._retrieval_instance: Optional[RetrievalAugmentor] = None
        self._cache_instance: Optional[DualResponseCache] = None
        self._kb_manager: Optional[KnowledgeBaseManager] = None
        self._initialized = False

    def get_llm(self) -> QuillAILLM:
        with self._lock:
            if self._llm_instance is None:
                raise HTTPException(status_code=503, detail="LLM not initialized")
            return self._llm_instance

    def get_retrieval(self) -> RetrievalAugmentor:
        with self._lock:
            if self._retrieval_instance is None:
                raise HTTPException(status_code=503, detail="Retrieval system not initialized")
            return self._retrieval_instance

    def get_cache(self) -> DualResponseCache:
        with self._lock:
            if self._cache_instance is None:
                raise HTTPException(status_code=503, detail="Cache not initialized")
            return self._cache_instance

    def get_kb_manager(self) -> KnowledgeBaseManager:
        with self._lock:
            if self._kb_manager is None:
                raise HTTPException(status_code=503, detail="Knowledge base manager not initialized")
            return self._kb_manager

    def initialize(self):
        with self._lock:
            if self._initialized:
                return
            
            logger.info("🚀 Initializing QuillAI components...")
            
            # Initialize components
            self._llm_instance = QuillAILLM(
                model_name="microsoft/DialoGPT-medium",
                force_model_check=True,
                debug_mode=False
            )
            
            self._retrieval_instance = RetrievalAugmentor(
                chunk_size=400,
                chunk_overlap=50
            )
            
            # Connect LLM with retrieval system
            self._llm_instance.set_retrieval_system(self._retrieval_instance)
            
            self._cache_instance = DualResponseCache()
            self._kb_manager = KnowledgeBaseManager(storage_dir="textbooks")
            
            self._initialized = True
            logger.info("✅ QuillAI components initialized successfully")

    def invalidate_cache(self):
        """Invalidate cache when knowledge base changes"""
        with self._lock:
            if self._cache_instance:
                self._cache_instance.clear_all()

# Global app state
app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app_state.initialize()
    yield
    # Shutdown - cleanup if needed
    pass

app = FastAPI(
    title="QuillAI Enhanced Academic Assistant API",
    description="Intelligent academic assistant with dual output system for Flutter integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration - restrict in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")

def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if credentials is None:
        # For legacy endpoints, allow without authentication in development
        if os.getenv("ENVIRONMENT", "development") == "development":
            return None
        raise HTTPException(status_code=401, detail="API key required")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials

def verify_api_key_required(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials

# Pydantic models
class QueryRequest(BaseModel):
    prompt: str
    mode: int
    marks: int

    @validator('mode')
    def validate_mode(cls, v):
        if v not in [0, 1]:
            raise ValueError('Mode must be 0 (learning) or 1 (question)')
        return v

    @validator('marks')
    def validate_marks(cls, v):
        if v not in [0, 2, 5, 10]:
            raise ValueError('Marks must be 0, 2, 5, or 10')
        return v

    def to_internal_format(self):
        return {
            'query': self.prompt,
            'mode': 'learning' if self.mode == 0 else 'question',
            'marks': self.marks if self.marks > 0 else None
        }

class PDFUploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    error_message: Optional[str] = None

class SystemStatsResponse(BaseModel):
    knowledge_base: dict
    retrieval_index: dict
    timestamp: str

class QueryResponse(BaseModel):
    success: bool
    dialogpt_output: Optional[str] = None
    custom_llm: Optional[str] = None
    intent: Optional[str] = None
    domain: Optional[str] = None
    topics: Optional[List[str]] = None
    context_used: bool = False
    context_sources: List[Dict[str, Any]] = []
    generation_time: float = 0.0
    word_count: int = 0
    timestamp: str
    warning: Optional[str] = None
    error_message: Optional[str] = None

# Dependency injection
def get_llm() -> QuillAILLM:
    return app_state.get_llm()

def get_retrieval() -> RetrievalAugmentor:
    return app_state.get_retrieval()

def get_cache() -> DualResponseCache:
    return app_state.get_cache()

def get_kb_manager() -> KnowledgeBaseManager:
    return app_state.get_kb_manager()

logger = logging.getLogger("quillai_api_server")
logging.basicConfig(level=logging.INFO)

# API v1 routes
@app.post("/v1/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    llm: QuillAILLM = Depends(get_llm),
    retrieval: RetrievalAugmentor = Depends(get_retrieval),
    cache: DualResponseCache = Depends(get_cache),
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key_required)
):
    """Process academic queries and return dual outputs (LLM + Custom)."""
    
    from types import SimpleNamespace
    internal_query_obj = SimpleNamespace(**request.to_internal_format())
    query_text = internal_query_obj.query
    mode_str = internal_query_obj.mode
    marks_val = internal_query_obj.marks

    # Check cache first
    cached_response = cache.get(query_text, mode_str, marks_val)
    if cached_response:
        logger.info(f"Cache hit for query: {query_text[:50]}...")
        return cached_response

    try:
        logger.info(f"📝 Processing query: {query_text[:50]}...")
        
        if not query_text or not query_text.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Retrieve context
        context_chunks = []
        context_sources = []
        context_used = False
        context_warning = None

        try:
            context_chunks = retrieval.retrieve_context(query_text, top_k=3)
            if context_chunks:
                context_used = True
                # Find source and chunk_id for each context chunk
                for chunk_text in context_chunks:
                    for meta in retrieval.metadata:
                        if meta['text'].strip() == chunk_text.strip():
                            context_sources.append({
                                "source": meta.get("source", "unknown"),
                                "chunk_id": meta.get("chunk_id", -1)
                            })
                            break
            else:
                context_used = False
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            context_chunks = []
            context_used = False

        if not context_used and retrieval and retrieval.metadata:
            context_warning = "No relevant context found for this query, answering generically."
            logger.warning(f"Query answered without context: {query_text[:80]}")

        # Generate answer using dual response
        result = llm.generate_dual_response(
            query=query_text,
            mode=mode_str,
            marks=marks_val,
            context_chunks=context_chunks
        )

        # Create response
        response = QueryResponse(
            success=True,
            dialogpt_output=result.get("llm_output", ""),
            custom_llm=result.get("custom_output", ""),
            intent=result.get("intent", ""),
            domain=result.get("domain", ""),
            topics=result.get("topics", []),
            context_used=context_used,
            context_sources=context_sources,
            generation_time=result.get("generation_times", {}).get("total", 0),
            word_count=result.get("word_counts", {}).get("custom", 0),
            timestamp=datetime.utcnow().isoformat() + "Z",
            warning=context_warning
        )

        # Cache the response
        cache.set(query_text, mode_str, marks_val, response.dict())
        logger.info(f"Successful query: {query_text[:100]}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/v1/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_ocr: bool = False,
    language: str = "eng",
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key_required)
):
    """Upload and process PDF files for the knowledge base."""
    
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Store directly in textbooks directory
        file_path = os.path.join("textbooks", file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"PDF '{file.filename}' uploaded. Starting background processing.")

        background_tasks.add_task(
            process_uploaded_pdf, 
            file_path, 
            file.filename, 
            use_ocr, 
            language
        )

        return PDFUploadResponse(
            success=True,
            message=f"PDF '{file.filename}' uploaded successfully and is being processed",
            filename=file.filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_uploaded_pdf(file_path: str, filename: str, use_ocr: bool, language: str):
    """Background task to process uploaded PDF."""
    try:
        logger.info(f"Processing uploaded PDF: {filename}")
        
        # Get instances
        kb_manager = app_state.get_kb_manager()
        retrieval_instance = app_state.get_retrieval()
        
        # Add to knowledge base
        kb_manager.add_pdf(file_path, force_ocr=use_ocr, language=language)
        logger.info(f"PDF '{filename}' added to knowledge base.")

        # Index the PDF
        retrieval_instance.build_or_update_index_from_pdf(
            file_path,
            source_name=filename,
            force_rebuild=True
        )
        logger.info(f"PDF '{filename}' indexed.")

        # Invalidate cache since knowledge base changed
        app_state.invalidate_cache()
        
        logger.info(f"PDF '{filename}' processing completed successfully.")

    except Exception as e:
        logger.error(f"Failed to process uploaded PDF {filename}: {e}")
        # Keep the file for debugging - don't remove it

@app.get("/v1/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
    retrieval: RetrievalAugmentor = Depends(get_retrieval),
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key_required)
):
    """Get comprehensive system statistics."""
    try:
        kb_stats = kb_manager.get_storage_stats()
        index_stats = retrieval.get_index_stats()

        return SystemStatsResponse(
            knowledge_base=kb_stats,
            retrieval_index=index_stats,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/v1/list-pdfs")
async def list_pdfs(
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager),
    retrieval: RetrievalAugmentor = Depends(get_retrieval),
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key_required)
):
    """List all PDFs in the knowledge base and all indexed sources."""
    try:
        kb_list = kb_manager.list_pdfs()
        sources = list(set(chunk.get("source", "unknown") for chunk in retrieval.metadata))

        return {
            "knowledge_base_pdfs": kb_list,
            "indexed_sources": sources,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Failed to list PDFs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list PDFs: {str(e)}")

@app.get("/v1/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "QuillAI API",
        "version": "1.0.0"
    }

@app.get("/v1/metrics")
async def get_metrics(
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key_required)
):
    """Get metrics for monitoring."""
    try:
        retrieval = app_state.get_retrieval()
        cache = app_state.get_cache()
        
        return {
            "total_queries": retrieval.stats.get('total_queries', 0),
            "successful_retrievals": retrieval.stats.get('successful_retrievals', 0),
            "cache_hits": cache.get_stats().get('hits', 0),
            "cache_misses": cache.get_stats().get('misses', 0),
            "total_chunks": len(retrieval.metadata),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/v1/search")
async def search_knowledge_base(
    query: str, 
    max_results: int = 5,
    retrieval: RetrievalAugmentor = Depends(get_retrieval),
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key_required)
):
    """Search the knowledge base for debugging purposes."""
    try:
        results = retrieval.search_chunks(query, max_results=max_results)
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Legacy endpoints for backward compatibility (no authentication required)
@app.post("/query", response_model=QueryResponse)
async def legacy_process_query(
    request: QueryRequest,
    llm: QuillAILLM = Depends(get_llm),
    retrieval: RetrievalAugmentor = Depends(get_retrieval),
    cache: DualResponseCache = Depends(get_cache)
):
    """Legacy endpoint - no authentication required for backward compatibility."""
    from types import SimpleNamespace
    internal_query_obj = SimpleNamespace(**request.to_internal_format())
    query_text = internal_query_obj.query
    mode_str = internal_query_obj.mode
    marks_val = internal_query_obj.marks

    # Check cache first
    cached_response = cache.get(query_text, mode_str, marks_val)
    if cached_response:
        logger.info(f"Cache hit for query: {query_text[:50]}...")
        return cached_response

    try:
        logger.info(f"📝 Processing legacy query: {query_text[:50]}...")
        
        if not query_text or not query_text.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Retrieve context
        context_chunks = []
        context_sources = []
        context_used = False
        context_warning = None

        try:
            context_chunks = retrieval.retrieve_context(query_text, top_k=3)
            if context_chunks:
                context_used = True
                # Find source and chunk_id for each context chunk
                for chunk_text in context_chunks:
                    for meta in retrieval.metadata:
                        if meta['text'].strip() == chunk_text.strip():
                            context_sources.append({
                                "source": meta.get("source", "unknown"),
                                "chunk_id": meta.get("chunk_id", -1)
                            })
                            break
            else:
                context_used = False
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            context_chunks = []
            context_used = False

        if not context_used and retrieval and retrieval.metadata:
            context_warning = "No relevant context found for this query, answering generically."
            logger.warning(f"Query answered without context: {query_text[:80]}")

        # Generate answer using dual response
        result = llm.generate_dual_response(
            query=query_text,
            mode=mode_str,
            marks=marks_val,
            context_chunks=context_chunks
        )

        # Create response
        response = QueryResponse(
            success=True,
            dialogpt_output=result.get("llm_output", ""),
            custom_llm=result.get("custom_output", ""),
            intent=result.get("intent", ""),
            domain=result.get("domain", ""),
            topics=result.get("topics", []),
            context_used=context_used,
            context_sources=context_sources,
            generation_time=result.get("generation_times", {}).get("total", 0),
            word_count=result.get("word_counts", {}).get("custom", 0),
            timestamp=datetime.utcnow().isoformat() + "Z",
            warning=context_warning
        )

        # Cache the response
        cache.set(query_text, mode_str, marks_val, response.dict())
        logger.info(f"Successful legacy query: {query_text[:100]}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def legacy_upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_ocr: bool = False,
    language: str = "eng",
    kb_manager: KnowledgeBaseManager = Depends(get_kb_manager)
):
    """Legacy endpoint - no authentication required for backward compatibility."""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Store directly in textbooks directory
        file_path = os.path.join("textbooks", file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"PDF '{file.filename}' uploaded via legacy endpoint. Starting background processing.")

        background_tasks.add_task(
            process_uploaded_pdf, 
            file_path, 
            file.filename, 
            use_ocr, 
            language
        )

        return PDFUploadResponse(
            success=True,
            message=f"PDF '{file.filename}' uploaded successfully and is being processed",
            filename=file.filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Legacy upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "Please check the API documentation at /docs",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Please try again later or contact support",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "QuillAI Enhanced Academic Assistant API",
        "version": "1.0.0",
        "description": "Intelligent academic assistant with dual output system",
        "endpoints": {
            "main_query": "/v1/query",
            "upload_pdf": "/v1/upload-pdf",
            "statistics": "/v1/stats",
            "list_pdfs": "/v1/list-pdfs",
            "health": "/v1/health",
            "search": "/v1/search",
            "metrics": "/v1/metrics",
            "documentation": "/docs",
            "legacy_query": "/query",
            "legacy_upload": "/upload-pdf"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)