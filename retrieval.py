import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime
import json
import pickle
from typing import List, Dict, Tuple, Optional
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import threading
import fcntl
from contextlib import contextmanager

# Environment variables used:
# - NLTK_DATA: NLTK data directory path

# --- Ensure NLTK data path is writable and robust ---
nltk_data_env = os.environ.get('NLTK_DATA', None)
if nltk_data_env:
    nltk_data_dir = nltk_data_env
else:
    nltk_data_dir = 'textbooks/nltk_data'

os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Try to load stopwords, download if necessary
try:
    stopword_set = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
    stopword_set = set(stopwords.words('english'))
except Exception:
    stopword_set = set()

class RetrievalAugmentor:
    """
    Enhanced retrieval system with persistent index, intelligent chunking, semantic reranking,
    robust error handling, and thread-safe operations.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(RetrievalAugmentor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self,
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 index_path="textbooks/faiss_index.bin",
                 metadata_path="textbooks/metadata.json",
                 chunk_size=400,
                 chunk_overlap=50):
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self._initialized = True
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = self._get_logger()
        self._file_lock = threading.RLock()
        
        self.logger.info("Initializing RetrievalAugmentor...")
        
        # Ensure the directory for index and metadata exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        
        # Initialize semantic model
        try:
            self.model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded semantic model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load semantic model: {e}")
            raise
        
        # Initialize stopwords from global set
        self.stop_words = stopword_set
        
        # Load existing index and metadata
        self._load_index_and_metadata()
        
        # Statistics
        self.stats = {
            'total_chunks': len(self.metadata),
            'total_queries': 0,
            'successful_retrievals': 0
        }
        
        # Save metadata on successful initialization
        if self.metadata:
            self._save_metadata_atomic()

    @contextmanager
    def _atomic_file_operation(self, file_path, mode='r'):
        """Context manager for atomic file operations with locking."""
        lock_file = file_path + '.lock'
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Try to create lock file with proper permissions
            try:
                with open(lock_file, 'w') as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    if 'w' in mode:
                        temp_file = file_path + '.tmp'
                        with open(temp_file, mode) as f:
                            yield f
                        os.replace(temp_file, file_path)
                    else:
                        try:
                            with open(file_path, mode) as f:
                                yield f
                        except FileNotFoundError:
                            yield None
            except (OSError, PermissionError) as e:
                self.logger.warning(f"File locking failed for {file_path}: {e}. Proceeding without lock.")
                # Fallback without locking
                if 'w' in mode:
                    temp_file = file_path + '.tmp'
                    with open(temp_file, mode) as f:
                        yield f
                    os.replace(temp_file, file_path)
                else:
                    try:
                        with open(file_path, mode) as f:
                            yield f
                    except FileNotFoundError:
                        yield None
        finally:
            try:
                os.remove(lock_file)
            except (FileNotFoundError, PermissionError):
                pass

    def _get_logger(self):
        import logging
        logger = logging.getLogger("quillai_retrieval")
        if not logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)
        return logger

    def _load_index_and_metadata(self):
        """Load index and metadata with thread safety."""
        with self._file_lock:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    with self._atomic_file_operation(self.metadata_path, 'r') as f:
                        if f:
                            self.metadata = json.load(f)
                        else:
                            self.metadata = []
                    self.logger.info(f"Loaded existing index with {len(self.metadata)} chunks from {self.index_path}")
                except Exception as e:
                    self.logger.warning(f"Could not load existing index from {self.index_path}: {e}")
                    self._initialize_new_index()
            else:
                self.logger.info(f"No existing index found at {self.index_path}. Initializing new index.")
                self._initialize_new_index()

    def _initialize_new_index(self):
        """Initialize new empty index and metadata."""
        # Use 384 as default dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(384)
        self.metadata = []
        self.logger.info("Initialized new FAISS index and metadata.")

    def _save_index_and_metadata(self):
        """Save index and metadata with atomic operations."""
        with self._file_lock:
            try:
                # Save index
                if self.index is not None and self.index.ntotal > 0:
                    temp_index_path = self.index_path + '.tmp'
                    faiss.write_index(self.index, temp_index_path)
                    os.replace(temp_index_path, self.index_path)
                    self.logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
                
                # Save metadata atomically
                self._save_metadata_atomic()
                
            except Exception as e:
                self.logger.error(f"Failed to save index/metadata: {e}")
                raise

    def _save_metadata_atomic(self):
        """Save metadata with atomic write operation."""
        try:
            with self._atomic_file_operation(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved metadata to {self.metadata_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def reload_index(self):
        """Reload index and metadata from disk only if needed."""
        with self._file_lock:
            # Check if files have been modified since last load
            try:
                index_mtime = os.path.getmtime(self.index_path) if os.path.exists(self.index_path) else 0
                metadata_mtime = os.path.getmtime(self.metadata_path) if os.path.exists(self.metadata_path) else 0
                
                # Only reload if files are newer than our current state
                if hasattr(self, '_last_load_time'):
                    if index_mtime <= self._last_load_time and metadata_mtime <= self._last_load_time:
                        return  # No need to reload
                
                self._load_index_and_metadata()
                self._last_load_time = max(index_mtime, metadata_mtime)
                
            except Exception as e:
                self.logger.warning(f"Error checking file modification times: {e}")
                # Fallback to always reload
                self._load_index_and_metadata()

    def build_or_update_index_from_pdf(self, pdf_path, source_name=None, force_rebuild=False):
        """Enhanced PDF indexing with intelligent chunking and comprehensive error handling."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        source_name = source_name or os.path.basename(pdf_path)
        print(f"📚 Indexing PDF: {source_name}")
        
        # Check if already indexed
        if not force_rebuild:
            existing_chunks = [m for m in self.metadata if m.get('source') == source_name]
            if existing_chunks:
                print(f"⚠ Source '{source_name}' already indexed ({len(existing_chunks)} chunks)")
                print("   Use force_rebuild=True to reindex")
                return
        
        try:
            raw_text = self._extract_text_from_pdf(pdf_path)
            if not raw_text.strip():
                print("No text extracted from PDF")
                return
            
            print(f"Extracted {len(raw_text):,} characters")
            
            cleaned_text = self._clean_and_preprocess_text(raw_text)
            print(f"Cleaned text: {len(cleaned_text):,} characters")
            
            chunks = self._create_intelligent_chunks(cleaned_text, source_name)
            print(f"Created {len(chunks)} intelligent chunks")
            
            if not chunks:
                print("No valid chunks created")
                return
            
            print("Generating embeddings...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self._generate_embeddings_batch(chunk_texts)
            
            if embeddings is None:
                print("Failed to generate embeddings")
                return
            
            self._update_index_with_chunks(chunks, embeddings)
            self._save_index_and_metadata()
            
            print(f"Successfully indexed {len(chunks)} chunks from {source_name}")
            self.stats['total_chunks'] = len(self.metadata)
            
        except Exception as e:
            error_msg = f"Failed to index PDF {source_name}: {e}"
            print(error_msg)
            raise

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with error handling."""
        text_blocks = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    try:
                        page_text = page.get_text()
                        if page_text.strip():
                            text_blocks.append(page_text)
                    except Exception as e:
                        print(f"Could not extract text from page {page_num + 1}: {e}")
                        continue
        except Exception as e:
            print(f"Could not open PDF {pdf_path}: {e}")
            raise
        
        return "\n".join(text_blocks)

    def _clean_and_preprocess_text(self, text):
        """Clean and preprocess extracted text."""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\bpage\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bchapter\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsection\s+\d+(\.\d+)*\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfigure\s+\d+(\.\d+)*\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\btable\s+\d+(\.\d+)*\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfig\.\s*\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        text = re.sub(r'[=]{3,}', '===', text)
        text = re.sub(r'\b\d+\b(?=\s|$)', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', text)
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _create_intelligent_chunks(self, text, source_name):
        """Create intelligent chunks using sentence tokenization."""
        if not text.strip():
            return []
        
        chunks = []
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return self._create_simple_chunks(text, source_name)
            
            current_chunk = ""
            current_word_count = 0
            chunk_id = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_words = len(sentence.split())
                
                if current_word_count + sentence_words > self.chunk_size and current_chunk:
                    if current_chunk.strip():
                        chunk_metadata = self._create_chunk_metadata(
                            current_chunk.strip(), source_name, chunk_id
                        )
                        if chunk_metadata:
                            chunks.append(chunk_metadata)
                            chunk_id += 1
                    
                    if self.chunk_overlap > 0:
                        overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                        current_chunk = overlap_text + " " + sentence
                        current_word_count = len(current_chunk.split())
                    else:
                        current_chunk = sentence
                        current_word_count = sentence_words
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                    current_word_count += sentence_words
            
            if current_chunk.strip():
                chunk_metadata = self._create_chunk_metadata(
                    current_chunk.strip(), source_name, chunk_id
                )
                if chunk_metadata:
                    chunks.append(chunk_metadata)
            
        except Exception as e:
            print(f"Intelligent chunking failed: {e}, falling back to simple chunking")
            return self._create_simple_chunks(text, source_name)
        
        return chunks

    def _create_simple_chunks(self, text, source_name):
        """Create simple word-based chunks as fallback."""
        words = text.split()
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_metadata = self._create_chunk_metadata(chunk_text, source_name, chunk_id)
            if chunk_metadata:
                chunks.append(chunk_metadata)
                chunk_id += 1
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
        
        return chunks

    def _get_overlap_text(self, text, overlap_words):
        """Get overlap text from the end of current chunk."""
        words = text.split()
        if len(words) <= overlap_words:
            return text
        return " ".join(words[-overlap_words:])

    def _create_chunk_metadata(self, text, source_name, chunk_id):
        """Create metadata for a chunk with quality assessment."""
        if not text or len(text.strip()) < 20:
            return None
        
        word_count = len(text.split())
        char_count = len(text)
        
        if word_count < 10:
            return None
        
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.5:
            return None
        
        return {
            'text': text,
            'source': source_name,
            'chunk_id': chunk_id,
            'word_count': word_count,
            'char_count': char_count,
            'created_at': datetime.utcnow().isoformat(),
            'quality_score': self._calculate_quality_score(text)
        }

    def _calculate_quality_score(self, text):
        """Calculate quality score for a chunk."""
        score = 0
        word_count = len(text.split())
        
        if 50 <= word_count <= 300:
            score += 2
        elif 20 <= word_count < 50 or 300 < word_count <= 500:
            score += 1
        
        academic_terms = [
            'algorithm', 'method', 'approach', 'technique', 'process',
            'system', 'model', 'theory', 'principle', 'concept',
            'definition', 'example', 'application', 'analysis'
        ]
        text_lower = text.lower()
        academic_score = sum(1 for term in academic_terms if term in text_lower)
        score += min(academic_score, 3)
        
        sentence_count = len([s for s in text.split('.') if s.strip()])
        if sentence_count >= 2:
            score += 1
        
        words = text.lower().split()
        unique_words = set(words)
        if len(words) > 0 and len(unique_words) / len(words) > 0.7:
            score += 1
        
        return score

    def _generate_embeddings_batch(self, texts, batch_size=32):
        """Generate embeddings for a batch of texts."""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=True
            )
            return embeddings
        except Exception as e:
            print(f"Failed to generate embeddings: {e}")
            return None

    def _update_index_with_chunks(self, chunks, embeddings):
        """Update FAISS index with new chunks and embeddings."""
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            print(f"✓ Created new FAISS index (dimension: {dimension})")
        
        self.index.add(embeddings)
        self.metadata.extend(chunks)
        print(f"✓ Added {len(chunks)} chunks to index")

    def retrieve_context(self, query, top_k=5, min_score_threshold=0.3):
        """Retrieve relevant context chunks for a query."""
        if self.index is None or not self.metadata:
            print("⚠ No index available for retrieval")
            return []
        
        self.stats['total_queries'] += 1
        
        try:
            print(f"🔍 Retrieving context for: {query[:50]}...")
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            search_k = min(top_k * 3, len(self.metadata))
            distances, indices = self.index.search(query_embedding, search_k)
            
            similarities = 1 / (1 + distances[0])
            candidates = []
            
            for idx, similarity in zip(indices[0], similarities):
                if 0 <= idx < len(self.metadata):
                    chunk = self.metadata[idx]
                    candidates.append({
                        'chunk': chunk,
                        'similarity': similarity,
                        'index': idx
                    })
            
            candidates = [c for c in candidates if c['similarity'] >= min_score_threshold]
            
            if not candidates:
                print(f"⚠ No chunks found above similarity threshold {min_score_threshold}")
                return []
            
            reranked_candidates = self._enhanced_rerank_candidates(query, candidates)
            top_candidates = reranked_candidates[:top_k]
            relevant_chunks = [c['chunk']['text'] for c in top_candidates]
            
            print(f"✓ Retrieved {len(relevant_chunks)} relevant chunks")
            
            if relevant_chunks:
                self.stats['successful_retrievals'] += 1
            
            return relevant_chunks
            
        except Exception as e:
            error_msg = f"Context retrieval failed: {e}"
            print(f"❌ {error_msg}")
            return []

    def _enhanced_rerank_candidates(self, query, candidates):
        """Enhanced reranking with domain awareness."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        query_words_filtered = query_words - self.stop_words
        
        for candidate in candidates:
            chunk_text = candidate['chunk']['text'].lower()
            chunk_words = set(chunk_text.split())
            
            base_score = candidate['similarity']
            
            keyword_overlap = len(query_words_filtered & chunk_words)
            keyword_bonus = keyword_overlap * 0.1
            
            quality_bonus = candidate['chunk'].get('quality_score', 0) * 0.05
            
            word_count = candidate['chunk']['word_count']
            length_penalty = 0
            if word_count < 30:
                length_penalty = -0.2
            elif word_count > 400:
                length_penalty = -0.1
            
            academic_terms = [
                'algorithm', 'method', 'approach', 'technique', 'process',
                'definition', 'example', 'principle', 'concept'
            ]
            academic_bonus = sum(0.02 for term in academic_terms if term in chunk_text)
            
            final_score = base_score + keyword_bonus + quality_bonus + length_penalty + academic_bonus
            candidate['final_score'] = final_score
        
        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)

    def get_index_stats(self):
        """Get comprehensive index statistics."""
        if not self.metadata:
            return {
                'total_chunks': 0,
                'total_sources': 0,
                'sources': [],
                'index_size': 0,
                'avg_chunk_words': 0,
                'avg_quality_score': 0,
                'total_queries': self.stats['total_queries'],
                'successful_retrievals': self.stats['successful_retrievals'],
                'success_rate': 0
            }
        
        sources = set(chunk.get('source', 'unknown') for chunk in self.metadata)
        word_counts = [chunk.get('word_count', 0) for chunk in self.metadata]
        quality_scores = [chunk.get('quality_score', 0) for chunk in self.metadata]
        
        stats = {
            'total_chunks': len(self.metadata),
            'total_sources': len(sources),
            'sources': list(sources),
            'index_size': self.index.ntotal if self.index else 0,
            'avg_chunk_words': np.mean(word_counts) if word_counts else 0,
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'total_queries': self.stats['total_queries'],
            'successful_retrievals': self.stats['successful_retrievals'],
            'success_rate': (self.stats['successful_retrievals'] / max(self.stats['total_queries'], 1)) * 100
        }
        
        return stats

    def print_index_stats(self):
        """Print comprehensive index statistics."""
        stats = self.get_index_stats()
        print("📊 Retrieval Index Statistics:")
        print("-" * 40)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total sources: {stats['total_sources']}")
        print(f"Index size: {stats['index_size']}")
        print(f"Average chunk words: {stats['avg_chunk_words']:.1f}")
        print(f"Average quality score: {stats['avg_quality_score']:.1f}")
        print(f"Total queries: {stats['total_queries']}")
        print(f"Successful retrievals: {stats['successful_retrievals']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        
        if stats['sources']:
            print(f"\nSources:")
            for source in stats['sources']:
                source_chunks = [c for c in self.metadata if c.get('source') == source]
                print(f"   • {source}: {len(source_chunks)} chunks")

    def search_chunks(self, query, max_results=10):
        """Search chunks and return detailed results."""
        if self.index is None or not self.metadata:
            return []
        
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            search_k = min(max_results * 2, len(self.metadata))
            distances, indices = self.index.search(query_embedding, search_k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.metadata):
                    chunk = self.metadata[idx]
                    similarity = 1 / (1 + distance)
                    results.append({
                        'text': chunk['text'],
                        'source': chunk.get('source', 'unknown'),
                        'chunk_id': chunk.get('chunk_id', 0),
                        'word_count': chunk.get('word_count', 0),
                        'quality_score': chunk.get('quality_score', 0),
                        'similarity': similarity,
                        'preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                    })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            print(f"Chunk search failed: {e}")
            return []

    def remove_source(self, source_name):
        """Remove all chunks from a specific source."""
        with self._file_lock:
            if not self.metadata:
                print(f"No chunks to remove for source: {source_name}")
                return False
            
            chunks_to_remove = [i for i, chunk in enumerate(self.metadata) if chunk.get('source') == source_name]
            
            if not chunks_to_remove:
                print(f"No chunks found for source: {source_name}")
                return False
            
            print(f"🗑️ Removing {len(chunks_to_remove)} chunks from source: {source_name}")
            
            try:
                for i in reversed(chunks_to_remove):
                    del self.metadata[i]
                
                if self.metadata:
                    print("🔄 Rebuilding index...")
                    chunk_texts = [chunk['text'] for chunk in self.metadata]
                    embeddings = self._generate_embeddings_batch(chunk_texts)
                    
                    if embeddings is not None:
                        dimension = embeddings.shape[1]
                        self.index = faiss.IndexFlatL2(dimension)
                        self.index.add(embeddings)
                        self._save_index_and_metadata()
                        print(f"✅ Successfully removed source: {source_name}")
                        self.stats['total_chunks'] = len(self.metadata)
                        return True
                    else:
                        print("❌ Failed to rebuild index")
                        return False
                else:
                    self._initialize_new_index()
                    self._save_index_and_metadata()
                    print(f"✅ Removed last source: {source_name}")
                    self.stats['total_chunks'] = 0
                    return True
                
            except Exception as e:
                error_msg = f"Failed to remove source {source_name}: {e}"
                print(f"❌ {error_msg}")
                return False


# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Retrieval Augmentor Test")
    print("=" * 50)
    
    # Initialize retrieval system
    retrieval = RetrievalAugmentor(chunk_size=300, chunk_overlap=30)
    
    # Print current stats
    retrieval.print_index_stats()
    
    print("\nRetrieval Augmentor ready for use!")
    print("Example usage:")
    print("   retrieval.build_or_update_index_from_pdf('textbook.pdf')")
    print("   chunks = retrieval.retrieve_context('machine learning', top_k=3)")
    print("   results = retrieval.search_chunks('algorithms', max_results=5)")