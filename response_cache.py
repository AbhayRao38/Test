import json
import os
import time
import fcntl
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import threading
import logging
from contextlib import contextmanager

class DualResponseCache:
    """
    Thread-safe file-based cache for dual LLM responses with expiry and atomic operations.
    """
    
    def __init__(self, cache_dir: str = "textbooks/cache", expiry_hours: int = 24):
        self.logger = logging.getLogger("quillai_cache")
        self.cache_dir = cache_dir
        self.expiry_seconds = expiry_hours * 3600
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'errors': 0
        }
        
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.logger.info(f"Initialized DualResponseCache at {self.cache_dir} with {expiry_hours}-hour expiry.")
        except Exception as e:
            self.logger.error(f"Failed to create cache directory {self.cache_dir}: {e}")
            raise
        
        # Perform initial cleanup on startup
        self._cleanup_expired()
    
    @contextmanager
    def _file_lock(self, file_path, mode='r'):
        """Context manager for file locking."""
        lock_file = file_path + '.lock'
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            try:
                with open(lock_file, 'w') as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    try:
                        with open(file_path, mode) as f:
                            yield f
                    except FileNotFoundError:
                        if 'r' in mode:
                            yield None
                        else:
                            with open(file_path, mode) as f:
                                yield f
            except (OSError, PermissionError) as e:
                self.logger.warning(f"File locking failed for {file_path}: {e}. Proceeding without lock.")
                # Fallback without locking
                try:
                    with open(file_path, mode) as f:
                        yield f
                except FileNotFoundError:
                    if 'r' in mode:
                        yield None
                    else:
                        with open(file_path, mode) as f:
                            yield f
        finally:
            try:
                os.remove(lock_file)
            except (FileNotFoundError, PermissionError):
                pass
    
    def _get_cache_key(self, query: str, mode: str, marks: Optional[int]) -> str:
        """Generates a unique cache key based on query, mode, and marks."""
        # Normalize query for consistent caching
        normalized_query = query.strip().lower().replace(" ", "_")
        key_parts = [normalized_query, mode]
        if marks is not None:
            key_parts.append(str(marks))
        return "_".join(key_parts) + ".json"
    
    def get(self, query: str, mode: str, marks: Optional[int]) -> Optional[Dict[str, Any]]:
        """Retrieves a cached response if available and not expired."""
        with self._lock:
            cache_key = self._get_cache_key(query, mode, marks)
            file_path = os.path.join(self.cache_dir, cache_key)
            
            try:
                with self._file_lock(file_path, 'r') as f:
                    if f is None:
                        self._stats['misses'] += 1
                        return None
                    
                    cached_data = json.load(f)
                
                cache_timestamp = datetime.fromisoformat(cached_data['timestamp'].replace('Z', '+00:00'))
                if datetime.utcnow() - cache_timestamp < timedelta(seconds=self.expiry_seconds):
                    # Verify word counts for question mode
                    if mode == "question" and marks is not None:
                        target_words = {2: 100, 5: 250, 10: 500}.get(marks)
                        
                        # Use the API response keys for word count validation
                        llm_output_wc = len(cached_data['response'].get('dialogpt_output', '').split())
                        custom_output_wc = len(cached_data['response'].get('custom_llm', '').split())
                        
                        # Allow a small tolerance for cached word counts
                        if target_words and (
                            abs(llm_output_wc - min(target_words, 500)) / min(target_words, 500) > 0.1 or
                            abs(custom_output_wc - target_words) / target_words > 0.1
                        ):
                            self.logger.info(f"Cache miss for {cache_key}: Word count mismatch. Recalculating.")
                            self.delete(query, mode, marks)  # Invalidate cache
                            self._stats['misses'] += 1
                            return None
                    
                    self.logger.info(f"Cache hit for {cache_key}")
                    self._stats['hits'] += 1
                    return cached_data['response']
                else:
                    self.logger.info(f"Cache miss for {cache_key}: Expired. Deleting.")
                    self.delete(query, mode, marks)
                    self._stats['misses'] += 1
            except Exception as e:
                self.logger.error(f"Error reading cache file {file_path}: {e}. Deleting corrupted entry.")
                self.delete(query, mode, marks)
                self._stats['errors'] += 1
            
            return None
    
    def set(self, query: str, mode: str, marks: Optional[int], response: Dict[str, Any]):
        """Stores a response in the cache with atomic write operations."""
        with self._lock:
            cache_key = self._get_cache_key(query, mode, marks)
            file_path = os.path.join(self.cache_dir, cache_key)
            temp_file = file_path + '.tmp'
            
            try:
                data_to_cache = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'query': query,
                    'mode': mode,
                    'marks': marks,
                    'response': response
                }
                
                # Write to temp file first
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data_to_cache, f, indent=2, ensure_ascii=False)
                
                # Atomic move to final location
                os.replace(temp_file, file_path)
                
                self.logger.info(f"Cached response for {cache_key}")
            except Exception as e:
                self.logger.error(f"Error writing cache file {file_path}: {e}")
                self._stats['errors'] += 1
                # Clean up temp file if it exists
                try:
                    os.remove(temp_file)
                except FileNotFoundError:
                    pass
    
    def delete(self, query: str, mode: str, marks: Optional[int]):
        """Deletes a specific cache entry."""
        with self._lock:
            cache_key = self._get_cache_key(query, mode, marks)
            file_path = os.path.join(self.cache_dir, cache_key)
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.info(f"Deleted cache entry: {cache_key}")
                except Exception as e:
                    self.logger.error(f"Error deleting cache file {file_path}: {e}")
                    self._stats['errors'] += 1
    
    def clear_all(self):
        """Clear all cache entries."""
        with self._lock:
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(self.cache_dir, filename)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            self.logger.error(f"Error deleting cache file {file_path}: {e}")
                
                self.logger.info("Cleared all cache entries")
                self._stats['invalidations'] += 1
            except Exception as e:
                self.logger.error(f"Error clearing cache: {e}")
                self._stats['errors'] += 1
    
    def _cleanup_expired(self):
        """Removes expired cache entries."""
        with self._lock:
            now = datetime.utcnow()
            try:
                filenames = os.listdir(self.cache_dir)
            except (OSError, FileNotFoundError):
                return
            
            for filename in filenames:
                if filename.endswith(".json"):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with self._file_lock(file_path, 'r') as f:
                            if f is None:
                                continue
                            cached_data = json.load(f)
                        
                        cache_timestamp = datetime.fromisoformat(cached_data['timestamp'].replace('Z', '+00:00'))
                        if now - cache_timestamp > timedelta(seconds=self.expiry_seconds):
                            os.remove(file_path)
                            self.logger.info(f"Cleaned up expired cache entry: {filename}")
                    except Exception as e:
                        self.logger.error(f"Error during cache cleanup for {filename}: {e}. Deleting corrupted entry.")
                        try:
                            os.remove(file_path)
                        except OSError as oe:
                            self.logger.error(f"Failed to remove corrupted file {file_path}: {oe}")
    
    def get_stats(self):
        """Get cache statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats['total_entries'] = len([f for f in os.listdir(self.cache_dir) if f.endswith('.json')])
            stats['hit_rate'] = stats['hits'] / max(stats['hits'] + stats['misses'], 1) * 100
            stats['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            return stats
