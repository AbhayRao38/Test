import json
import os
import time
import fcntl
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import threading
import logging
from contextlib import contextmanager
import hashlib

# Environment variables used:
# - CACHE_DIR: Cache directory path (default: textbooks/cache)
# - CACHE_EXPIRY_HOURS: Cache expiry time in hours (default: 24)

class DualResponseCache:
    """
    Thread-safe file-based cache for dual LLM responses with comprehensive expiry management,
    word count validation, atomic operations, and robust error handling.
    """
    
    def __init__(self, cache_dir: str = None, expiry_hours: int = None):
        self.logger = logging.getLogger("quillai_cache")
        
        # Use environment variables with fallbacks
        self.cache_dir = cache_dir or os.getenv("CACHE_DIR", "textbooks/cache")
        self.expiry_hours = expiry_hours or int(os.getenv("CACHE_EXPIRY_HOURS", "24"))
        self.expiry_seconds = self.expiry_hours * 3600
        
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'errors': 0,
            'word_count_mismatches': 0,
            'expired_entries': 0,
            'corrupted_entries': 0
        }
        
        # Create cache directory with proper permissions
        try:
            os.makedirs(self.cache_dir, mode=0o755, exist_ok=True)
            self.logger.info(f"✓ Cache initialized at {self.cache_dir} with {self.expiry_hours}-hour expiry")
        except Exception as e:
            self.logger.error(f"Failed to create cache directory {self.cache_dir}: {e}")
            raise
        
        # Cache configuration validation
        self._validate_cache_configuration()
        
        # Perform initial cleanup on startup
        self._cleanup_expired_entries()
        
        # Initialize metadata tracking
        self._metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        self._load_metadata()
    
    def _validate_cache_configuration(self):
        """Validate cache configuration and settings."""
        if self.expiry_hours < 1:
            raise ValueError("Cache expiry must be at least 1 hour")
        
        if self.expiry_hours > 168:  # 7 days
            self.logger.warning(f"Long cache expiry time: {self.expiry_hours} hours")
        
        # Test write permissions
        test_file = os.path.join(self.cache_dir, "test_write_permissions.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            raise PermissionError(f"No write permissions for cache directory {self.cache_dir}: {e}")
    
    @contextmanager
    def _file_lock(self, file_path, mode='r'):
        """Context manager for robust file locking with comprehensive error handling."""
        lock_file = file_path + '.lock'
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            try:
                with open(lock_file, 'w') as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    try:
                        if 'w' in mode:
                            # For write operations, use atomic write pattern
                            temp_file = file_path + '.tmp'
                            with open(temp_file, mode) as f:
                                yield f
                            # Atomic move after successful write
                            os.replace(temp_file, file_path)
                        else:
                            # For read operations
                            try:
                                with open(file_path, mode) as f:
                                    yield f
                            except FileNotFoundError:
                                yield None
                    except Exception as e:
                        # Clean up temp file if write failed
                        if 'w' in mode:
                            temp_file = file_path + '.tmp'
                            try:
                                os.remove(temp_file)
                            except FileNotFoundError:
                                pass
                        raise
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
    
    def _load_metadata(self):
        """Load cache metadata for tracking and analytics."""
        self._metadata = {
            'created': datetime.utcnow().isoformat() + 'Z',
            'last_cleanup': None,
            'total_entries_created': 0,
            'total_entries_expired': 0
        }
        
        try:
            with self._file_lock(self._metadata_file, 'r') as f:
                if f:
                    stored_metadata = json.load(f)
                    self._metadata.update(stored_metadata)
        except Exception as e:
            self.logger.warning(f"Could not load cache metadata: {e}")
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with self._file_lock(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save cache metadata: {e}")
    
    def _get_cache_key(self, query: str, mode: str, marks: Optional[int]) -> str:
        """Generate a unique, filesystem-safe cache key with collision resistance."""
        # Create a comprehensive key string
        key_parts = [
            query.strip().lower(),
            mode.lower(),
            str(marks) if marks is not None else "none"
        ]
        key_string = "|".join(key_parts)
        
        # Use SHA-256 hash for collision resistance and filesystem safety
        key_hash = hashlib.sha256(key_string.encode('utf-8')).hexdigest()
        
        # Include readable prefix for debugging
        readable_prefix = query.replace(" ", "_")[:20]
        readable_prefix = "".join(c for c in readable_prefix if c.isalnum() or c == "_")
        
        return f"{readable_prefix}_{key_hash[:16]}.json"
    
    def _validate_word_counts(self, cached_data: Dict[str, Any], mode: str, marks: Optional[int]) -> bool:
        """Validate cached response word counts against current requirements."""
        if mode != "question" or marks is None:
            return True  # No validation needed for learning mode
        
        target_words = {2: 100, 5: 250, 10: 500}.get(marks)
        if not target_words:
            return True
        
        response = cached_data.get('response', {})
        
        # Check DialogGPT output (allow some flexibility)
        dialogpt_words = len(response.get('dialogpt_output', '').split())
        dialogpt_target = min(target_words, 400)  # DialogGPT is capped at 400 words
        dialogpt_valid = abs(dialogpt_words - dialogpt_target) / max(dialogpt_target, 1) <= 0.2
        
        # Check Custom output (must be more precise)
        custom_words = len(response.get('custom_llm', '').split())
        custom_valid = abs(custom_words - target_words) / max(target_words, 1) <= 0.15
        
        if not (dialogpt_valid and custom_valid):
            self._stats['word_count_mismatches'] += 1
            self.logger.debug(f"Word count mismatch - DialogGPT: {dialogpt_words}/{dialogpt_target}, Custom: {custom_words}/{target_words}")
            return False
        
        return True
    
    def _validate_response_quality(self, cached_data: Dict[str, Any]) -> bool:
        """Validate cached response quality and completeness."""
        response = cached_data.get('response', {})
        
        # Check required fields
        required_fields = ['dialogpt_output', 'custom_llm', 'success', 'timestamp']
        if not all(field in response for field in required_fields):
            self._stats['corrupted_entries'] += 1
            return False
        
        # Check output quality
        dialogpt_output = response.get('dialogpt_output', '')
        custom_output = response.get('custom_llm', '')
        
        # Both outputs must have minimum content
        if len(dialogpt_output.strip()) < 20 or len(custom_output.strip()) < 20:
            self._stats['corrupted_entries'] += 1
            return False
        
        # Check for error messages (should not be cached)
        error_indicators = [
            'i apologize', 'please rephrase', 'unable to generate',
            'error occurred', 'failed to', 'cannot provide'
        ]
        
        for output in [dialogpt_output.lower(), custom_output.lower()]:
            if any(indicator in output for indicator in error_indicators):
                return False
        
        return True
    
    def get(self, query: str, mode: str, marks: Optional[int]) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response with comprehensive validation."""
        with self._lock:
            cache_key = self._get_cache_key(query, mode, marks)
            file_path = os.path.join(self.cache_dir, cache_key)
            
            try:
                with self._file_lock(file_path, 'r') as f:
                    if f is None:
                        self._stats['misses'] += 1
                        return None
                    
                    cached_data = json.load(f)
                
                # Validate timestamp and expiry
                try:
                    cache_timestamp = datetime.fromisoformat(cached_data['timestamp'].replace('Z', '+00:00'))
                    if datetime.utcnow() - cache_timestamp >= timedelta(seconds=self.expiry_seconds):
                        self.logger.debug(f"Cache entry expired: {cache_key}")
                        self.delete(query, mode, marks)
                        self._stats['misses'] += 1
                        self._stats['expired_entries'] += 1
                        return None
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Invalid timestamp in cache entry {cache_key}: {e}")
                    self.delete(query, mode, marks)
                    self._stats['misses'] += 1
                    self._stats['corrupted_entries'] += 1
                    return None
                
                # Validate response quality
                if not self._validate_response_quality(cached_data):
                    self.logger.debug(f"Cache entry quality validation failed: {cache_key}")
                    self.delete(query, mode, marks)
                    self._stats['misses'] += 1
                    return None
                
                # Validate word counts for question mode
                if not self._validate_word_counts(cached_data, mode, marks):
                    self.logger.debug(f"Cache entry word count validation failed: {cache_key}")
                    self.delete(query, mode, marks)
                    self._stats['misses'] += 1
                    return None
                
                self.logger.debug(f"Cache hit: {cache_key}")
                self._stats['hits'] += 1
                return cached_data['response']
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Corrupted cache file {file_path}: {e}")
                self.delete(query, mode, marks)
                self._stats['errors'] += 1
                self._stats['corrupted_entries'] += 1
            except Exception as e:
                self.logger.error(f"Error reading cache file {file_path}: {e}")
                self._stats['errors'] += 1
            
            self._stats['misses'] += 1
            return None
    
    def set(self, query: str, mode: str, marks: Optional[int], response: Dict[str, Any]):
        """Store a response in the cache with comprehensive validation and atomic operations."""
        with self._lock:
            # Validate response before caching
            if not self._should_cache_response(response):
                self.logger.debug("Response not suitable for caching")
                return
            
            cache_key = self._get_cache_key(query, mode, marks)
            file_path = os.path.join(self.cache_dir, cache_key)
            
            try:
                # Prepare data for caching
                data_to_cache = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'query': query,
                    'mode': mode,
                    'marks': marks,
                    'expiry_hours': self.expiry_hours,
                    'response': response,
                    'cache_version': '1.0'
                }
                
                # Atomic write operation
                with self._file_lock(file_path, 'w') as f:
                    json.dump(data_to_cache, f, indent=2, ensure_ascii=False)
                
                self.logger.debug(f"Cached response: {cache_key}")
                self._metadata['total_entries_created'] += 1
                self._save_metadata()
                
            except Exception as e:
                self.logger.error(f"Error caching response to {file_path}: {e}")
                self._stats['errors'] += 1
    
    def _should_cache_response(self, response: Dict[str, Any]) -> bool:
        """Determine if a response should be cached based on quality criteria."""
        # Check if response indicates success
        if not response.get('success', False):
            return False
        
        # Check for minimum content in both outputs
        dialogpt_output = response.get('dialogpt_output', '')
        custom_output = response.get('custom_llm', '')
        
        if len(dialogpt_output.strip()) < 30 or len(custom_output.strip()) < 50:
            return False
        
        # Check for error indicators
        error_indicators = [
            'i apologize', 'please rephrase', 'unable to generate',
            'error occurred', 'failed to', 'cannot provide'
        ]
        
        for output in [dialogpt_output.lower(), custom_output.lower()]:
            if any(indicator in output for indicator in error_indicators):
                return False
        
        return True
    
    def delete(self, query: str, mode: str, marks: Optional[int]):
        """Delete a specific cache entry with error handling."""
        with self._lock:
            cache_key = self._get_cache_key(query, mode, marks)
            file_path = os.path.join(self.cache_dir, cache_key)
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.debug(f"Deleted cache entry: {cache_key}")
                except Exception as e:
                    self.logger.error(f"Error deleting cache file {file_path}: {e}")
                    self._stats['errors'] += 1
    
    def clear_all(self):
        """Clear all cache entries with comprehensive cleanup."""
        with self._lock:
            cleared_count = 0
            error_count = 0
            
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json') and filename != "cache_metadata.json":
                        file_path = os.path.join(self.cache_dir, filename)
                        try:
                            os.remove(file_path)
                            cleared_count += 1
                        except Exception as e:
                            self.logger.error(f"Error deleting cache file {file_path}: {e}")
                            error_count += 1
                
                self.logger.info(f"Cleared {cleared_count} cache entries ({error_count} errors)")
                self._stats['invalidations'] += 1
                
                # Reset metadata
                self._metadata['total_entries_expired'] += cleared_count
                self._save_metadata()
                
            except Exception as e:
                self.logger.error(f"Error clearing cache: {e}")
                self._stats['errors'] += 1
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries and perform maintenance."""
        with self._lock:
            now = datetime.utcnow()
            cleaned_count = 0
            error_count = 0
            
            try:
                filenames = os.listdir(self.cache_dir)
            except (OSError, FileNotFoundError):
                return
            
            for filename in filenames:
                if filename.endswith(".json") and filename != "cache_metadata.json":
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with self._file_lock(file_path, 'r') as f:
                            if f is None:
                                continue
                            cached_data = json.load(f)
                        
                        # Check expiry
                        cache_timestamp = datetime.fromisoformat(cached_data['timestamp'].replace('Z', '+00:00'))
                        if now - cache_timestamp >= timedelta(seconds=self.expiry_seconds):
                            os.remove(file_path)
                            cleaned_count += 1
                            self.logger.debug(f"Cleaned expired entry: {filename}")
                    except json.JSONDecodeError:
                        # Remove corrupted files
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                            self.logger.debug(f"Removed corrupted entry: {filename}")
                        except OSError:
                            error_count += 1
                    except Exception as e:
                        self.logger.error(f"Error during cleanup of {filename}: {e}")
                        error_count += 1
            
            if cleaned_count > 0:
                self.logger.info(f"Cleanup completed: {cleaned_count} entries removed ({error_count} errors)")
                self._metadata['last_cleanup'] = now.isoformat() + 'Z'
                self._metadata['total_entries_expired'] += cleaned_count
                self._save_metadata()
    
    def get_stats(self):
        """Get comprehensive cache statistics and health metrics."""
        with self._lock:
            stats = self._stats.copy()
            
            # Calculate additional metrics
            total_requests = stats['hits'] + stats['misses']
            stats['hit_rate'] = (stats['hits'] / max(total_requests, 1)) * 100
            stats['total_requests'] = total_requests
            
            # Count current entries
            try:
                current_entries = len([f for f in os.listdir(self.cache_dir) 
                                     if f.endswith('.json') and f != 'cache_metadata.json'])
            except OSError:
                current_entries = 0
            
            stats['current_entries'] = current_entries
            stats['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            stats['cache_dir'] = self.cache_dir
            stats['expiry_hours'] = self.expiry_hours
            
            # Add metadata
            stats['metadata'] = self._metadata.copy()
            
            return stats
    
    def print_detailed_stats(self):
        """Print comprehensive cache statistics."""
        stats = self.get_stats()
        
        print("📊 Cache Statistics:")
        print("=" * 40)
        print(f"�� Cache directory: {stats['cache_dir']}")
        print(f"⏰ Expiry time: {stats['expiry_hours']} hours")
        print(f"📄 Current entries: {stats['current_entries']}")
        print(f"🎯 Hit rate: {stats['hit_rate']:.1f}%")
        print(f"✅ Cache hits: {stats['hits']}")
        print(f"❌ Cache misses: {stats['misses']}")
        print(f"🧹 Invalidations: {stats['invalidations']}")
        print(f"⚠️ Errors: {stats['errors']}")
        print(f"📏 Word count mismatches: {stats['word_count_mismatches']}")
        print(f"⏳ Expired entries: {stats['expired_entries']}")
        print(f"💔 Corrupted entries: {stats['corrupted_entries']}")
        
        metadata = stats['metadata']
        if metadata.get('last_cleanup'):
            print(f"🧹 Last cleanup: {metadata['last_cleanup']}")
        print(f"📈 Total entries created: {metadata['total_entries_created']}")
        print(f"🗑️ Total entries expired: {metadata['total_entries_expired']}")
    
    def perform_maintenance(self):
        """Perform comprehensive cache maintenance."""
        self.logger.info("Starting cache maintenance...")
        
        with self._lock:
            # Clean expired entries
            self._cleanup_expired_entries()
            
            # Update metadata
            self._metadata['last_maintenance'] = datetime.utcnow().isoformat() + 'Z'
            self._save_metadata()
        
        self.logger.info("Cache maintenance completed")


# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Dual Response Cache Test")
    print("=" * 50)
    
    # Initialize cache
    cache = DualResponseCache(expiry_hours=24)
    
    # Print current stats
    cache.print_detailed_stats()
    
    print("\n✅ Dual Response Cache ready for use!")
    print("Features:")
    print("  • Thread-safe operations")
    print("  • Atomic file operations")
    print("  • Word count validation")
    print("  • Quality assessment")
    print("  • Comprehensive statistics")
    print("  • Automatic cleanup")
