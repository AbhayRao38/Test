import os
import fitz  # PyMuPDF
import shutil
import pytesseract
from PIL import Image
import io
import re
from datetime import datetime
import hashlib
import json
import threading
import fcntl
from contextlib import contextmanager

# Environment variables used:
# - TESSERACT_CMD: Path to tesseract executable (optional)

class KnowledgeBaseManager:
    """
    Enhanced knowledge base manager with OCR support, validation, robust error handling,
    thread-safe operations, comprehensive duplicate detection, and academic text quality filtering.
    """
    
    def __init__(self, storage_dir="textbooks"):
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self.supported_languages = ['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'chi_sim', 'jpn']
        self._lock = threading.RLock()
        
        # Create storage directory with proper permissions
        try:
            os.makedirs(self.storage_dir, mode=0o755, exist_ok=True)
            print(f"✓ Knowledge base directory ready: {self.storage_dir}")
        except Exception as e:
            print(f"⚠ Warning: Could not create storage directory {self.storage_dir}: {e}")
            raise
        
        # Initialize metadata with thread safety
        self.metadata = self._load_metadata()
        
        # Check OCR availability and configure
        self.ocr_available = self._check_and_configure_ocr()
        
        # Academic text quality filters
        self.academic_indicators = [
            'definition', 'concept', 'theory', 'principle', 'method',
            'algorithm', 'approach', 'technique', 'framework', 'model',
            'analysis', 'research', 'study', 'investigation', 'examination',
            'application', 'implementation', 'example', 'instance', 'case'
        ]
    
    @contextmanager
    def _file_lock(self, file_path, mode='r'):
        """Context manager for robust file locking with fallback."""
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
                print(f"⚠ Warning: File locking failed for {file_path}: {e}. Proceeding without lock.")
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
    
    def _check_and_configure_ocr(self):
        """Check if OCR (Tesseract) is available and configure it."""
        try:
            # Check for custom tesseract path
            tesseract_cmd = os.environ.get('TESSERACT_CMD')
            if tesseract_cmd and os.path.exists(tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
            # Test OCR availability
            version = pytesseract.get_tesseract_version()
            print(f"✓ OCR (Tesseract) available: v{version}")
            return True
        except Exception as e:
            print(f"⚠ OCR not available: {e}")
            print("  Install tesseract-ocr for scanned PDF support")
            return False
    
    def _load_metadata(self):
        """Load metadata from file with comprehensive error handling."""
        with self._lock:
            try:
                with self._file_lock(self.metadata_file, 'r') as f:
                    if f:
                        metadata = json.load(f)
                        # Validate and clean metadata structure
                        return self._validate_metadata_structure(metadata)
            except Exception as e:
                print(f"Warning: Could not load metadata from {self.metadata_file}: {e}")
            return {}
    
    def _validate_metadata_structure(self, metadata):
        """Validate and clean metadata structure."""
        if not isinstance(metadata, dict):
            return {}
        
        validated_metadata = {}
        for filename, meta in metadata.items():
            if isinstance(meta, dict) and self._is_valid_metadata_entry(meta):
                validated_metadata[filename] = meta
        
        return validated_metadata
    
    def _is_valid_metadata_entry(self, meta):
        """Check if metadata entry has required structure."""
        required_fields = ['added_date', 'file_size', 'page_count', 'hash']
        return all(field in meta for field in required_fields)
    
    def _save_metadata(self):
        """Save metadata to file with atomic writes and comprehensive error handling."""
        with self._lock:
            try:
                temp_file = self.metadata_file + '.tmp'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                
                # Atomic move
                os.replace(temp_file, self.metadata_file)
                print(f"✓ Metadata saved to {self.metadata_file}")
            except Exception as e:
                print(f"❌ Error: Could not save metadata to {self.metadata_file}: {e}")
                # Clean up temp file if it exists
                try:
                    os.remove(temp_file)
                except FileNotFoundError:
                    pass
                raise
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of file for comprehensive duplicate detection."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Could not calculate hash for {file_path}: {e}")
            return None
    
    def _validate_pdf_comprehensive(self, pdf_path):
        """Comprehensive PDF validation with detailed diagnostics and quality assessment."""
        validation_result = {
            'valid': False,
            'file_size': 0,
            'page_count': 0,
            'has_text': False,
            'has_images': False,
            'text_quality': 'unknown',
            'academic_content': False,
            'language': 'unknown',
            'warnings': [],
            'errors': [],
            'academic_score': 0
        }
        
        try:
            # Check file existence and basic properties
            if not os.path.isfile(pdf_path):
                validation_result['errors'].append(f"File not found: {pdf_path}")
                return validation_result
            
            file_size = os.path.getsize(pdf_path)
            validation_result['file_size'] = file_size
            
            if file_size == 0:
                validation_result['errors'].append("File is empty")
                return validation_result
            
            if file_size > 200 * 1024 * 1024:  # 200MB limit
                validation_result['warnings'].append(f"Very large file ({file_size / (1024*1024):.1f}MB) may cause performance issues")
            elif file_size > 50 * 1024 * 1024:  # 50MB warning
                validation_result['warnings'].append(f"Large file ({file_size / (1024*1024):.1f}MB)")
            
            # Validate PDF structure and content
            with fitz.open(pdf_path) as doc:
                validation_result['page_count'] = len(doc)
                
                if len(doc) == 0:
                    validation_result['errors'].append("PDF has no pages")
                    return validation_result
                
                # Analyze content across multiple pages
                text_pages = 0
                image_pages = 0
                total_text_length = 0
                academic_terms_found = set()
                sample_text_blocks = []
                
                # Sample pages for comprehensive analysis
                sample_pages = min(10, len(doc))
                page_indices = [int(i * len(doc) / sample_pages) for i in range(sample_pages)]
                
                for page_num in page_indices:
                    try:
                        page = doc[page_num]
                        
                        # Extract and analyze text
                        text = page.get_text()
                        if text.strip():
                            text_pages += 1
                            total_text_length += len(text)
                            sample_text_blocks.append(text[:1000])  # Sample first 1000 chars
                            
                            # Check for academic terms
                            text_lower = text.lower()
                            for term in self.academic_indicators:
                                if term in text_lower:
                                    academic_terms_found.add(term)
                        
                        # Check for images
                        image_list = page.get_images()
                        if image_list:
                            image_pages += 1
                            
                    except Exception as e:
                        validation_result['warnings'].append(f"Could not analyze page {page_num + 1}: {e}")
                        continue
                
                validation_result['has_text'] = text_pages > 0
                validation_result['has_images'] = image_pages > 0
                validation_result['academic_score'] = len(academic_terms_found)
                validation_result['academic_content'] = len(academic_terms_found) >= 3
                
                # Assess text quality
                if total_text_length > 1000:
                    if text_pages / sample_pages > 0.8:
                        validation_result['text_quality'] = 'high'
                    elif text_pages / sample_pages > 0.5:
                        validation_result['text_quality'] = 'medium'
                    else:
                        validation_result['text_quality'] = 'low'
                
                # Determine if PDF is primarily scanned
                if image_pages > text_pages and total_text_length < 500:
                    validation_result['warnings'].append("PDF appears to be scanned - OCR will be required")
                    validation_result['text_quality'] = 'scanned'
                
                # Language detection with improved accuracy
                if total_text_length > 200:
                    combined_sample = ' '.join(sample_text_blocks)[:2000]
                    validation_result['language'] = self._detect_language_enhanced(combined_sample)
                
                # Overall validation
                if validation_result['has_text'] or (validation_result['has_images'] and self.ocr_available):
                    validation_result['valid'] = True
                else:
                    validation_result['errors'].append("PDF has no extractable text and OCR is not available")
                
        except Exception as e:
            validation_result['errors'].append(f"PDF validation failed: {str(e)}")
            print(f"PDF validation error for {pdf_path}: {e}")
        
        return validation_result
    
    def _detect_language_enhanced(self, text):
        """Enhanced language detection with multiple indicators."""
        text_lower = text.lower()
        
        # Language-specific word patterns with weights
        language_indicators = {
            'eng': (['the', 'and', 'or', 'of', 'in', 'to', 'a', 'is', 'that', 'for'], 3),
            'fra': (['le', 'la', 'et', 'de', 'un', 'une', 'est', 'dans', 'pour', 'avec'], 3),
            'deu': (['der', 'die', 'das', 'und', 'oder', 'ist', 'in', 'zu', 'auf', 'mit'], 3),
            'spa': (['el', 'la', 'y', 'o', 'de', 'un', 'una', 'es', 'en', 'para'], 3),
            'ita': (['il', 'la', 'e', 'o', 'di', 'un', 'una', 'è', 'in', 'per'], 3),
            'por': (['o', 'a', 'e', 'ou', 'de', 'um', 'uma', 'é', 'em', 'para'], 3),
            'rus': (['и', 'в', 'не', 'на', 'с', 'что', 'как', 'по', 'для', 'это'], 2),
        }
        
        scores = {}
        words = text_lower.split()[:500]  # Analyze first 500 words
        
        for lang, (indicators, weight) in language_indicators.items():
            score = sum(weight for word in words if word in indicators)
            if score > 0:
                scores[lang] = score
        
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'eng'  # Default to English
    
    def is_duplicate(self, pdf_path):
        """Comprehensive duplicate detection using hash and metadata comparison."""
        file_hash = self._calculate_file_hash(pdf_path)
        if not file_hash:
            return False
        
        filename = os.path.basename(pdf_path)
        
        # Check exact file match (same name and hash)
        if filename in self.metadata:
            existing_hash = self.metadata[filename].get('hash')
            if existing_hash == file_hash:
                extraction_method = self.metadata[filename].get('extraction_method')
                if extraction_method and extraction_method != 'pending':
                    return True
        
        # Check for same content with different filename
        for existing_filename, meta in self.metadata.items():
            if meta.get('hash') == file_hash:
                extraction_method = meta.get('extraction_method')
                if extraction_method and extraction_method != 'pending':
                    print(f"⚠ Content duplicate found: {filename} matches {existing_filename}")
                    return True
        
        return False
    
    def add_pdf(self, pdf_path, force_ocr=False, language='eng'):
        """Enhanced PDF addition with comprehensive validation, quality assessment, and robust error handling."""
        with self._lock:
            print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
            
            # Comprehensive validation
            validation = self._validate_pdf_comprehensive(pdf_path)
            
            if not validation['valid']:
                error_msg = f"PDF validation failed: {'; '.join(validation['errors'])}"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # Display comprehensive validation results
            print(f"   📊 Validation Results:")
            print(f"      File size: {validation['file_size'] / (1024*1024):.1f}MB")
            print(f"      Pages: {validation['page_count']}")
            print(f"      Text quality: {validation['text_quality']}")
            print(f"      Academic content: {'Yes' if validation['academic_content'] else 'No'}")
            print(f"      Academic score: {validation['academic_score']}/10")
            print(f"      Language: {validation['language']}")
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    print(f"      ⚠ {warning}")
            
            # Check for duplicates with detailed reporting
            if self.is_duplicate(pdf_path):
                print("⚠ Duplicate content detected - skipping")
                return
            
            # Determine storage path
            filename = os.path.basename(pdf_path)
            dest_path = os.path.join(self.storage_dir, filename)
            
            # Handle filename conflicts
            if os.path.exists(dest_path) and pdf_path != dest_path:
                base_name, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest_path):
                    filename = f"{base_name}_{counter}{ext}"
                    dest_path = os.path.join(self.storage_dir, filename)
                    counter += 1
                print(f"   📝 Renamed to avoid conflict: {filename}")
            
            # Copy PDF to storage
            try:
                if pdf_path != dest_path:
                    shutil.copy2(pdf_path, dest_path)  # Preserve metadata
                    print(f"✓ PDF copied to knowledge base: {filename}")
                else:
                    print(f"✓ PDF already in knowledge base: {filename}")
            except Exception as e:
                error_msg = f"Failed to copy PDF to {dest_path}: {e}"
                print(f"❌ {error_msg}")
                raise
            
            # Calculate file hash for the stored file
            file_hash = self._calculate_file_hash(dest_path)
            
            # Store comprehensive metadata
            self.metadata[filename] = {
                'original_path': pdf_path,
                'storage_path': dest_path,
                'added_date': datetime.utcnow().isoformat(),
                'file_size': validation['file_size'],
                'page_count': validation['page_count'],
                'has_text': validation['has_text'],
                'has_images': validation['has_images'],
                'text_quality': validation['text_quality'],
                'academic_content': validation['academic_content'],
                'academic_score': validation['academic_score'],
                'language': validation['language'],
                'hash': file_hash,
                'extraction_method': 'pending',
                'force_ocr_requested': force_ocr,
                'ocr_language': language if force_ocr else validation['language']
            }
            
            self._save_metadata()
            print(f"✓ PDF successfully added to knowledge base with comprehensive metadata")
    
    def extract_text(self, pdf_filename, use_ocr=False, language='eng', clean_text=True):
        """Enhanced text extraction with OCR fallback, quality assessment, and academic filtering."""
        pdf_path = os.path.join(self.storage_dir, pdf_filename)
        
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found in knowledge base: {pdf_filename}")
        
        print(f"📖 Extracting text from: {pdf_filename}")
        
        # Check OCR availability if requested
        if use_ocr and not self.ocr_available:
            print("⚠ OCR requested but not available - falling back to standard extraction")
            use_ocr = False
        
        text_blocks = []
        extraction_stats = {
            'pages_processed': 0,
            'pages_with_text': 0,
            'pages_with_ocr': 0,
            'total_characters': 0,
            'academic_content_ratio': 0,
            'extraction_method': 'standard' if not use_ocr else 'ocr',
            'quality_score': 0
        }
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    extraction_stats['pages_processed'] += 1
                    page_text = ""
                    
                    if use_ocr:
                        # OCR extraction
                        page_text = self._extract_text_with_ocr(page, language)
                        if page_text.strip():
                            extraction_stats['pages_with_ocr'] += 1
                    else:
                        # Standard text extraction
                        page_text = page.get_text()
                        if page_text.strip():
                            extraction_stats['pages_with_text'] += 1
                        
                        # Automatic OCR fallback for pages with no text
                        elif self.ocr_available and not page_text.strip():
                            print(f"   Page {page_num + 1}: No text found, trying OCR...")
                            ocr_text = self._extract_text_with_ocr(page, language)
                            if ocr_text.strip():
                                page_text = ocr_text
                                extraction_stats['pages_with_ocr'] += 1
                                extraction_stats['extraction_method'] = 'hybrid'
                    
                    # Process extracted text
                    if page_text.strip():
                        if clean_text:
                            page_text = self._clean_extracted_text_enhanced(page_text)
                        
                        # Filter for academic quality
                        if self._is_academic_quality_text(page_text):
                            text_blocks.append(page_text)
                            extraction_stats['total_characters'] += len(page_text)
                    
                    # Progress indicator for large documents
                    if extraction_stats['pages_processed'] % 20 == 0:
                        print(f"   Processed {extraction_stats['pages_processed']} pages...")
        
        except Exception as e:
            error_msg = f"Text extraction failed for {pdf_filename}: {e}"
            print(f"❌ {error_msg}")
            raise
        
        # Calculate quality metrics
        if text_blocks:
            combined_text = "\n".join(text_blocks)
            extraction_stats['academic_content_ratio'] = self._calculate_academic_content_ratio(combined_text)
            extraction_stats['quality_score'] = self._calculate_text_quality_score(combined_text)
        
        # Update metadata with extraction results
        if pdf_filename in self.metadata:
            self.metadata[pdf_filename].update({
                'extraction_method': extraction_stats['extraction_method'],
                'extraction_stats': extraction_stats,
                'last_extracted': datetime.utcnow().isoformat(),
                'text_quality_score': extraction_stats['quality_score']
            })
            self._save_metadata()
        
        # Display comprehensive extraction results
        print(f"✓ Text extraction completed:")
        print(f"   📊 Statistics:")
        print(f"      Pages processed: {extraction_stats['pages_processed']}")
        print(f"      Pages with text: {extraction_stats['pages_with_text']}")
        if extraction_stats['pages_with_ocr'] > 0:
            print(f"      Pages with OCR: {extraction_stats['pages_with_ocr']}")
        print(f"      Total characters: {extraction_stats['total_characters']:,}")
        print(f"      Academic content: {extraction_stats['academic_content_ratio']:.1f}%")
        print(f"      Quality score: {extraction_stats['quality_score']}/10")
        print(f"      Method: {extraction_stats['extraction_method']}")
        
        if not text_blocks:
            print("⚠ No high-quality academic text extracted from PDF")
            if not use_ocr and self.ocr_available:
                print("💡 Try using OCR: extract_text(filename, use_ocr=True)")
        
        combined_text = "\n".join(text_blocks)
        return combined_text
    
    def _extract_text_with_ocr(self, page, language='eng'):
        """Extract text from a PDF page using OCR with enhanced configuration."""
        try:
            # Convert page to high-resolution image for better OCR accuracy
            mat = fitz.Matrix(3, 3)  # 3x zoom for better OCR accuracy
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Enhanced OCR configuration
            custom_config = f'--oem 3 --psm 6 -l {language}'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            return text
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""
    
    def _clean_extracted_text_enhanced(self, text):
        """Enhanced text cleaning with academic focus and structure preservation."""
        if not text:
            return ""
        
        # Basic normalization
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts while preserving academic structure
        text = re.sub(r'\bpage\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bchapter\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsection\s+\d+(\.\d+)*\b', '', text, flags=re.IGNORECASE)
        
        # Clean up figure and table references
        text = re.sub(r'\bfigure\s+\d+(\.\d+)*\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\btable\s+\d+(\.\d+)*\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bfig\.\s*\d+\b', '', text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation while preserving academic formatting
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        text = re.sub(r'[=]{3,}', '===', text)
        text = re.sub(r'[_]{3,}', '___', text)
        
        # Remove standalone numbers but preserve numbered lists
        text = re.sub(r'(?<!\d)\b\d+\b(?!\d|\.|:)', '', text)
        
        # Enhanced OCR artifact removal
        text = re.sub(r'\b[a-zA-Z]\b(?=\s)', '', text)  # Single letters
        text = re.sub(r'[|]{2,}', '', text)  # Multiple pipes
        text = re.sub(r'[\\]{2,}', '', text)  # Multiple backslashes
        text = re.sub(r'[~]{2,}', '', text)  # Multiple tildes
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Normalize quotes and apostrophes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Preserve academic formatting (definitions, lists, etc.)
        # But remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def _is_academic_quality_text(self, text):
        """Check if extracted text meets academic quality standards."""
        if not text or len(text.strip()) < 50:
            return False
        
        text_lower = text.lower()
        
        # Must contain academic indicators
        academic_count = sum(1 for term in self.academic_indicators if term in text_lower)
        if academic_count == 0:
            return False
        
        # Check text structure and quality
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return False
        
        # Check character composition (must be mostly alphabetic)
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.6:
            return False
        
        # Check for reasonable word length distribution
        words = text.split()
        if len(words) < 20:
            return False
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 3 or avg_word_length > 15:
            return False
        
        return True
    
    def _calculate_academic_content_ratio(self, text):
        """Calculate the ratio of academic content in the text."""
        if not text:
            return 0
        
        text_lower = text.lower()
        academic_terms_found = sum(1 for term in self.academic_indicators if term in text_lower)
        
        # Calculate ratio based on text length and term frequency
        words = text_lower.split()
        if len(words) == 0:
            return 0
        
        # Academic ratio: (academic terms / total words) * 100, capped at 100
        ratio = min((academic_terms_found / len(words)) * 1000, 100)
        return ratio
    
    def _calculate_text_quality_score(self, text):
        """Calculate overall text quality score (0-10)."""
        if not text:
            return 0
        
        score = 0
        text_lower = text.lower()
        words = text.split()
        
        # Length scoring
        if 1000 <= len(text) <= 10000:
            score += 3
        elif 500 <= len(text) < 1000 or 10000 < len(text) <= 20000:
            score += 2
        elif 200 <= len(text) < 500:
            score += 1
        
        # Academic content scoring
        academic_terms = sum(1 for term in self.academic_indicators if term in text_lower)
        score += min(academic_terms / 5, 3)  # Up to 3 points for academic terms
        
        # Structure scoring
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) >= 5:
            score += 2
        elif len(sentences) >= 2:
            score += 1
        
        # Vocabulary diversity
        if words:
            unique_words = set(words)
            diversity = len(unique_words) / len(words)
            if diversity > 0.6:
                score += 2
            elif diversity > 0.4:
                score += 1
        
        return min(score, 10)
    
    def get_pdf_info(self, pdf_filename):
        """Get comprehensive information about a PDF in the knowledge base."""
        if pdf_filename not in self.metadata:
            return None
        
        info = self.metadata[pdf_filename].copy()
        
        # Add current file status
        pdf_path = os.path.join(self.storage_dir, pdf_filename)
        info['current_file_exists'] = os.path.exists(pdf_path)
        
        if info['current_file_exists']:
            info['current_file_size'] = os.path.getsize(pdf_path)
            # Verify hash integrity
            current_hash = self._calculate_file_hash(pdf_path)
            info['hash_verified'] = (current_hash == info.get('hash'))
        
        return info
    
    def list_pdfs(self):
        """List all PDFs in the knowledge base with comprehensive metadata display."""
        if not self.metadata:
            print("📚 Knowledge base is empty")
            return []
        
        print(f"📚 Knowledge Base Contents ({len(self.metadata)} files):")
        print("=" * 80)
        
        # Sort by academic score and added date
        sorted_pdfs = sorted(
            self.metadata.items(),
            key=lambda x: (x[1].get('academic_score', 0), x[1].get('added_date', '')),
            reverse=True
        )
        
        for filename, meta in sorted_pdfs:
            print(f"📄 {filename}")
            print(f"   📅 Added: {meta.get('added_date', 'Unknown')}")
            print(f"   📏 Size: {meta.get('file_size', 0) / (1024*1024):.1f}MB")
            print(f"   📑 Pages: {meta.get('page_count', 'Unknown')}")
            print(f"   📝 Text: {'Yes' if meta.get('has_text') else 'No'}")
            print(f"   🎓 Academic: {'Yes' if meta.get('academic_content') else 'No'} (Score: {meta.get('academic_score', 0)})")
            print(f"   🌐 Language: {meta.get('language', 'Unknown')}")
            print(f"   🔍 Extraction: {meta.get('extraction_method', 'Not extracted')}")
            
            if 'extraction_stats' in meta:
                stats = meta['extraction_stats']
                quality_score = meta.get('text_quality_score', 0)
                print(f"   📊 Quality: {quality_score}/10 ({stats.get('total_characters', 0):,} chars)")
            
            print()
        
        return list(self.metadata.keys())
    
    def remove_pdf(self, pdf_filename):
        """Remove a PDF from the knowledge base with comprehensive cleanup."""
        with self._lock:
            if pdf_filename not in self.metadata:
                print(f"❌ PDF not found in knowledge base: {pdf_filename}")
                return False
            
            pdf_path = os.path.join(self.storage_dir, pdf_filename)
            
            try:
                # Remove physical file if it exists
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    print(f"🗑️ Removed file: {pdf_filename}")
                
                # Remove from metadata
                del self.metadata[pdf_filename]
                self._save_metadata()
                
                print(f"✅ Successfully removed PDF from knowledge base: {pdf_filename}")
                return True
                
            except Exception as e:
                error_msg = f"Failed to remove PDF: {e}"
                print(f"❌ {error_msg}")
                return False
    
    def get_storage_stats(self):
        """Get comprehensive storage statistics for the knowledge base."""
        total_files = len(self.metadata)
        total_size = sum(meta.get('file_size', 0) for meta in self.metadata.values())
        total_pages = sum(meta.get('page_count', 0) for meta in self.metadata.values())
        
        files_with_text = sum(1 for meta in self.metadata.values() if meta.get('has_text'))
        files_with_images = sum(1 for meta in self.metadata.values() if meta.get('has_images'))
        academic_files = sum(1 for meta in self.metadata.values() if meta.get('academic_content'))
        
        extraction_methods = {}
        text_qualities = {}
        languages = {}
        academic_scores = []
        
        for meta in self.metadata.values():
            # Extraction methods
            method = meta.get('extraction_method', 'not_extracted')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
            
            # Text qualities
            quality = meta.get('text_quality', 'unknown')
            text_qualities[quality] = text_qualities.get(quality, 0) + 1
            
            # Languages
            language = meta.get('language', 'unknown')
            languages[language] = languages.get(language, 0) + 1
            
            # Academic scores
            score = meta.get('academic_score', 0)
            if score > 0:
                academic_scores.append(score)
        
        avg_academic_score = sum(academic_scores) / len(academic_scores) if academic_scores else 0
        
        return {
            'total_files': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'total_pages': total_pages,
            'files_with_text': files_with_text,
            'files_with_images': files_with_images,
            'academic_files': academic_files,
            'avg_academic_score': avg_academic_score,
            'extraction_methods': extraction_methods,
            'text_qualities': text_qualities,
            'languages': languages,
            'ocr_available': self.ocr_available
        }
    
    def print_storage_stats(self):
        """Print comprehensive storage statistics."""
        stats = self.get_storage_stats()
        
        print("📊 Knowledge Base Statistics:")
        print("=" * 40)
        print(f"📁 Total files: {stats['total_files']}")
        print(f"💾 Total size: {stats['total_size_mb']:.1f} MB")
        print(f"📄 Total pages: {stats['total_pages']}")
        print(f"📝 Files with text: {stats['files_with_text']}")
        print(f"🖼️ Files with images: {stats['files_with_images']}")
        print(f"🎓 Academic files: {stats['academic_files']}")
        print(f"📊 Avg academic score: {stats['avg_academic_score']:.1f}/10")
        print(f"🔍 OCR available: {'Yes' if stats['ocr_available'] else 'No'}")
        
        if stats['extraction_methods']:
            print("\n📤 Extraction methods:")
            for method, count in stats['extraction_methods'].items():
                print(f"   {method}: {count} files")
        
        if stats['text_qualities']:
            print("\n📈 Text qualities:")
            for quality, count in stats['text_qualities'].items():
                print(f"   {quality}: {count} files")
        
        if stats['languages']:
            print("\n🌐 Languages:")
            for language, count in stats['languages'].items():
                print(f"   {language}: {count} files")

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Knowledge Base Manager Test")
    print("=" * 50)
    
    # Initialize knowledge base
    kb = KnowledgeBaseManager()
    
    # Print current stats
    kb.print_storage_stats()
    
    # List existing PDFs
    kb.list_pdfs()
    
    print("\n✅ Knowledge Base Manager ready for use!")
    print("Features:")
    print("  • Comprehensive PDF validation")
    print("  • Academic content detection")
    print("  • OCR support for scanned documents")
    print("  • Duplicate detection")
    print("  • Text quality assessment")
    print("  • Thread-safe operations")
