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

class KnowledgeBaseManager:
    """
    Enhanced knowledge base manager with OCR support, validation, robust error handling,
    and thread-safe operations using file locks.
    """
    
    def __init__(self, storage_dir="textbooks"):
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self.supported_languages = ['eng', 'fra', 'deu', 'spa', 'ita']
        self._lock = threading.RLock()
        
        # Create storage directory
        try:
            os.makedirs(self.storage_dir, mode=0o755, exist_ok=True)
            print(f"✓ Created storage directory: {self.storage_dir}")
        except Exception as e:
            print(f"⚠ Warning: Could not create storage directory {self.storage_dir}: {e}")
        
        # Initialize metadata
        self.metadata = self._load_metadata()
        
        # Check OCR availability
        self.ocr_available = self._check_ocr_availability()
    
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
                    with open(file_path, mode) as f:
                        yield f
            except (OSError, PermissionError) as e:
                print(f"⚠ Warning: File locking failed for {file_path}: {e}. Proceeding without lock.")
                # Fallback without locking
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
    
    def _check_ocr_availability(self):
        """Check if OCR (Tesseract) is available."""
        try:
            pytesseract.get_tesseract_version()
            print("✓ OCR (Tesseract) available for scanned PDF processing")
            return True
        except Exception as e:
            print(f"⚠ OCR not available: {e}")
            return False
    
    def _load_metadata(self):
        """Load metadata from file with thread safety."""
        with self._lock:
            try:
                with self._file_lock(self.metadata_file, 'r') as f:
                    if f:
                        return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata from {self.metadata_file}: {e}")
            return {}
    
    def _save_metadata(self):
        """Save metadata to file with thread safety and atomic writes."""
        with self._lock:
            try:
                temp_file = self.metadata_file + '.tmp'
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                
                # Atomic move
                os.replace(temp_file, self.metadata_file)
                print(f"✓ Saved metadata to {self.metadata_file}")
            except Exception as e:
                print(f"Warning: Could not save metadata to {self.metadata_file}: {e}")
                # Clean up temp file if it exists
                try:
                    os.remove(temp_file)
                except FileNotFoundError:
                    pass
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of file for duplicate detection."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Could not calculate hash for {file_path}: {e}")
            return None
    
    def _validate_pdf(self, pdf_path):
        """Comprehensive PDF validation with detailed diagnostics."""
        validation_result = {
            'valid': False,
            'file_size': 0,
            'page_count': 0,
            'has_text': False,
            'has_images': False,
            'language': 'unknown',
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check file existence and size
            if not os.path.isfile(pdf_path):
                validation_result['errors'].append(f"File not found: {pdf_path}")
                return validation_result
            
            file_size = os.path.getsize(pdf_path)
            validation_result['file_size'] = file_size
            
            if file_size == 0:
                validation_result['errors'].append("File is empty")
                return validation_result
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                validation_result['warnings'].append(f"Large file ({file_size / (1024*1024):.1f}MB) may cause performance issues")
            
            # Open and validate PDF structure
            with fitz.open(pdf_path) as doc:
                validation_result['page_count'] = len(doc)
                
                if len(doc) == 0:
                    validation_result['errors'].append("PDF has no pages")
                    return validation_result
                
                # Check for text content and images
                text_pages = 0
                image_pages = 0
                total_text_length = 0
                
                # Sample first few pages for analysis
                sample_pages = min(5, len(doc))
                
                for page_num in range(sample_pages):
                    page = doc[page_num]
                    
                    # Check for text
                    text = page.get_text()
                    if text.strip():
                        text_pages += 1
                        total_text_length += len(text)
                    
                    # Check for images
                    image_list = page.get_images()
                    if image_list:
                        image_pages += 1
                
                validation_result['has_text'] = text_pages > 0
                validation_result['has_images'] = image_pages > 0
                
                # Determine if PDF is primarily scanned
                if image_pages > text_pages and total_text_length < 100:
                    validation_result['warnings'].append("PDF appears to be scanned - OCR may be needed")
                
                # Basic language detection
                if total_text_length > 100:
                    sample_text = ""
                    for page_num in range(min(3, len(doc))):
                        sample_text += doc[page_num].get_text()[:500]
                    
                    validation_result['language'] = self._detect_language(sample_text)
                
                validation_result['valid'] = True
                
        except Exception as e:
            validation_result['errors'].append(f"PDF validation failed: {str(e)}")
            print(f"PDF validation error for {pdf_path}: {e}")
        
        return validation_result
    
    def _detect_language(self, text):
        """Simple language detection based on character patterns."""
        text_lower = text.lower()
        
        # Simple heuristics for common languages
        if any(word in text_lower for word in ['the', 'and', 'or', 'of', 'in', 'to', 'a', 'is']):
            return 'eng'
        elif any(word in text_lower for word in ['le', 'la', 'et', 'de', 'un', 'une', 'est']):
            return 'fra'
        elif any(word in text_lower for word in ['der', 'die', 'das', 'und', 'oder', 'ist']):
            return 'deu'
        elif any(word in text_lower for word in ['el', 'la', 'y', 'o', 'de', 'un', 'una', 'es']):
            return 'spa'
        else:
            return 'eng'  # Default to English
    
    def is_duplicate(self, pdf_path):
        """Check if PDF is already processed and indexed."""
        file_hash = self._calculate_file_hash(pdf_path)
        if not file_hash:
            return False
        
        filename = os.path.basename(pdf_path)
        
        # Check if file with same name and hash exists
        if filename in self.metadata:
            existing_hash = self.metadata[filename].get('hash')
            if existing_hash == file_hash:
                # Check if file was successfully processed
                extraction_method = self.metadata[filename].get('extraction_method')
                if extraction_method and extraction_method != 'pending':
                    return True
        
        # Check if hash exists with different filename
        for meta in self.metadata.values():
            if meta.get('hash') == file_hash:
                extraction_method = meta.get('extraction_method')
                if extraction_method and extraction_method != 'pending':
                    return True
        
        return False
    
    def add_pdf(self, pdf_path, force_ocr=False, language='eng'):
        """Enhanced PDF addition with validation, OCR support, and comprehensive error handling."""
        with self._lock:
            print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")
            
            # Validate PDF
            validation = self._validate_pdf(pdf_path)
            
            if not validation['valid']:
                error_msg = f"PDF validation failed: {'; '.join(validation['errors'])}"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # Display validation results
            print(f"   File size: {validation['file_size'] / (1024*1024):.1f}MB")
            print(f"   Pages: {validation['page_count']}")
            print(f"   Has text: {'Yes' if validation['has_text'] else 'No'}")
            print(f"   Has images: {'Yes' if validation['has_images'] else 'No'}")
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    print(f"   ⚠ {warning}")
            
            # Check for duplicates
            if self.is_duplicate(pdf_path):
                print("⚠ Duplicate file detected - skipping")
                return
            
            # Copy PDF to storage (directly in textbooks directory)
            filename = os.path.basename(pdf_path)
            dest_path = os.path.join(self.storage_dir, filename)
            
            try:
                if pdf_path != dest_path:  # Only copy if not already in destination
                    shutil.copy(pdf_path, dest_path)
                    print(f"✓ PDF copied to knowledge base: {filename}")
                else:
                    print(f"✓ PDF already in knowledge base: {filename}")
            except Exception as e:
                error_msg = f"Failed to copy PDF to {dest_path}: {e}"
                print(f"❌ {error_msg}")
                raise
            
            # Store metadata
            file_hash = self._calculate_file_hash(dest_path)
            self.metadata[filename] = {
                'original_path': pdf_path,
                'added_date': datetime.utcnow().isoformat(),
                'file_size': validation['file_size'],
                'page_count': validation['page_count'],
                'has_text': validation['has_text'],
                'has_images': validation['has_images'],
                'language': validation['language'],
                'hash': file_hash,
                'extraction_method': 'pending'
            }
            
            self._save_metadata()
            print(f"✓ PDF successfully added to knowledge base")
    
    def extract_text(self, pdf_filename, use_ocr=False, language='eng', clean_text=True):
        """Enhanced text extraction with OCR fallback and comprehensive cleaning."""
        pdf_path = os.path.join(self.storage_dir, pdf_filename)
        
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found in knowledge base: {pdf_filename}")
        
        print(f"📖 Extracting text from: {pdf_filename}")
        
        if use_ocr and not self.ocr_available:
            print("⚠ OCR requested but not available - falling back to standard extraction")
            use_ocr = False
        
        text_blocks = []
        extraction_stats = {
            'pages_processed': 0,
            'pages_with_text': 0,
            'pages_with_ocr': 0,
            'total_characters': 0,
            'extraction_method': 'standard' if not use_ocr else 'ocr'
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
                        
                        # Fallback to OCR if no text found and OCR is available
                        elif self.ocr_available and not page_text.strip():
                            print(f"   Page {page_num + 1}: No text found, trying OCR...")
                            ocr_text = self._extract_text_with_ocr(page, language)
                            if ocr_text.strip():
                                page_text = ocr_text
                                extraction_stats['pages_with_ocr'] += 1
                                extraction_stats['extraction_method'] = 'hybrid'
                    
                    if page_text.strip():
                        if clean_text:
                            page_text = self._clean_extracted_text(page_text)
                        
                        text_blocks.append(page_text)
                        extraction_stats['total_characters'] += len(page_text)
                    
                    # Progress indicator for large documents
                    if extraction_stats['pages_processed'] % 10 == 0:
                        print(f"   Processed {extraction_stats['pages_processed']} pages...")
        
        except Exception as e:
            error_msg = f"Text extraction failed for {pdf_filename}: {e}"
            print(f"❌ {error_msg}")
            raise
        
        # Update metadata
        if pdf_filename in self.metadata:
            self.metadata[pdf_filename]['extraction_method'] = extraction_stats['extraction_method']
            self.metadata[pdf_filename]['extraction_stats'] = extraction_stats
            self.metadata[pdf_filename]['last_extracted'] = datetime.utcnow().isoformat()
            self._save_metadata()
        
        # Display extraction results
        print(f"✓ Text extraction completed:")
        print(f"   Pages processed: {extraction_stats['pages_processed']}")
        print(f"   Pages with text: {extraction_stats['pages_with_text']}")
        if extraction_stats['pages_with_ocr'] > 0:
            print(f"   Pages with OCR: {extraction_stats['pages_with_ocr']}")
        print(f"   Total characters: {extraction_stats['total_characters']:,}")
        print(f"   Extraction method: {extraction_stats['extraction_method']}")
        
        if not text_blocks:
            print("⚠ No text extracted from PDF")
            if not use_ocr and self.ocr_available:
                print("💡 Try using OCR: extract_text(filename, use_ocr=True)")
        
        combined_text = "\n".join(text_blocks)
        return combined_text
    
    def _extract_text_with_ocr(self, page, language='eng'):
        """Extract text from a PDF page using OCR."""
        try:
            # Convert page to image
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR accuracy
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            custom_config = f'--oem 3 --psm 6 -l {language}'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            return text
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""
    
    def _clean_extracted_text(self, text):
        """Comprehensive text cleaning for better retrieval quality."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\bpage\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bchapter\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsection\s+\d+(\.\d+)*\b', '', text, flags=re.IGNORECASE)
        
        # Remove repeated punctuation and symbols
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        text = re.sub(r'[=]{3,}', '===', text)
        text = re.sub(r'[_]{3,}', '___', text)
        
        # Remove standalone numbers (often page numbers or figure numbers)
        text = re.sub(r'\b\d+\b(?=\s|$)', '', text)
        
        # Clean up common OCR artifacts
        text = re.sub(r'\b[a-zA-Z]\b(?=\s)', '', text)  # Single letters
        text = re.sub(r'[|]{2,}', '', text)  # Multiple pipes
        text = re.sub(r'[\\]{2,}', '', text)  # Multiple backslashes
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*$$$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Normalize quotes and apostrophes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Remove excessive whitespace again
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
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
        
        return info
    
    def list_pdfs(self):
        """List all PDFs in the knowledge base with their metadata."""
        if not self.metadata:
            print("No PDFs in knowledge base")
            return []
        
        print(f"📚 Knowledge Base Contents ({len(self.metadata)} files):")
        print("-" * 80)
        
        for filename, meta in self.metadata.items():
            print(f"📄 {filename}")
            print(f"   Added: {meta.get('added_date', 'Unknown')}")
            print(f"   Size: {meta.get('file_size', 0) / (1024*1024):.1f}MB")
            print(f"   Pages: {meta.get('page_count', 'Unknown')}")
            print(f"   Has text: {'Yes' if meta.get('has_text') else 'No'}")
            print(f"   Language: {meta.get('language', 'Unknown')}")
            print(f"   Extraction: {meta.get('extraction_method', 'Not extracted')}")
            
            if 'extraction_stats' in meta:
                stats = meta['extraction_stats']
                print(f"   Characters extracted: {stats.get('total_characters', 0):,}")
            
            print()
        
        return list(self.metadata.keys())
    
    def remove_pdf(self, pdf_filename):
        """Remove a PDF from the knowledge base."""
        with self._lock:
            if pdf_filename not in self.metadata:
                print(f"PDF not found in knowledge base: {pdf_filename}")
                return False
            
            pdf_path = os.path.join(self.storage_dir, pdf_filename)
            
            try:
                # Remove file if it exists
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                
                # Remove from metadata
                del self.metadata[pdf_filename]
                self._save_metadata()
                
                print(f"✓ Removed PDF from knowledge base: {pdf_filename}")
                return True
                
            except Exception as e:
                error_msg = f"Failed to remove PDF: {e}"
                print(f"❌ {error_msg}")
                return False
    
    def get_storage_stats(self):
        """Get storage statistics for the knowledge base."""
        total_files = len(self.metadata)
        total_size = sum(meta.get('file_size', 0) for meta in self.metadata.values())
        total_pages = sum(meta.get('page_count', 0) for meta in self.metadata.values())
        
        files_with_text = sum(1 for meta in self.metadata.values() if meta.get('has_text'))
        files_with_images = sum(1 for meta in self.metadata.values() if meta.get('has_images'))
        
        extraction_methods = {}
        for meta in self.metadata.values():
            method = meta.get('extraction_method', 'not_extracted')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
        
        return {
            'total_files': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'total_pages': total_pages,
            'files_with_text': files_with_text,
            'files_with_images': files_with_images,
            'extraction_methods': extraction_methods,
            'ocr_available': self.ocr_available
        }
    
    def print_storage_stats(self):
        """Print comprehensive storage statistics."""
        stats = self.get_storage_stats()
        
        print("📊 Knowledge Base Statistics:")
        print("-" * 40)
        print(f"Total files: {stats['total_files']}")
        print(f"Total size: {stats['total_size_mb']:.1f} MB")
        print(f"Total pages: {stats['total_pages']}")
        print(f"Files with text: {stats['files_with_text']}")
        print(f"Files with images: {stats['files_with_images']}")
        print(f"OCR available: {'Yes' if stats['ocr_available'] else 'No'}")
        
        if stats['extraction_methods']:
            print("\nExtraction methods:")
            for method, count in stats['extraction_methods'].items():
                print(f"   {method}: {count} files")

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
    
    print("\nKnowledge Base Manager ready for use!")
