import argparse
import sys
import os
from datetime import datetime
from knowledgebase import KnowledgeBaseManager
from retrieval import RetrievalAugmentor
import quillai_llm

# Environment variables used:
# - QUILLAI_MODEL: Model name override (default: microsoft/DialoGPT-medium)
# - QUILLAI_STORAGE_DIR: Knowledge base storage directory (default: textbooks)
# - QUILLAI_CHUNK_SIZE: Default chunk size (default: 400)
# - QUILLAI_CHUNK_OVERLAP: Default chunk overlap (default: 50)

print(f"✅ Loaded Enhanced QuillAI LLM from: {quillai_llm.__file__}")
QuillAILLM = quillai_llm.QuillAILLM

def initialize_components(chunk_size=400, chunk_overlap=50, storage_dir="textbooks"):
    """Initialize LLM and retrieval components for CLI use with enhanced error handling."""
    try:
        print("🔧 Initializing QuillAI CLI components...")
        
        # Initialize LLM with strict model checking
        model_name = os.getenv("QUILLAI_MODEL", "microsoft/DialoGPT-medium")
        llm = QuillAILLM(
            model_name=model_name,
            force_model_check=True,
            debug_mode=False
        )
        print(f"✅ LLM initialized: {model_name}")
        
        # Initialize retrieval system
        retrieval_system = RetrievalAugmentor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"✅ Retrieval system initialized (chunks: {chunk_size}, overlap: {chunk_overlap})")
        
        # Connect LLM with retrieval system
        llm.set_retrieval_system(retrieval_system)
        print("✅ LLM connected to retrieval system")
        
        print("✅ CLI components initialized successfully")
        return llm, retrieval_system
        
    except Exception as e:
        print(f"❌ Failed to initialize CLI components: {e}")
        raise

def process_dual_query(args):
    """Process a single query using CLI with dual output display."""
    print(f"🔍 Processing query: {args.query}")
    
    try:
        llm, retrieval_system = initialize_components(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            storage_dir=args.storage_dir
        )
        
        print(f"\n🧠 Query Analysis:")
        print(f"   📝 Query: {args.query}")
        print(f"   🎯 Mode: {args.mode or 'learning'}")
        if args.marks:
            print(f"   📊 Marks: {args.marks}")
        print(f"   🌡️ Temperature: {args.temperature}")
        
        # Retrieve context if available
        context_chunks = []
        if not args.no_context:
            try:
                print(f"\n🔍 Retrieving context...")
                context_chunks = retrieval_system.retrieve_context(
                    args.query, 
                    top_k=args.top_k,
                    min_score_threshold=args.min_similarity
                )
                if context_chunks:
                    print(f"✅ Retrieved {len(context_chunks)} relevant context chunks")
                    if args.verbose:
                        for i, chunk in enumerate(context_chunks, 1):
                            preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                            print(f"   [{i}] {preview}")
                else:
                    print("⚠️ No relevant context found")
            except Exception as e:
                print(f"⚠️ Warning: Context retrieval failed: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        else:
            print("⚠️ Context retrieval disabled by --no_context flag")
        
        # Generate dual response
        print(f"\n🚀 Generating dual responses...")
        start_time = datetime.utcnow()
        
        dual_response = llm.generate_dual_response(
            query=args.query,
            mode=args.mode or "learning",
            marks=args.marks,
            context_chunks=context_chunks
        )
        
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Display results with clear separation
        print("\n" + "="*100)
        print("📝 DIALOGPT OUTPUT (microsoft/DialoGPT-medium)")
        print("="*100)
        print(dual_response['llm_output'])
        print("="*100)
        
        print("\n" + "="*100)
        print("📝 CUSTOM EXTRACTIVE OUTPUT (Academic Analysis)")
        print("="*100)
        print(dual_response['custom_output'])
        print("="*100)
        
        # Comprehensive statistics
        llm_word_count = dual_response['word_counts']['llm']
        custom_word_count = dual_response['word_counts']['custom']
        llm_time = dual_response['generation_times']['llm']
        custom_time = dual_response['generation_times']['custom']
        total_time = dual_response['generation_times']['total']
        
        print(f"\n📊 Generation Statistics:")
        print("-" * 50)
        print(f"⏱️ Total Time: {total_time:.2f} seconds")
        print(f"📝 DialogGPT:")
        print(f"   Words: {llm_word_count}")
        print(f"   Time: {llm_time:.2f}s")
        print(f"📝 Custom Extractive:")
        print(f"   Words: {custom_word_count}")
        print(f"   Time: {custom_time:.2f}s")
        print(f"🎯 Mode: {args.mode or 'learning'}")
        
        # Target word count analysis for question mode
        if args.marks:
            target_words = {2: 100, 5: 250, 10: 500}.get(args.marks, 100)
            custom_accuracy = abs(custom_word_count - target_words) / target_words * 100
            dialogpt_target = min(target_words, 400)  # DialogGPT is capped
            dialogpt_accuracy = abs(llm_word_count - dialogpt_target) / dialogpt_target * 100
            
            print(f"🎯 Target Analysis:")
            print(f"   Target for Custom: {target_words} words")
            print(f"   Custom Accuracy: {100-custom_accuracy:.1f}% (±{abs(custom_word_count-target_words)} words)")
            print(f"   Target for DialogGPT: {dialogpt_target} words")
            print(f"   DialogGPT Accuracy: {100-dialogpt_accuracy:.1f}% (±{abs(llm_word_count-dialogpt_target)} words)")
        
        # Context analysis
        if context_chunks:
            print(f"📚 Context Analysis:")
            print(f"   Chunks used: {len(context_chunks)}")
            total_context_words = sum(len(chunk.split()) for chunk in context_chunks)
            print(f"   Total context words: {total_context_words}")
        else:
            print(f"📚 Context: No context used")
        
        print(f"🔧 Configuration:")
        print(f"   Model: {args.model_name}")
        print(f"   Temperature: {args.temperature}")
        print(f"   Top-k retrieval: {args.top_k}")
        print(f"   Min similarity: {args.min_similarity}")
        print("="*100)
        
        # Quality assessment
        print(f"\n✅ Query processed successfully!")
        if args.verbose:
            print(f"📈 Quality Indicators:")
            print(f"   Both outputs > 20 words: {'✅' if llm_word_count > 20 and custom_word_count > 20 else '❌'}")
            print(f"   Academic content detected: {'✅' if 'academic' in dual_response['custom_output'].lower() else '❓'}")
            print(f"   Context utilized: {'✅' if context_chunks else '❌'}")
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def handle_pdf_addition(args):
    """Handle PDF addition with comprehensive feedback and error handling."""
    print(f"📄 Processing PDF: {args.pdf}")
    
    if not os.path.exists(args.pdf):
        print(f"❌ Error: PDF file not found: {args.pdf}")
        sys.exit(1)
    
    try:
        # Initialize knowledge base
        print(f"📚 Initializing knowledge base...")
        storage_dir = args.storage_dir or os.getenv("QUILLAI_STORAGE_DIR", "textbooks")
        kb_manager = KnowledgeBaseManager(storage_dir=storage_dir)
        
        # Add PDF with comprehensive validation
        print(f"📄 Adding PDF to knowledge base...")
        kb_manager.add_pdf(
            args.pdf,
            force_ocr=args.use_ocr,
            language=args.pdf_language
        )
        
        print(f"✅ PDF processed and added to knowledge base")
        
        # Build search index
        print(f"🔍 Building search index...")
        try:
            retrieval_system = RetrievalAugmentor(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            retrieval_system.build_or_update_index_from_pdf(
                args.pdf,
                source_name=os.path.basename(args.pdf),
                force_rebuild=args.force_rebuild
            )
            
            print(f"✅ Search index built successfully")
            
            # Display index statistics
            if args.verbose:
                print(f"\n📊 Index Statistics:")
                retrieval_system.print_index_stats()
            
        except Exception as index_error:
            print(f"⚠️ Warning: Could not build search index")
            print(f"   Error: {str(index_error)}")
            print(f"   PDF added but search functionality will be limited")
            if args.debug:
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 PDF ready for queries!")
        print(f"📚 Suggested next steps:")
        print(f'   python app.py --mode learning --query "Summarize key concepts from {os.path.basename(args.pdf)}"')
        print(f'   python app.py --mode question --marks 5 --query "What are the main topics in this document?"')
        
    except Exception as e:
        print(f"❌ Error adding PDF: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def handle_kb_management(args):
    """Handle knowledge base management operations with enhanced display."""
    try:
        storage_dir = args.storage_dir or os.getenv("QUILLAI_STORAGE_DIR", "textbooks")
        kb_manager = KnowledgeBaseManager(storage_dir=storage_dir)
        
        if args.list_pdfs:
            print("📚 Listing PDFs in knowledge base...")
            pdfs = kb_manager.list_pdfs()
            if not pdfs:
                print("📭 Knowledge base is empty")
                print("💡 Add PDFs using: python app.py --pdf <filename>")
        
        if args.kb_stats:
            print("📊 Knowledge base statistics:")
            kb_manager.print_storage_stats()
            
    except Exception as e:
        print(f"❌ Error accessing knowledge base: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def handle_index_management(args):
    """Handle retrieval index management operations with enhanced display."""
    try:
        retrieval_system = RetrievalAugmentor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        print("📊 Retrieval index statistics:")
        retrieval_system.print_index_stats()
        
        if args.verbose:
            print(f"\n🔧 Configuration:")
            print(f"   Chunk size: {args.chunk_size}")
            print(f"   Chunk overlap: {args.chunk_overlap}")
            print(f"   Model: sentence-transformers/all-MiniLM-L6-v2")
            
    except Exception as e:
        print(f"❌ Error accessing retrieval index: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def handle_source_removal(args):
    """Handle source removal from knowledge base and index with confirmation."""
    try:
        print(f"🗑️ Removing source: {args.remove_source}")
        
        if not args.force:
            response = input(f"Are you sure you want to remove '{args.remove_source}'? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("❌ Operation cancelled")
                return
        
        # Remove from retrieval index
        retrieval_system = RetrievalAugmentor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        if retrieval_system.remove_source(args.remove_source):
            print("✅ Source removed from retrieval index")
        else:
            print("⚠️ Source not found in retrieval index")
        
        # Remove from knowledge base
        storage_dir = args.storage_dir or os.getenv("QUILLAI_STORAGE_DIR", "textbooks")
        kb_manager = KnowledgeBaseManager(storage_dir=storage_dir)
        if kb_manager.remove_pdf(args.remove_source):
            print("✅ Source removed from knowledge base")
        else:
            print("⚠️ Source not found in knowledge base")
            
    except Exception as e:
        print(f"❌ Error removing source: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def handle_chunk_search(args):
    """Handle chunk search in the retrieval index with enhanced display."""
    try:
        retrieval_system = RetrievalAugmentor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        print(f"🔍 Searching chunks for: '{args.search_chunks}'")
        results = retrieval_system.search_chunks(args.search_chunks, max_results=args.search_results)
        
        if results:
            print(f"\n📊 Found {len(results)} relevant chunks:")
            print("=" * 100)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Source: {result['source']} (Chunk {result['chunk_id']})")
                print(f"   📊 Similarity: {result['similarity']:.3f}")
                print(f"   🎓 Quality: {result['quality_score']}/10")
                print(f"   📝 Words: {result['word_count']}")
                print(f"   �� Preview: {result['preview']}")
                
                if args.verbose and result['similarity'] > 0.7:
                    print(f"   📋 Full text: {result['text'][:300]}...")
                
                print("-" * 50)
        else:
            print("❌ No relevant chunks found")
            print("💡 Try:")
            print("   • Different search terms")
            print("   • Adding more PDFs to the knowledge base")
            print("   • Lowering the similarity threshold")
            
    except Exception as e:
        print(f"❌ Error searching chunks: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="QuillAI CLI - Enhanced Academic Assistant with Dual Output",
        epilog="Example: python app.py --mode learning --query 'What is machine learning?' --verbose",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Core query arguments
    parser.add_argument("--mode", type=str, choices=["learning", "question"],
                        help="Response mode: 'learning' (detailed explanations) or 'question' (concise answers)")
    parser.add_argument("--query", type=str,
                        help="User question or request for dual response generation")
    
    # Question mode specific
    parser.add_argument("--marks", type=int, choices=[2, 5, 10],
                        help="Target marks for question mode: 2 (100 words), 5 (250 words), 10 (500 words)")
    
    # PDF processing
    parser.add_argument("--pdf", type=str,
                        help="Path to PDF file to add to the knowledge base")
    parser.add_argument("--use_ocr", action="store_true",
                        help="Force OCR for PDF text extraction (useful for scanned documents)")
    parser.add_argument("--pdf_language", type=str, default="eng",
                        choices=['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus'],
                        help="Language for OCR processing (default: eng)")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Force rebuild of PDF index even if it already exists")
    
    # Storage configuration
    parser.add_argument("--storage_dir", type=str,
                        help="Knowledge base storage directory (default: textbooks)")
    
    # Retrieval configuration
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of retrieved chunks for context (default: 3)")
    parser.add_argument("--chunk_size", type=int, default=400,
                        help="Target chunk size in words (default: 400)")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Overlap between chunks in words (default: 50)")
    parser.add_argument("--min_similarity", type=float, default=0.4,
                        help="Minimum similarity threshold for retrieval (default: 0.4)")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium",
                        help="Model name (must be microsoft/DialoGPT-medium)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for generation (default: 0.8, range: 0.1-1.5)")
    
    # Feature toggles
    parser.add_argument("--no_context", action="store_true",
                        help="Disable context retrieval entirely")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output with detailed information")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with stack traces")
    parser.add_argument("--force", action="store_true",
                        help="Force operations without confirmation prompts")
    
    # Knowledge base management
    parser.add_argument("--list_pdfs", action="store_true",
                        help="List all PDFs in the knowledge base")
    parser.add_argument("--kb_stats", action="store_true",
                        help="Show detailed knowledge base statistics")
    parser.add_argument("--index_stats", action="store_true",
                        help="Show retrieval index statistics")
    parser.add_argument("--remove_source", type=str,
                        help="Remove a source from both knowledge base and index")
    parser.add_argument("--search_chunks", type=str,
                        help="Search chunks in the retrieval index")
    parser.add_argument("--search_results", type=int, default=10,
                        help="Maximum number of search results (default: 10)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.temperature < 0.1 or args.temperature > 1.5:
        print("❌ Error: Temperature must be between 0.1 and 1.5")
        sys.exit(1)
    
    if args.model_name != "microsoft/DialoGPT-medium":
        print("❌ Error: Only microsoft/DialoGPT-medium is supported")
        sys.exit(1)
    
    if args.mode == "question" and args.marks is None:
        print("⚠️ Warning: Question mode recommended with --marks argument")
    
    # Print header
    print("=" * 100)
    print("🤖 QUILLAI - ENHANCED ACADEMIC ASSISTANT WITH DUAL OUTPUT")
    print("=" * 100)
    print(f"📅 Session: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"🏠 Working Directory: {os.getcwd()}")
    
    if args.query:
        print(f"🎯 Mode: {args.mode or 'learning'}")
        if args.marks:
            print(f"📝 Target: {args.marks} marks")
        print(f"❓ Query: {args.query[:80]}{'...' if len(args.query) > 80 else ''}")
    
    print("=" * 100)
    
    # Route to appropriate handler
    try:
        if args.query:
            process_dual_query(args)
        elif args.pdf:
            handle_pdf_addition(args)
        elif args.list_pdfs or args.kb_stats:
            handle_kb_management(args)
        elif args.index_stats:
            handle_index_management(args)
        elif args.remove_source:
            handle_source_removal(args)
        elif args.search_chunks:
            handle_chunk_search(args)
        else:
            print("❌ No operation specified.")
            print("\n💡 Usage Examples:")
            print("   # Dual response generation")
            print("   python app.py --mode learning --query 'What is machine learning?'")
            print("   python app.py --mode question --marks 5 --query 'Define artificial intelligence'")
            print("")
            print("   # PDF management")
            print("   python app.py --pdf textbook.pdf")
            print("   python app.py --pdf scanned_book.pdf --use_ocr")
            print("")
            print("   # Knowledge base management")
            print("   python app.py --list_pdfs --kb_stats")
            print("   python app.py --index_stats")
            print("   python app.py --search_chunks 'neural networks'")
            print("")
            print("   # Advanced options")
            print("   python app.py --query 'Explain deep learning' --verbose --top_k 5")
            print("   python app.py --remove_source 'textbook.pdf' --force")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
