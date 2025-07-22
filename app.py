import argparse
import sys
import os
from datetime import datetime
from knowledgebase import KnowledgeBaseManager
from retrieval import RetrievalAugmentor
import quillai_llm

print(f"✅ Loaded Enhanced QuillAILLM from: {quillai_llm.__file__}")
QuillAILLM = quillai_llm.QuillAILLM

def initialize_components():
    """Initialize LLM and retrieval components for CLI use."""
    try:
        # Initialize LLM
        llm = QuillAILLM(
            model_name="microsoft/DialoGPT-medium",
            force_model_check=True,
            debug_mode=False
        )
        
        # Initialize retrieval system
        retrieval_system = RetrievalAugmentor(
            chunk_size=400,
            chunk_overlap=50
        )
        
        # Connect LLM with retrieval system
        llm.set_retrieval_system(retrieval_system)
        
        print("✅ CLI components initialized successfully")
        return llm, retrieval_system
        
    except Exception as e:
        print(f"❌ Failed to initialize CLI components: {e}")
        raise

def process_query(args):
    """Process a single query using CLI."""
    print(f"Processing query: {args.query}")
    
    try:
        llm, retrieval_system = initialize_components()
        
        # Detect query intent and domain
        intent, intent_confidence, all_intents = llm.detect_query_intent(args.query)
        domain, topics, domain_confidence = llm.detect_domain_and_topic(args.query)
        
        print(f"🧠 Query Analysis:")
        print(f"   Intent: {intent.upper()}")
        print(f"   Domain: {domain.upper()}")
        if topics:
            print(f"   Topics: {', '.join(topics)}")
        
        # Retrieve context if available
        context_chunks = []
        if not args.no_context:
            try:
                context_chunks = retrieval_system.retrieve_context(args.query, top_k=3)
                if context_chunks:
                    print(f"✅ Retrieved {len(context_chunks)} relevant context chunks")
                else:
                    print("⚠️ No relevant context found")
            except Exception as e:
                print(f"⚠️ Warning: Context retrieval failed: {e}")
        
        # Generate dual response
        dual_response = llm.generate_dual_response(
            query=args.query,
            mode=args.mode or "learning",
            marks=args.marks,
            context_chunks=context_chunks,
            temperature=args.temperature
        )
        
        # Display results
        print("\n" + "="*80)
        print("📝 DIALOGPT OUTPUT")
        print("="*80)
        print(dual_response['llm_output'])
        print("="*80)
        
        print("\n" + "="*80)
        print("📝 CUSTOM LLM OUTPUT")
        print("="*80)
        print(dual_response['custom_output'])
        print("="*80)
        
        # Statistics
        llm_word_count = dual_response['word_counts']['llm']
        custom_word_count = dual_response['word_counts']['custom']
        total_time = dual_response['generation_times']['total']
        
        print(f"📊 Generation Statistics:")
        print(f"   ⏱️ Total Time: {total_time:.2f} seconds")
        print(f"   📝 DialoGPT Words: {llm_word_count}")
        print(f"   📝 Custom LLM Words: {custom_word_count}")
        print(f"   🎯 Mode: {args.mode or 'learning'}")
        
        if args.marks:
            target_words = {2: 100, 5: 250, 10: 500}[args.marks]
            custom_accuracy = abs(custom_word_count - target_words) / target_words * 100
            print(f"   🎯 Target for Custom: {target_words} words")
            print(f"   📝 Custom Accuracy: {100-custom_accuracy:.1f}% (±{abs(custom_word_count-target_words)} words)")
        
        if context_chunks:
            print(f"   📚 Context: {len(context_chunks)} chunks used")
        
        print(f"   🌡️ Temperature: {args.temperature}")
        print(f"   🤖 Model: {args.model_name}")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

def handle_pdf_addition(args):
    """Handle PDF addition with comprehensive feedback."""
    print(f"📄 Processing PDF: {args.pdf}")
    
    if not os.path.exists(args.pdf):
        print(f"❌ Error: PDF file not found: {args.pdf}")
        return
    
    try:
        # Initialize knowledge base
        print(f"📄 Initializing knowledge base...")
        kb_manager = KnowledgeBaseManager(storage_dir="textbooks")
        
        # Add PDF
        print(f"📄 Adding PDF to knowledge base...")
        kb_manager.add_pdf(
            args.pdf,
            force_ocr=args.use_ocr,
            language=args.pdf_language
        )
        
        print(f"✅ PDF processed successfully")
        
        # Build search index
        print(f"📄 Building search index...")
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
            retrieval_system.print_index_stats()
            
        except Exception as index_error:
            print(f"⚠️ Warning: Could not build search index")
            print(f"   Error: {str(index_error)}")
            print(f"   PDF added but search functionality limited")
        
        print(f"\n🎉 PDF ready for queries!")
        print(f"📚 Suggested next steps:")
        print(f'   python app.py --mode learning --query "Summarize key concepts"')
        print(f'   python app.py --mode question --marks 5 --query "Main topics"')
        
    except Exception as e:
        print(f"❌ Error adding PDF: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

def handle_kb_management(args):
    """Handle knowledge base management operations."""
    try:
        kb_manager = KnowledgeBaseManager(storage_dir="textbooks")
        
        if args.list_pdfs:
            kb_manager.list_pdfs()
        
        if args.kb_stats:
            kb_manager.print_storage_stats()
            
    except Exception as e:
        print(f"❌ Error accessing knowledge base: {e}")

def handle_index_management(args):
    """Handle retrieval index management operations."""
    try:
        retrieval_system = RetrievalAugmentor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        retrieval_system.print_index_stats()
    except Exception as e:
        print(f"❌ Error accessing retrieval index: {e}")

def handle_source_removal(args):
    """Handle source removal from knowledge base and index."""
    try:
        print(f"🗑️ Removing source: {args.remove_source}")
        
        # Remove from retrieval index
        retrieval_system = RetrievalAugmentor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        if retrieval_system.remove_source(args.remove_source):
            print("✅ Source removed from retrieval index")
        
        # Remove from knowledge base
        kb_manager = KnowledgeBaseManager(storage_dir="textbooks")
        if kb_manager.remove_pdf(args.remove_source):
            print("✅ Source removed from knowledge base")
            
    except Exception as e:
        print(f"❌ Error removing source: {e}")

def handle_chunk_search(args):
    """Handle chunk search in the retrieval index."""
    try:
        retrieval_system = RetrievalAugmentor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        print(f"🔍 Searching chunks for: {args.search_chunks}")
        results = retrieval_system.search_chunks(args.search_chunks, max_results=10)
        
        if results:
            print(f"\n📊 Found {len(results)} relevant chunks:")
            print("-" * 80)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Source: {result['source']} (Chunk {result['chunk_id']})")
                print(f"   Similarity: {result['similarity']:.3f}")
                print(f"   Quality: {result['quality_score']}/5")
                print(f"   Words: {result['word_count']}")
                print(f"   Preview: {result['preview']}")
        else:
            print("❌ No relevant chunks found")
            
    except Exception as e:
        print(f"❌ Error searching chunks: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="QuillAI CLI - Academic Assistant",
        epilog="Example: python app.py --mode learning --query 'What is machine learning?' --verbose"
    )
    
    # Core arguments
    parser.add_argument("--mode", type=str, choices=["learning", "question"],
                        help="Select either 'learning' (detailed explanations) or 'question' (concise answers)")
    parser.add_argument("--query", type=str,
                        help="User question or request.")
    
    # Question mode specific
    parser.add_argument("--marks", type=int, choices=[2, 5, 10], required=False,
                        help="Specify marks if mode=question. Options: 2 (100 words), 5 (250 words), 10 (500 words).")
    
    # PDF processing
    parser.add_argument("--pdf", type=str,
                        help="Path to a PDF to add to the knowledge base.")
    parser.add_argument("--use_ocr", action="store_true",
                        help="Force OCR for PDF text extraction.")
    parser.add_argument("--pdf_language", type=str, default="eng",
                        help="Language for OCR processing (default: eng).")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Force rebuild of PDF index even if exists.")
    
    # Retrieval configuration
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of retrieved chunks (default: 3).")
    parser.add_argument("--chunk_size", type=int, default=400,
                        help="Target chunk size in words (default: 400).")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Overlap between chunks in words (default: 50).")
    parser.add_argument("--min_similarity", type=float, default=0.3,
                        help="Minimum similarity threshold for retrieval (default: 0.3).")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium",
                        help="Model name (default: microsoft/DialoGPT-medium)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8). Range: 0.1-1.5")
    
    # Feature toggles
    parser.add_argument("--no_context", action="store_true",
                        help="Disable context retrieval entirely.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging and detailed output.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with extensive logging.")
    
    # Knowledge base management
    parser.add_argument("--list_pdfs", action="store_true",
                        help="List all PDFs in knowledge base.")
    parser.add_argument("--kb_stats", action="store_true",
                        help="Show knowledge base statistics.")
    parser.add_argument("--index_stats", action="store_true",
                        help="Show retrieval index statistics.")
    parser.add_argument("--remove_source", type=str,
                        help="Remove a source from the knowledge base.")
    parser.add_argument("--search_chunks", type=str,
                        help="Search chunks in the index.")
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("🤖 QUILLAI - INTELLIGENT ACADEMIC ASSISTANT")
    print("=" * 80)
    print(f"📅 Session: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if hasattr(args, 'mode') and args.mode:
        print(f"🎯 Mode: {args.mode}")
    if hasattr(args, 'marks') and args.marks:
        print(f"📝 Target: {args.marks} marks")
    if hasattr(args, 'query') and args.query:
        print(f"❓ Query: {args.query[:60]}{'...' if len(args.query) > 60 else ''}")
    print("=" * 80)
    
    # Route to appropriate handler
    if args.query:
        process_query(args)
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
        print("❌ No query or operation specified.")
        print("\n💡 Usage Examples:")
        print("   python app.py --mode learning --query 'What is machine learning?'")
        print("   python app.py --mode question --marks 5 --query 'Define AI'")
        print("   python app.py --pdf textbook.pdf")
        print("   python app.py --list_pdfs --kb_stats")

if __name__ == "__main__":
    main()
