import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from collections import Counter
import concurrent.futures
import logging
import random
import time

# Environment variables used:
# - HF_HOME: Hugging Face cache directory
# - TRANSFORMERS_CACHE: Transformers cache directory

class TextbookLLM:
    """
    Retrieval-augmented LLM that uses actual textbook content for generation.
    True RAG implementation replacing template-based responses.
    """
    
    def __init__(self, retrieval_system=None, model_name="microsoft/DialoGPT-medium"):
        self.retrieval_system = retrieval_system
        self.model_name = model_name
        self.logger = logging.getLogger("textbook_llm")
        
        # Initialize the base LLM for text generation
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"✅ TextbookLLM initialized with {model_name}")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize TextbookLLM: {e}")
            raise

    def generate_answer(self, query, mode="learning", marks=None, max_tokens=500):
        """Generate answer using retrieval-augmented generation from textbook content."""
        
        # Check for nonsensical queries
        if self._is_nonsensical_query(query):
            return "Please rephrase your question for my better understanding."
        
        # Retrieve relevant textbook content
        context_chunks = []
        if self.retrieval_system:
            try:
                context_chunks = self.retrieval_system.retrieve_context(query, top_k=5)
            except Exception as e:
                self.logger.warning(f"Context retrieval failed: {e}")
        
        if not context_chunks:
            return "Please rephrase your question for my better understanding."
        
        # Determine target word count
        target_words = self._get_target_words(mode, marks)
        
        # Handle question generation specially
        if "generate" in query.lower() and "question" in query.lower():
            return self._generate_questions_from_content(query, context_chunks, marks)
        
        # Generate answer using textbook content
        answer = self._synthesize_answer_from_content(query, context_chunks, target_words, mode)
        
        return answer

    def _is_nonsensical_query(self, query):
        """Check if query is nonsensical or irrelevant."""
        query_lower = query.lower().strip()
        
        # Very short queries
        if len(query_lower) < 3:
            return True
        
        # Nonsensical patterns
        nonsensical_patterns = [
            r'^[^a-zA-Z]*$',  # Only numbers/symbols
            r'^(.)\1{4,}',    # Repeated characters
            r'^\w{1,2}$',     # Single/double letters
        ]
        
        for pattern in nonsensical_patterns:
            if re.match(pattern, query_lower):
                return True
        
        # Check for meaningful words
        words = query_lower.split()
        if len(words) < 2:
            return True
        
        # Must contain at least one meaningful word
        meaningful_words = [w for w in words if len(w) > 2 and w.isalpha()]
        if len(meaningful_words) == 0:
            return True
        
        return False

    def _get_target_words(self, mode, marks):
        """Get target word count based on mode and marks."""
        if mode == "question" and marks:
            return {2: 100, 5: 250, 10: 500}.get(marks, 100)
        elif mode == "learning":
            return 300
        else:
            return 200

    def _generate_questions_from_content(self, query, context_chunks, marks):
        """Generate questions based on textbook content."""
        # Extract number of questions requested
        numbers = re.findall(r'\d+', query)
        num_questions = int(numbers[0]) if numbers else 5
        num_questions = min(num_questions, 10)  # Cap at 10
        
        # Extract topic from context
        combined_content = " ".join(context_chunks[:3])  # Use top 3 chunks
        
        # Generate questions based on content
        questions = []
        question_starters = [
            "What is", "Define", "Explain", "How does", "Why is",
            "Compare", "Describe", "What are the advantages of",
            "What are the applications of", "How can"
        ]
        
        # Extract key concepts from content
        key_concepts = self._extract_key_concepts(combined_content)
        
        for i in range(num_questions):
            if i < len(key_concepts):
                concept = key_concepts[i]
                starter = question_starters[i % len(question_starters)]
                
                if starter in ["What is", "Define"]:
                    question = f"{starter} {concept}?"
                elif starter in ["Explain", "Describe"]:
                    question = f"{starter} {concept} in detail."
                elif starter == "How does":
                    question = f"How does {concept} work?"
                elif starter == "Compare":
                    if i + 1 < len(key_concepts):
                        question = f"Compare {concept} and {key_concepts[i+1]}."
                    else:
                        question = f"Compare different types of {concept}."
                else:
                    question = f"{starter} {concept}?"
                
                questions.append(question)
        
        # Format response
        marks_info = f" ({marks} marks each)" if marks else ""
        response = f"**Questions based on textbook content{marks_info}:**\n\n"
        
        for i, question in enumerate(questions, 1):
            response += f"**Q{i}.** {question}\n\n"
        
        response += f"*These questions are derived from the available textbook content.*"
        
        return response

    def _extract_key_concepts(self, content):
        """Extract key concepts from textbook content."""
        concepts = []
        
        # Look for definitions (X is a/an...)
        definition_pattern = r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+is\s+(?:a|an)\s+'
        definitions = re.findall(definition_pattern, content)
        concepts.extend(definitions[:3])
        
        # Look for capitalized terms (likely important concepts)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        # Filter out common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'With', 'By'}
        important_terms = [term for term in capitalized_terms if term not in common_words and len(term) > 3]
        concepts.extend(important_terms[:5])
        
        # Look for technical terms (words ending in -tion, -ing, -ment, etc.)
        technical_pattern = r'\b[a-z]+(?:tion|ing|ment|ness|ity|ism|ogy|ics)\b'
        technical_terms = re.findall(technical_pattern, content, re.IGNORECASE)
        concepts.extend(technical_terms[:3])
        
        # Remove duplicates and return
        unique_concepts = []
        seen = set()
        for concept in concepts:
            if concept.lower() not in seen and len(concept) > 2:
                unique_concepts.append(concept)
                seen.add(concept.lower())
        
        return unique_concepts[:10]

    def _synthesize_answer_from_content(self, query, context_chunks, target_words, mode):
        """Synthesize answer from textbook content using the LLM."""
        # Combine relevant chunks
        combined_content = " ".join(context_chunks[:3])  # Use top 3 chunks
        
        # Create a prompt that instructs the model to answer based on the content
        prompt = self._create_rag_prompt(query, combined_content, target_words, mode)
        
        # Generate using the LLM
        try:
            answer = self._generate_with_llm(prompt, target_words)
            
            # Post-process to ensure quality
            answer = self._post_process_answer(answer, query, target_words)
            
            return answer
        except Exception as e:
            self.logger.warning(f"LLM generation failed: {e}")
            # Fallback to extractive summarization
            return self._extractive_answer(query, combined_content, target_words)

    def _create_rag_prompt(self, query, content, target_words, mode):
        """Create a prompt for retrieval-augmented generation."""
        if mode == "learning":
            instruction = f"Based on the following textbook content, provide a comprehensive academic explanation for the question. Include definitions, key concepts, examples, and applications. Write approximately {target_words} words."
        else:
            instruction = f"Based on the following textbook content, provide a precise academic answer to the question. Write exactly {target_words} words."
        
        prompt = f"""Textbook Content:
{content[:1000]}...

{instruction}

Question: {query}

Academic Answer:"""
        
        return prompt

    def _generate_with_llm(self, prompt, target_words):
        """Generate text using the LLM with improved parameters."""
        # Tokenize prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        
        # Calculate max_new_tokens based on target
        max_new_tokens = min(target_words * 2, 500)  # Allow up to 500 tokens
        
        # Generate with improved parameters
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max(30, target_words // 2),
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode and extract answer
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        if "Academic Answer:" in full_text:
            answer = full_text.split("Academic Answer:")[-1].strip()
        else:
            # Fallback: take the part after the prompt
            answer = full_text[len(prompt):].strip()
        
        return answer

    def _post_process_answer(self, answer, query, target_words):
        """Post-process the generated answer for quality and length."""
        if not answer:
            return "Please rephrase your question for my better understanding."
        
        # Clean up common artifacts
        answer = re.sub(r'<\|.*?\|>', '', answer)  # Remove special tokens
        answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)  # Clean excessive newlines
        answer = answer.strip()
        
        # Ensure minimum quality
        if len(answer.split()) < 20:
            return "Please rephrase your question for my better understanding."
        
        # Adjust length to target
        current_words = len(answer.split())
        if target_words and abs(current_words - target_words) > target_words * 0.3:
            answer = self._adjust_length(answer, target_words)
        
        return answer

    def _adjust_length(self, answer, target_words):
        """Adjust answer length to meet target word count."""
        current_words = len(answer.split())
        
        if current_words > target_words * 1.2:
            # Truncate if too long
            words = answer.split()[:target_words]
            answer = " ".join(words)
            if not answer.endswith('.'):
                answer += '.'
        elif current_words < target_words * 0.8:
            # Expand if too short
            expansion = " This concept is fundamental in the field and has wide-ranging applications. Understanding these principles is essential for academic and professional development."
            words_needed = target_words - current_words
            expansion_words = expansion.split()[:words_needed]
            answer += " " + " ".join(expansion_words)
        
        return answer

    def _extractive_answer(self, query, content, target_words):
        """Fallback extractive answer when generation fails."""
        # Simple extractive approach
        sentences = content.split('.')
        relevant_sentences = []
        
        query_words = set(query.lower().split())
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum length
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words & sentence_words)
                if overlap > 0:
                    relevant_sentences.append((sentence, overlap))
        
        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        answer_parts = []
        word_count = 0
        
        for sentence, _ in relevant_sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= target_words:
                answer_parts.append(sentence)
                word_count += sentence_words
            else:
                break
        
        if answer_parts:
            return ". ".join(answer_parts) + "."
        else:
            return "Please rephrase your question for my better understanding."


class QuillAILLM:
    def __init__(self, model_name="microsoft/DialoGPT-medium", force_model_check=True, debug_mode=False):
        """Initialize QuillAI LLM with specified model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.debug_mode = debug_mode
        self.logger = logging.getLogger("quillai_llm")
        
        # Initialize semantic understanding for query analysis
        try:
            self.semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.logger.info("✓ Semantic understanding model loaded")
        except Exception as e:
            self.logger.warning(f"⚠ Warning: Could not load semantic model: {e}")
            self.semantic_model = None
        
        # Intent detection patterns
        self.intent_patterns = {
            'question_generation': [
                r'generate.*questions?', r'create.*questions?', r'make.*questions?',
                r'question.*paper', r'exam.*questions?', r'test.*questions?',
                r'quiz.*questions?', r'assessment.*questions?', r'\d+.*questions?'
            ],
            'rubric_creation': [
                r'rubric', r'marking.*scheme', r'grading.*criteria',
                r'evaluation.*criteria', r'assessment.*criteria'
            ],
            'list_generation': [
                r'list.*of', r'enumerate', r'give.*examples?', r'provide.*examples?',
                r'name.*\d+', r'mention.*\d+', r'state.*\d+'
            ],
            'summary_request': [
                r'summarize', r'summary', r'key.*points?', r'main.*points?',
                r'overview', r'brief.*explanation'
            ],
            'comparison': [
                r'compare', r'contrast', r'difference', r'versus', r'vs\.?',
                r'similarities?.*differences?', r'pros.*cons'
            ],
            'definition': [
                r'what.*is', r'define', r'definition', r'meaning.*of',
                r'explain.*term', r'concept.*of'
            ],
            'explanation': [
                r'explain', r'how.*does', r'how.*to', r'describe',
                r'process.*of', r'steps?.*to', 'procedure'
            ],
            'application': [
                r'examples?.*of', r'applications?.*of', r'uses?.*of',
                r'real.*world', r'practical.*use', 'implement'
            ]
        }
        
        # Domain knowledge for multiple academic fields
        self.domain_knowledge = {
            'computer_science': {
                'algorithms': ['sorting', 'searching', 'graph', 'dynamic programming', 'greedy'],
                'machine_learning': ['supervised', 'unsupervised', 'neural networks', 'deep learning'],
                'data_structures': ['array', 'tree', 'graph', 'hash table', 'stack', 'queue'],
                'programming': ['object oriented', 'functional', 'procedural', 'languages']
            },
            'mathematics': {
                'calculus': ['derivative', 'integral', 'limit', 'continuity'],
                'linear_algebra': ['matrix', 'vector', 'eigenvalue', 'determinant'],
                'statistics': ['probability', 'distribution', 'hypothesis testing', 'regression'],
                'discrete_math': ['graph theory', 'combinatorics', 'logic', 'set theory']
            },
            'physics': {
                'mechanics': ['force', 'momentum', 'energy', 'motion'],
                'thermodynamics': ['entropy', 'enthalpy', 'heat', 'temperature'],
                'electromagnetism': ['electric field', 'magnetic field', 'current', 'voltage'],
                'quantum': ['superposition', 'entanglement', 'wave function', 'uncertainty']
            },
            'chemistry': {
                'organic': ['hydrocarbons', 'functional groups', 'reactions', 'synthesis'],
                'inorganic': ['periodic table', 'bonding', 'coordination', 'crystals'],
                'physical': ['thermodynamics', 'kinetics', 'equilibrium', 'spectroscopy'],
                'analytical': ['chromatography', 'spectroscopy', 'titration', 'mass spec']
            },
            'biology': {
                'molecular': ['dna', 'rna', 'proteins', 'enzymes', 'metabolism'],
                'cellular': ['cell structure', 'organelles', 'membrane', 'division'],
                'genetics': ['inheritance', 'mutations', 'gene expression', 'evolution'],
                'ecology': ['ecosystems', 'biodiversity', 'conservation', 'populations']
            },
            'economics': {
                'microeconomics': ['supply', 'demand', 'elasticity', 'market structures'],
                'macroeconomics': ['gdp', 'inflation', 'unemployment', 'monetary policy'],
                'finance': ['investment', 'risk', 'portfolio', 'derivatives'],
                'behavioral': ['decision making', 'biases', 'game theory', 'psychology']
            }
        }
        
        # Force check to ensure we're not loading problematic models
        if force_model_check and "gpt2-medium" in model_name.lower() and "microsoft" not in model_name.lower():
            raise ValueError(f"ERROR: Attempting to load base GPT-2 model '{model_name}'. Use 'microsoft/DialoGPT-medium' instead!")
        
        self.logger.info(f"Loading model: {model_name} on {self.device} ...")
        self.logger.info(f"Expected model: {model_name}")
        self.logger.info(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                device_map=None  # We'll move to device manually
            )
            
            # Validate we loaded the correct model
            actual_model_name = getattr(self.model.config, '_name_or_path', model_name)
            self.logger.info(f"Actual loaded model: {actual_model_name}")
            
            self.logger.info("Model loaded successfully.")
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            if "gated repo" in str(e) or "access" in str(e).lower() or "401" in str(e):
                self.logger.info(f"Error: Model {model_name} requires authentication or access.")
                self.logger.info("Falling back to microsoft/DialoGPT-medium...")
                # Fallback to DialoGPT
                model_name = "microsoft/DialoGPT-medium"
                self.model_name = model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                    device_map=None
                )
                self.logger.info(f"Successfully loaded fallback model: {model_name}")
                self.model.to(self.device)
                self.model.eval()
            else:
                raise e
        
        # Initialize the TextbookLLM for custom responses
        self.textbook_llm = None  # Will be set when retrieval system is available

    def set_retrieval_system(self, retrieval_system):
        """Set the retrieval system for the TextbookLLM."""
        try:
            self.textbook_llm = TextbookLLM(retrieval_system, self.model_name)
            self.logger.info("✅ TextbookLLM initialized with retrieval system")
        except Exception as e:
            self.logger.warning(f"⚠ Warning: Could not initialize TextbookLLM: {e}")
            self.textbook_llm = None

    def detect_query_intent(self, query):
        """Detect the intent/type of the query using pattern matching and semantic analysis."""
        query_lower = query.lower()
        detected_intents = {}
        
        # Pattern-based detection
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intents[intent] = detected_intents.get(intent, 0) + 1
        
        # Semantic similarity detection (if available)
        if self.semantic_model:
            semantic_intents = self._semantic_intent_detection(query)
            for intent, score in semantic_intents.items():
                detected_intents[intent] = detected_intents.get(intent, 0) + score
        
        # Determine primary intent
        if detected_intents:
            primary_intent = max(detected_intents, key=detected_intents.get)
            confidence = detected_intents[primary_intent] / sum(detected_intents.values())
        else:
            primary_intent = 'definition'  # Default fallback
            confidence = 0.5
        
        return primary_intent, confidence, detected_intents

    def _semantic_intent_detection(self, query):
        """Use semantic similarity to detect query intent."""
        intent_examples = {
            'question_generation': "Generate 10 questions about machine learning",
            'rubric_creation': "Create a marking rubric for this assignment",
            'list_generation': "List 5 examples of sorting algorithms",
            'summary_request': "Summarize the key points of this topic",
            'comparison': "Compare supervised and unsupervised learning",
            'definition': "What is artificial intelligence?",
            'explanation': "Explain how neural networks work",
            'application': "Give examples of AI applications"
        }
        
        try:
            query_embedding = self.semantic_model.encode([query])
            intent_scores = {}
            
            for intent, example in intent_examples.items():
                example_embedding = self.semantic_model.encode([example])
                similarity = np.dot(query_embedding[0], example_embedding[0]) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(example_embedding[0])
                )
                if similarity > 0.3:  # Threshold for relevance
                    intent_scores[intent] = similarity * 2  # Weight semantic scores
            
            return intent_scores
        except Exception as e:
            if self.debug_mode:
                self.logger.debug(f"Semantic intent detection failed: {e}")
            return {}

    def detect_domain_and_topic(self, query):
        """Detect the academic domain and specific topics from the query."""
        query_lower = query.lower()
        domain_scores = {}
        detected_topics = []
        
        for domain, categories in self.domain_knowledge.items():
            domain_score = 0
            for category, topics in categories.items():
                for topic in topics:
                    if topic in query_lower:
                        domain_score += 1
                        detected_topics.append(topic)
            if domain_score > 0:
                domain_scores[domain] = domain_score
        
        # Determine primary domain
        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
            confidence = domain_scores[primary_domain] / sum(domain_scores.values())
        else:
            primary_domain = 'general'
            confidence = 0.3
        
        return primary_domain, detected_topics, confidence

    def generate_answer(self, query, mode="learning", marks=None, context_chunks=None,
                       rerank_context=True, return_citations=True, feedback_callback=None,
                       temperature=1.0, max_new_tokens=500):
        """Enhanced generate_answer with intent detection and improved processing."""
        
        start_time = datetime.utcnow()
        
        # Detect query intent and domain
        intent, intent_confidence, all_intents = self.detect_query_intent(query)
        domain, topics, domain_confidence = self.detect_domain_and_topic(query)
        
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Detected intent: {intent} (confidence: {intent_confidence:.2f})")
        self.logger.info(f"Detected domain: {domain} (confidence: {domain_confidence:.2f})")
        self.logger.info(f"Topics: {topics}")
        
        # Route to specialized handlers based on intent
        if intent in ['question_generation', 'rubric_creation', 'list_generation', 'summary_request']:
            return self._handle_special_intent(intent, query, mode, marks, context_chunks)
        
        # Standard processing for other intents
        target_words = None
        if mode.lower() == "question" and marks is not None:
            target_words = {2: 100, 5: 250, 10: 500}.get(marks, 100)
        
        # Enhanced context handling
        selected_context = []
        if context_chunks and len(context_chunks) > 0:
            if rerank_context:
                context_chunks = self._enhanced_rerank_context(query, context_chunks, domain, topics)
            
            relevant_chunks = []
            for chunk in context_chunks[:2]:
                if self._is_context_relevant(chunk, query, domain, topics):
                    if len(chunk) > 300:
                        chunk = chunk[:300] + "..."
                    relevant_chunks.append(chunk)
            
            selected_context = relevant_chunks[:1]  # Use only the top 1 relevant chunk
        
        # Create enhanced prompt
        prompt = self._create_enhanced_prompt(query, mode, marks, target_words, selected_context)
        
        # Generate with LLM
        answer = self._generate_with_llm(prompt, max_new_tokens, temperature, query, mode, target_words)
        
        # Apply deduplication
        answer = self.deduplicate_response(answer)
        
        # Enhance and expand response
        answer = self._enhance_and_expand_response(answer, query, mode, target_words)
        
        # Add citations if requested
        if return_citations and selected_context:
            citations = []
            for i, chunk in enumerate(selected_context):
                clean_chunk = re.sub(r'[\s\n\r]+', ' ', chunk[:80])
                citations.append(f"[{i+1}] {clean_chunk}")
            
            if citations:
                answer += "\n\nReferences:\n" + "\n".join(citations)
        
        # Final quality check
        answer = self._final_quality_check(answer, query, mode, target_words)
        
        generation_time = (datetime.utcnow() - start_time).total_seconds()
        self.logger.info(f"Generated answer: {len(answer)} chars, {len(answer.split())} words in {generation_time:.2f}s")
        
        if feedback_callback is not None:
            feedback_callback(query, answer, context_chunks)
        
        return answer

    def _handle_special_intent(self, intent, query, mode, marks, context_chunks):
        """Handle special intents like question generation, rubric creation, etc."""
        
        if intent == 'question_generation':
            # Extract number of questions
            numbers = re.findall(r'\d+', query)
            num_questions = int(numbers[0]) if numbers else 5
            num_questions = min(num_questions, 15)  # Cap at 15
            
            # Extract topic
            topic = self._extract_topic_from_query(query)
            
            # Generate questions
            questions = self._generate_topic_questions(topic, num_questions, marks)
            
            # Format response
            response = f"**Question Paper: {topic.title()}**\n\n"
            if marks:
                response += f"*Instructions: Each question carries {marks} marks.*\n\n"
            
            for i, question in enumerate(questions, 1):
                response += f"**Q{i}.** {question}\n\n"
            
            response += f"*Total Questions: {num_questions}*\n"
            response += f"*Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC*"
            
            return response
        
        elif intent == 'rubric_creation':
            topic = self._extract_topic_from_query(query)
            marks = marks or 10  # Default to 10 marks
            
            rubric = f"**Marking Rubric: {topic.title()}**\n\n"
            rubric += f"*Total Marks: {marks}*\n\n"
            
            # Create rubric based on marks
            if marks <= 2:
                criteria = [
                    ("Definition/Understanding", marks * 0.6),
                    ("Clarity/Expression", marks * 0.4)
                ]
            elif marks <= 5:
                criteria = [
                    ("Conceptual Understanding", marks * 0.4),
                    ("Examples/Applications", marks * 0.3),
                    ("Clarity and Structure", marks * 0.3)
                ]
            else:
                criteria = [
                    ("Theoretical Understanding", marks * 0.3),
                    ("Practical Examples", marks * 0.25),
                    ("Analysis and Evaluation", marks * 0.25),
                    ("Structure and Presentation", marks * 0.2)
                ]
            
            for criterion, mark_allocation in criteria:
                rubric += f"**{criterion}** ({mark_allocation:.1f} marks)\n"
                rubric += f"- Excellent: Clear, comprehensive, accurate\n"
                rubric += f"- Good: Generally accurate with minor gaps\n"
                rubric += f"- Fair: Basic understanding with some errors\n"
                rubric += f"- Poor: Significant gaps or inaccuracies\n\n"
            
            return rubric
        
        # Add other special intent handlers as needed
        return self._generate_standard_response(query, mode, marks, context_chunks)

    def _generate_standard_response(self, query, mode, marks, context_chunks):
        """Generate standard academic response."""
        if not context_chunks:
            return "Please rephrase your question for my better understanding."
        
        # Use textbook content to generate response
        combined_content = " ".join(context_chunks[:3])
        target_words = self._get_target_words(mode, marks)
        
        # Create academic response based on content
        response = self._create_content_based_response(query, combined_content, target_words, mode)
        
        return response

    def _get_target_words(self, mode, marks):
        """Get target word count based on mode and marks."""
        if mode == "question" and marks:
            return {2: 100, 5: 250, 10: 500}.get(marks, 100)
        elif mode == "learning":
            return 300
        else:
            return 200

    def _create_content_based_response(self, query, content, target_words, mode):
        """Create response based on textbook content."""
        # Extract key information from content
        sentences = content.split('.')
        relevant_sentences = []
        
        query_words = set(query.lower().split())
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words & sentence_words)
                if overlap > 0:
                    relevant_sentences.append((sentence, overlap))
        
        # Sort by relevance
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Format response
        if mode == "learning":
            response = f"**Academic Explanation**\n\n"
            
            # Introduction
            intro_sentences = relevant_sentences[:2]
            if intro_sentences:
                response += "".join([s[0] + ". " for s in intro_sentences]) + "\n\n"
            
            # Main content
            main_sentences = relevant_sentences[2:8]
            if main_sentences:
                response += "**Key Points**\n\n"
                for i, (sentence, _) in enumerate(main_sentences, 1):
                    response += f"{i}. {sentence}.\n"
                response += "\n"
            
            # Conclusion
            response += "**Conclusion**\n\n"
            response += "Understanding these concepts is essential for academic and professional development in this field."
            
        else:  # question mode
            response = ""
            
            # Take sentences up to target word count
            word_count = 0
            selected_sentences = []
            
            for sentence, _ in relevant_sentences:
                sentence_words = len(sentence.split())
                if word_count + sentence_words <= target_words:
                    selected_sentences.append(sentence)
                    word_count += sentence_words
                else:
                    break
            
            if selected_sentences:
                response = ". ".join(selected_sentences) + "."
            else:
                response = "Please rephrase your question for my better understanding."
        
        return response

    def _generate_topic_questions(self, topic, num_questions, marks=None):
        """Generate topic-specific questions."""
        question_types = ['definition', 'explanation', 'application']
        
        if marks == 2:
            question_types = ['definition', 'short_answer']
        elif marks == 5:
            question_types = ['explanation', 'comparison', 'application']
        elif marks == 10:
            question_types = ['analysis', 'evaluation', 'synthesis']
        
        # Question templates by type
        templates = {
            'definition': [
                f"Define {topic} and explain its key characteristics.",
                f"What is {topic}? Provide a comprehensive definition.",
                f"Explain the concept of {topic} with suitable examples."
            ],
            'short_answer': [
                f"List the main features of {topic}.",
                f"What are the advantages of {topic}?",
                f"Briefly explain the importance of {topic}."
            ],
            'explanation': [
                f"Explain how {topic} works with detailed examples.",
                f"Describe the process involved in {topic}.",
                f"Explain the working principle of {topic}."
            ],
            'comparison': [
                f"Compare different types of {topic}.",
                f"What are the similarities and differences between various {topic} approaches?",
                f"Contrast the advantages and disadvantages of {topic}."
            ],
            'application': [
                f"Discuss the real-world applications of {topic}.",
                f"How is {topic} used in modern technology?",
                f"Provide examples of {topic} in practical scenarios."
            ],
            'analysis': [
                f"Analyze the impact of {topic} on modern computing.",
                f"Critically evaluate the effectiveness of {topic}.",
                f"Examine the challenges and limitations of {topic}."
            ],
            'evaluation': [
                f"Evaluate the significance of {topic} in computer science.",
                f"Assess the future prospects of {topic}.",
                f"Critically analyze the role of {topic} in solving real-world problems."
            ],
            'synthesis': [
                f"Design a system that incorporates {topic} principles.",
                f"Propose improvements to existing {topic} methods.",
                f"Synthesize information about {topic} to solve a complex problem."
            ]
        }
        
        # Generate questions using templates
        questions = []
        question_count = 0
        
        for question_type in question_types:
            if question_count >= num_questions:
                break
            
            type_templates = templates.get(question_type, templates['definition'])
            questions_needed = min(num_questions - question_count, len(type_templates))
            
            for i in range(questions_needed):
                if i < len(type_templates):
                    questions.append(type_templates[i])
                    question_count += 1
        
        # Fill remaining slots with mixed questions
        while len(questions) < num_questions:
            remaining_types = [t for t in templates.keys() if t in question_types]
            if remaining_types:
                question_type = remaining_types[len(questions) % len(remaining_types)]
                type_templates = templates[question_type]
                template_idx = len(questions) % len(type_templates)
                questions.append(type_templates[template_idx])
        
        return questions[:num_questions]

    def deduplicate_response(self, text):
        """Remove repeated sentences and paragraphs from LLM output."""
        if not text:
            return text
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Normalize sentence for comparison
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            
            # Skip very short sentences (likely fragments)
            if len(normalized.split()) < 3:
                continue
            
            # Check for exact duplicates
            if normalized not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(normalized)
        
        # Reconstruct text
        deduplicated = '. '.join(unique_sentences)
        if deduplicated and not deduplicated.endswith('.'):
            deduplicated += '.'
        
        # Additional check for repeated phrases
        deduplicated = self._remove_repeated_phrases(deduplicated)
        
        return deduplicated

    def _remove_repeated_phrases(self, text):
        """Remove repeated phrases within the text."""
        words = text.split()
        
        # Look for repeated 3-5 word phrases
        for phrase_length in range(3, 6):
            phrase_counts = Counter()
            
            # Count phrase occurrences
            for i in range(len(words) - phrase_length + 1):
                phrase = ' '.join(words[i:i + phrase_length])
                phrase_counts[phrase] += 1
            
            # Remove repeated phrases (keep only first occurrence)
            for phrase, count in phrase_counts.items():
                if count > 1:
                    # Find all occurrences and remove extras
                    phrase_pattern = re.escape(phrase)
                    matches = list(re.finditer(phrase_pattern, text))
                    
                    if len(matches) > 1:
                        # Remove all but the first occurrence
                        for match in reversed(matches[1:]):
                            text = text[:match.start()] + text[match.end():]
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _enhanced_rerank_context(self, query, context_chunks, domain, topics):
        """Enhanced context reranking with domain awareness."""
        def score_chunk(chunk):
            chunk_lower = chunk.lower()
            query_lower = query.lower()
            
            # Base score from query word matches
            query_words = set(query_lower.split())
            chunk_words = set(chunk_lower.split())
            base_score = len(query_words & chunk_words) * 10
            
            # Domain-specific bonus
            domain_bonus = 0
            if domain in self.domain_knowledge:
                domain_terms = []
                for category, terms in self.domain_knowledge[domain].items():
                    domain_terms.extend(terms)
                domain_bonus = sum(5 for term in domain_terms if term in chunk_lower)
            
            # Topic-specific bonus
            topic_bonus = sum(8 for topic in topics if topic in chunk_lower)
            
            # Academic content bonus
            academic_terms = ['definition', 'example', 'characteristic', 'principle', 'method', 'approach']
            academic_bonus = sum(3 for term in academic_terms if term in chunk_lower)
            
            total_score = base_score + domain_bonus + topic_bonus + academic_bonus
            return total_score
        
        ranked_chunks = sorted(context_chunks, key=score_chunk, reverse=True)
        return [chunk for chunk in ranked_chunks if score_chunk(chunk) > 0]

    def _is_context_relevant(self, chunk, query, domain, topics):
        """Check if context chunk is relevant to query with domain awareness."""
        chunk_lower = chunk.lower()
        query_lower = query.lower()
        
        # Check query word overlap
        query_words = set(query_lower.split())
        chunk_words = set(chunk_lower.split())
        overlap = len(query_words & chunk_words)
        
        if overlap > 0:
            return True
        
        # Check domain-specific terms
        if domain in self.domain_knowledge:
            domain_terms = []
            for category, terms in self.domain_knowledge[domain].items():
                domain_terms.extend(terms)
            if any(term in chunk_lower for term in domain_terms):
                return True
        
        # Check for academic content
        academic_indicators = ['algorithm', 'method', 'approach', 'technique', 'process', 'system', 'model', 'theory', 'principle', 'concept']
        return any(indicator in chunk_lower for indicator in academic_indicators)

    def _create_enhanced_prompt(self, query, mode, marks, target_words, context_chunks):
        """Create an enhanced prompt that encourages proper response length and structure."""
        
        parts = []
        
        # Add relevant context if available
        if context_chunks:
            context_text = context_chunks[0]
            if len(context_text) > 200:
                context_text = context_text[:200] + "..."
            parts.append(f"Context: {context_text}")
        
        # Create mode-specific instruction with word count guidance
        if mode.lower() == "learning":
            instruction = (
                "Provide a comprehensive academic explanation. "
                "Include: 1) Clear definition, 2) Key characteristics/principles, "
                "3) Multiple practical examples, 4) Applications and significance. "
                "Write 200-300 words in formal academic tone."
            )
        else:
            if target_words:
                instruction = (
                    f"Provide a complete academic answer in approximately {target_words} words. "
                    f"Include: 1) Clear definition, 2) Main characteristics, 3) Examples. "
                    f"Write exactly {target_words} words in formal academic tone. Be comprehensive within the word limit."
                )
            else:
                instruction = (
                    "Provide a concise academic answer with definition, key points, and examples. "
                    "Write 100-150 words in formal academic tone."
                )
        
        parts.append(f"Instructions: {instruction}")
        parts.append(f"Question: {query}")
        parts.append("Academic Response:")
        
        return "\n\n".join(parts)

    def _generate_with_llm(self, prompt, max_new_tokens, temperature, query, mode, target_words):
        """Generate response using the LLM with improved parameters."""
        # Tokenize and ensure within limits
        max_input_tokens = 1024 - max_new_tokens - 30
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        # Truncation if needed
        if len(prompt_ids) > max_input_tokens:
            excess = len(prompt_ids) - max_input_tokens
            prompt_ids = prompt_ids[excess:]
        
        # Prepare tensors
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        
        try:
            # Generate response with improved parameters
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=50,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.1,
                    use_cache=True
                )
            
            # Decode and clean response
            result = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = self._extract_and_clean_response(result, prompt, prompt_ids)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Exception occurred: {repr(e)}")
            
            # Fallback
            return "Please rephrase your question for my better understanding."

    def _extract_and_clean_response(self, result, prompt, prompt_ids):
        """Extract and clean the model's response from the full output."""
        
        # Extract only the model's response (remove prompt)
        prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        if prompt_text in result:
            answer = result.split(prompt_text, 1)[-1].strip()
        else:
            # Fallback: try to find "Academic Response:" marker
            if "Academic Response:" in result:
                answer = result.split("Academic Response:", 1)[-1].strip()
            else:
                answer = result.strip()
        
        # Clean up DialoGPT output issues
        answer = re.sub(r'<\|endoftext\|>.*$', '', answer, flags=re.DOTALL).strip()
        answer = re.sub(r'<pad>.*$', '', answer, flags=re.DOTALL).strip()
        answer = re.sub(r'<unk>.*$', '', answer, flags=re.DOTALL).strip()
        answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)  # Clean up excessive newlines
        
        # Remove repetitive patterns but be less aggressive
        lines = answer.split('\n')
        cleaned_lines = []
        prev_line = ""
        for line in lines:
            line = line.strip()
            if line and (line != prev_line or len(line) < 30):  # Allow some repetition for structure
                cleaned_lines.append(line)
            prev_line = line
        
        return '\n'.join(cleaned_lines)

    def _enhance_and_expand_response(self, answer, query, mode, target_words):
        """Enhanced response expansion with better word count targeting."""
        
        if not answer.strip():
            return "Please rephrase your question for my better understanding."
        
        # Check if response needs enhancement
        current_words = len(answer.split())
        
        # Add academic structure if missing
        if not any(marker in answer.lower() for marker in ['definition:', '**definition', 'characteristics:', 'examples:']):
            if mode.lower() == "learning":
                structured = f"**Academic Explanation:**\n\n{answer}\n\nThis provides a comprehensive overview of the topic with practical implications for further study and application."
            else:
                structured = f"**Academic Answer:**\n{answer}"
            return structured
        
        return answer

    def _final_quality_check(self, answer, query, mode, target_words):
        """Perform final quality checks with better word count enforcement."""
        
        # Remove any remaining artifacts
        answer = re.sub(r'\\\+', '*', answer)
        answer = re.sub(r'---+', '', answer)
        answer = re.sub(r'===+', '', answer)
        
        # Ensure proper capitalization
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Ensure proper ending
        if answer and not answer.rstrip().endswith(('.', '!', '?', ':')):
            answer = answer.rstrip() + '.'
        
        # Minimum length check for learning mode
        if mode.lower() == "learning" and len(answer.split()) < 80:
            answer += "\n\nThis overview provides essential foundational knowledge for understanding the topic. Further exploration through academic literature and practical experience will deepen comprehension and enable advanced application of these concepts."
        
        return answer.strip()

    def _extract_topic_from_query(self, query):
        """Extract meaningful topic from query with enhanced processing."""
        # Remove common question words and phrases
        topic = re.sub(r'\b(what|is|are|how|why|when|where|explain|define|describe|tell|me|about|the|a|an|generate|create|make|questions?|for|of)\b', '', query.lower(), flags=re.IGNORECASE)
        topic = re.sub(r'[^\w\s]', '', topic).strip()
        topic = re.sub(r'\s+', ' ', topic)
        
        # Remove numbers (often from question generation requests)
        topic = re.sub(r'\b\d+\b', '', topic).strip()
        
        # Take meaningful words
        words = [w for w in topic.split() if len(w) > 2][:4]
        
        if words:
            return ' '.join(words)
        else:
            return "Academic Topic"

    def _infer_target_word_count(self, marks):
        """Infer target word count from marks."""
        if marks is None:
            return None
        return {2: 100, 5: 250, 10: 500}.get(marks, 100)

    def generate_dual_response(self, query, mode="learning", marks=None, temperature=0.7, context_chunks=None):
        """
        Generate both LLM and enhanced custom academic responses independently using parallel execution.
        
        Args:
            query: The user's question/prompt
            mode: "learning" for detailed responses, "question" for concise answers
            marks: For question mode, affects target word count (2/5/10 -> 100/250/500 words)
            temperature: Sampling temperature for LLM generation
            context_chunks: List of relevant text chunks to include as context
        
        Returns:
            Dict with both outputs and metadata
        """
        start_time = datetime.utcnow()
        context_chunks = context_chunks or []
        target_words = self._infer_target_word_count(marks)
        
        self.logger.info(f"=== DUAL RESPONSE GENERATION ===")
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Mode: {mode}, Marks: {marks}, Target words: {target_words}")
        self.logger.info(f"Context chunks: {len(context_chunks)}")
        
        # Determine intent and domain for metadata
        intent, _, _ = self.detect_query_intent(query)
        domain, topics, _ = self.detect_domain_and_topic(query)
        
        # Use ThreadPoolExecutor for parallel generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Future for LLM generation
            llm_future = executor.submit(
                self.generate_answer,
                query=query,
                mode=mode,
                marks=marks,
                temperature=temperature,
                context_chunks=context_chunks,
                rerank_context=True,
                return_citations=True,
                max_new_tokens=500  # Cap for LLM output
            )
            
            # Future for custom academic generation (handles full word count using TextbookLLM)
            custom_future = executor.submit(
                self._generate_custom_response,
                query=query,
                mode=mode,
                marks=marks,
                target_words=target_words,
                context_chunks=context_chunks
            )
            
            llm_answer = ""
            llm_generation_time = 0.0
            llm_word_count = 0
            
            custom_answer = ""
            custom_generation_time = 0.0
            custom_word_count = 0
            
            try:
                llm_start_time = datetime.utcnow()
                self.logger.info(f"--- Starting LLM Generation (Parallel) ---")
                llm_answer = llm_future.result()
                llm_generation_time = (datetime.utcnow() - llm_start_time).total_seconds()
                llm_word_count = len(llm_answer.split())
                self.logger.info(f"LLM generation successful: {llm_word_count} words in {llm_generation_time:.2f}s")
            except Exception as e:
                self.logger.error(f"LLM generation failed: {repr(e)}")
                llm_answer = "Please rephrase your question for my better understanding."
                llm_generation_time = (datetime.utcnow() - llm_start_time).total_seconds()
                llm_word_count = len(llm_answer.split())
                self.logger.info(f"LLM fallback used: {llm_word_count} words in {llm_generation_time:.2f}s")
            
            try:
                custom_start_time = datetime.utcnow()
                self.logger.info(f"--- Starting Custom Academic Generation (Parallel) ---")
                custom_answer = custom_future.result()
                custom_generation_time = (datetime.utcnow() - custom_start_time).total_seconds()
                custom_word_count = len(custom_answer.split())
                self.logger.info(f"Custom generation successful: {custom_word_count} words in {custom_generation_time:.2f}s")
            except Exception as e:
                self.logger.error(f"Custom generation failed: {repr(e)}")
                custom_answer = "Please rephrase your question for my better understanding."
                custom_generation_time = (datetime.utcnow() - custom_start_time).total_seconds()
                custom_word_count = len(custom_answer.split())
                self.logger.info(f"Custom fallback used: {custom_word_count} words in {custom_generation_time:.2f}s")
        
        total_time = (datetime.utcnow() - start_time).total_seconds()
        self.logger.info(f"=== DUAL GENERATION COMPLETE ===")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"LLM: {llm_word_count} words, Custom: {custom_word_count} words")
        
        # Return structured response
        result = {
            "llm_output": llm_answer,
            "custom_output": custom_answer,
            "intent": intent,
            "domain": domain,
            "topics": topics,
            "word_counts": {
                "llm": llm_word_count,
                "custom": custom_word_count
            },
            "generation_times": {
                "llm": llm_generation_time,
                "custom": custom_generation_time,
                "total": total_time
            }
        }
        
        return result

    def _generate_custom_response(self, query, mode, marks, target_words, context_chunks):
        """Generate custom academic response using TextbookLLM for true RAG."""
        
        # Use TextbookLLM if available
        if self.textbook_llm:
            try:
                answer = self.textbook_llm.generate_answer(query, mode, marks, target_words or 300)
                self.logger.info(f"TextbookLLM generated response: {len(answer.split())} words")
                return answer
            except Exception as e:
                self.logger.warning(f"TextbookLLM failed: {e}")
                return "Please rephrase your question for my better understanding."
        else:
            # Fallback if TextbookLLM not available
            return "Please rephrase your question for my better understanding."

    def get_model_info(self):
        """Return information about the loaded model."""
        return {
            "model_name": getattr(self.model.config, '_name_or_path', 'unknown'),
            "model_type": getattr(self.model.config, 'model_type', 'unknown'),
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": 1024,  # DialoGPT limit
            "device": self.device,
            "current_user": "QuillAI",
            "session_time": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + " UTC",
            "semantic_model_available": self.semantic_model is not None
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Initializing QuillAI LLM with Advanced Features...")
    print(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)
    
    try:
        # Initialize model
        model = QuillAILLM(model_name="microsoft/DialoGPT-medium", force_model_check=True)
        
        # Test intent detection
        query = "What is machine learning?"
        intent, confidence, all_intents = model.detect_query_intent(query)
        print(f"Intent: {intent} (confidence: {confidence:.2f})")
        
        # Test domain detection
        domain, topics, domain_confidence = model.detect_domain_and_topic(query)
        print(f"Domain: {domain}, Topics: {topics}")
        
        print("QuillAI LLM initialized successfully")
        
    except Exception as ex:
        print(f"FAIL: {ex}")