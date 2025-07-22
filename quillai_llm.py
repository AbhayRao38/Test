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

class ExtractiveCustomLLM:
    """
    Deterministic extractive academic response generator using only retrieval and semantic ranking.
    No generative LLM components - purely extractive and rule-based.
    """
    
    def __init__(self, retrieval_system=None):
        self.retrieval_system = retrieval_system
        self.logger = logging.getLogger("extractive_custom_llm")
        
        # Academic terms for quality scoring
        self.academic_terms = {
            'definition': ['define', 'definition', 'is a', 'refers to', 'means', 'concept'],
            'explanation': ['explain', 'because', 'due to', 'therefore', 'thus', 'hence'],
            'example': ['example', 'instance', 'such as', 'for example', 'including'],
            'comparison': ['compare', 'contrast', 'similar', 'different', 'whereas'],
            'application': ['application', 'used', 'applied', 'implementation', 'practice'],
            'analysis': ['analysis', 'analyze', 'examine', 'evaluate', 'assess'],
            'conclusion': ['conclusion', 'summary', 'finally', 'in conclusion', 'overall']
        }
        
        # Academic connectors and templates
        self.academic_connectors = [
            "Furthermore,", "Additionally,", "Moreover,", "In addition,",
            "However,", "Nevertheless,", "Consequently,", "Therefore,"
        ]
        
        self.logger.info("✅ ExtractiveCustomLLM initialized")

    def generate_academic_response(self, query, mode="learning", marks=None, context_chunks=None):
        """Generate deterministic academic response using only extractive methods."""
        
        if not context_chunks:
            return self._generate_fallback_response(query, mode, marks)
        
        # Target word count based on mode and marks
        target_words = self._get_target_word_count(mode, marks)
        
        # Detect query type for structured response
        query_type = self._detect_query_type(query)
        
        # Extract and rank relevant sentences
        relevant_sentences = self._extract_and_rank_sentences(query, context_chunks)
        
        if not relevant_sentences:
            return self._generate_fallback_response(query, mode, marks)
        
        # Generate structured academic response based on query type
        if query_type in ['explain', 'define', 'describe', 'what_is']:
            response = self._generate_structured_academic_response(
                query, context_chunks, target_words, mode
            )
        else:
            response = self._synthesize_academic_response(
                query, relevant_sentences, target_words, mode
            )
        
        # Quality check and ensure minimum word count
        response = self._ensure_quality_and_length(response, target_words, query)
        
        return response

    def _get_target_word_count(self, mode, marks):
        """Get target word count based on mode and marks."""
        if mode == "question" and marks:
            return {2: 100, 5: 250, 10: 500}.get(marks, 100)
        elif mode == "learning":
            return 300
        else:
            return 200

    def _detect_query_type(self, query):
        """Detect the type of query to determine response structure."""
        query_lower = query.lower().strip()
        
        # Patterns for different query types
        if any(pattern in query_lower for pattern in ['what is', 'what are', 'define', 'definition of']):
            return 'define'
        elif any(pattern in query_lower for pattern in ['explain', 'how does', 'how do', 'describe']):
            return 'explain'
        elif query_lower.startswith('describe'):
            return 'describe'
        elif any(pattern in query_lower for pattern in ['tell me about', 'give me information about']):
            return 'explain'
        else:
            return 'general'

    def _extract_and_rank_sentences(self, query, context_chunks):
        """Extract and rank sentences by relevance to query."""
        query_words = set(query.lower().split())
        query_words = {w for w in query_words if len(w) > 2}  # Filter short words
        
        all_sentences = []
        
        for chunk in context_chunks:
            # Split into sentences
            sentences = re.split(r'[.!?]+', chunk)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                # Calculate relevance score
                score = self._calculate_sentence_relevance(sentence, query_words)
                
                if score > 0:
                    all_sentences.append({
                        'text': sentence,
                        'score': score,
                        'word_count': len(sentence.split())
                    })
        
        # Sort by score and remove duplicates
        all_sentences.sort(key=lambda x: x['score'], reverse=True)
        return self._deduplicate_sentences(all_sentences)

    def _calculate_sentence_relevance(self, sentence, query_words):
        """Calculate relevance score for a sentence based on query words and academic terms."""
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())
        
        # Base score from query word overlap
        overlap = len(query_words & sentence_words)
        base_score = overlap * 10
        
        # Academic term bonus
        academic_bonus = 0
        for category, terms in self.academic_terms.items():
            for term in terms:
                if term in sentence_lower:
                    academic_bonus += 5
        
        # Length bonus (prefer medium-length sentences)
        word_count = len(sentence.split())
        if 15 <= word_count <= 40:
            length_bonus = 3
        elif 10 <= word_count < 15 or 40 < word_count <= 60:
            length_bonus = 1
        else:
            length_bonus = 0
        
        # Definition/explanation bonus
        if any(indicator in sentence_lower for indicator in ['is a', 'is an', 'refers to', 'means']):
            definition_bonus = 8
        else:
            definition_bonus = 0
        
        return base_score + academic_bonus + length_bonus + definition_bonus

    def _deduplicate_sentences(self, sentences):
        """Remove duplicate and highly similar sentences."""
        unique_sentences = []
        seen_content = set()
        
        for sentence_data in sentences:
            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', sentence_data['text'].lower().strip())
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            # Check for exact or high similarity
            is_duplicate = False
            for seen in seen_content:
                similarity = self._calculate_text_similarity(normalized, seen)
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sentences.append(sentence_data)
                seen_content.add(normalized)
        
        return unique_sentences

    def _calculate_text_similarity(self, text1, text2):
        """Calculate simple word-based similarity between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0

    def _generate_structured_academic_response(self, query, context_chunks, target_words, mode):
        """Generate structured academic response with clear sections."""
        # Extract all sentences from all chunks
        all_sentences = []
        for chunk in context_chunks:
            sentences = re.split(r'[.!?]+', chunk)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15:  # Minimum meaningful sentence length
                    all_sentences.append(sentence)
        
        if not all_sentences:
            return self._generate_fallback_response(query, mode, None)
        
        # Extract content for each section
        sections = self._extract_academic_sections(query, all_sentences)
        
        # Build structured response
        response_parts = []
        
        if sections['definition']:
            response_parts.append("**Definition:**\n")
            response_parts.append(self._format_section_content(sections['definition']))
            response_parts.append("\n\n")
        
        if sections['explanation']:
            response_parts.append("**Key Principles / Explanation:**\n")
            response_parts.append(self._format_section_content(sections['explanation']))
            response_parts.append("\n\n")
        
        if sections['applications']:
            response_parts.append("**Applications:**\n")
            response_parts.append(self._format_section_content(sections['applications']))
            response_parts.append("\n\n")
        
        if sections['examples']:
            response_parts.append("**Examples:**\n")
            response_parts.append(self._format_section_content(sections['examples']))
            response_parts.append("\n\n")
        
        if not response_parts:
            # No structured content found, fall back to general synthesis
            relevant_sentences = self._extract_and_rank_sentences(query, context_chunks)
            return self._synthesize_academic_response(query, relevant_sentences, target_words, mode)
        
        # Combine and clean up
        response = "".join(response_parts).strip()
        
        # Ensure minimum word count by adding general content if needed
        current_words = len(response.split())
        if current_words < 50 and len(all_sentences) > 0:
            # Add some general relevant sentences
            query_words = set(query.lower().split())
            relevant_general = []
            for sentence in all_sentences[:10]:  # Check first 10 sentences
                sentence_words = set(sentence.lower().split())
                if len(query_words & sentence_words) > 0:
                    relevant_general.append(sentence)
            
            if relevant_general and len(response_parts) > 0:
                response += "\n\n**Additional Information:**\n"
                response += ". ".join(relevant_general[:3]) + "."
        
        return response

    def _extract_academic_sections(self, query, sentences):
        """Extract sentences for different academic sections."""
        sections = {
            'definition': [],
            'explanation': [],
            'applications': [],
            'examples': []
        }
        
        # Extract key terms from query for matching
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Patterns for each section
        definition_patterns = [
            r'\b\w+\s+is\s+(?:a|an)\s+',
            r'\b\w+\s+refers\s+to\s+',
            r'\b\w+\s+can\s+be\s+defined\s+as\s+',
            r'\b\w+\s+means\s+',
            r'definition\s+of\s+',
            r'defined\s+as\s+',
            r'known\s+as\s+'
        ]
        
        explanation_patterns = [
            r'\bworks?\s+by\b',
            r'\bfunctions?\s+by\b',
            r'\boperates?\s+by\b',
            r'\bmechanism\b',
            r'\bprinciple\b',
            r'\bcharacteristic\b',
            r'\bfeature\b',
            r'\bproperty\b',
            r'\bhow\s+it\s+works\b',
            r'\bprocess\s+of\b'
        ]
        
        application_patterns = [
            r'\bapplicat\w+\b',
            r'\bused\s+(?:in|for|to)\b',
            r'\bapplied\s+(?:in|to)\b',
            r'\breal.world\b',
            r'\bpractical\b',
            r'\butility\b',
            r'\bbenefit\b',
            r'\bimplement\w+\b',
            r'\bindustry\b',
            r'\bcommercial\b'
        ]
        
        example_patterns = [
            r'\bfor\s+example\b',
            r'\bsuch\s+as\b',
            r'\be\.g\.\b',
            r'\bincluding\b',
            r'\bfor\s+instance\b',
            r'\bnamely\b',
            r'\bspecifically\b'
        ]
        
        used_sentences = set()
        
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean or len(sentence_clean) < 20:
                continue
            
            sentence_lower = sentence_clean.lower()
            
            # Skip if already used
            normalized = re.sub(r'\s+', ' ', sentence_lower.strip())
            if normalized in used_sentences:
                continue
            
            # Check relevance to query
            sentence_words = set(sentence_lower.split())
            relevance = len(query_words & sentence_words)
            
            if relevance == 0:
                continue
            
            # Categorize sentence
            if any(re.search(pattern, sentence_lower) for pattern in definition_patterns):
                sections['definition'].append(sentence_clean)
                used_sentences.add(normalized)
            elif any(re.search(pattern, sentence_lower) for pattern in explanation_patterns):
                sections['explanation'].append(sentence_clean)
                used_sentences.add(normalized)
            elif any(re.search(pattern, sentence_lower) for pattern in application_patterns):
                sections['applications'].append(sentence_clean)
                used_sentences.add(normalized)
            elif any(re.search(pattern, sentence_lower) for pattern in example_patterns):
                sections['examples'].append(sentence_clean)
                used_sentences.add(normalized)
            elif relevance >= 2:  # High relevance sentences go to explanation
                sections['explanation'].append(sentence_clean)
                used_sentences.add(normalized)
        
        # Limit sentences per section and prioritize by relevance
        for section_name in sections:
            if len(sections[section_name]) > 3:
                # Sort by length and relevance, keep top 3
                sections[section_name] = sections[section_name][:3]
        
        return sections

    def _format_section_content(self, sentences):
        """Format sentences for a section."""
        if not sentences:
            return ""
        
        # Remove duplicates while preserving order
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            if normalized not in seen:
                unique_sentences.append(sentence)
                seen.add(normalized)
        
        if not unique_sentences:
            return ""
        
        # Join sentences properly
        formatted = ". ".join(s.rstrip('.') for s in unique_sentences) + "."
        
        return formatted

    def _synthesize_academic_response(self, query, relevant_sentences, target_words, mode):
        """Synthesize academic response from ranked sentences."""
        if mode == "learning":
            return self._generate_learning_response(query, relevant_sentences, target_words)
        else:
            return self._generate_question_response(query, relevant_sentences, target_words)

    def _generate_learning_response(self, query, relevant_sentences, target_words):
        """Generate comprehensive learning response."""
        response_parts = []
        current_word_count = 0
        
        # Introduction
        intro = f"**Academic Overview:**\n\n"
        response_parts.append(intro)
        current_word_count += len(intro.split())
        
        # Main content - select best sentences up to target
        used_sentences = []
        for sentence_data in relevant_sentences:
            sentence = sentence_data['text']
            sentence_words = sentence_data['word_count']
            
            if current_word_count + sentence_words <= target_words * 0.8:  # Leave room for conclusion
                used_sentences.append(sentence)
                current_word_count += sentence_words
            else:
                break
        
        if used_sentences:
            # Format with academic connectors
            formatted_content = self._format_with_connectors(used_sentences)
            response_parts.append(formatted_content)
            response_parts.append("\n\n")
        
        # Conclusion
        conclusion = "**Key Insights:**\n\nThese concepts form essential knowledge for academic understanding and practical application in the field."
        response_parts.append(conclusion)
        
        return "".join(response_parts).strip()

    def _generate_question_response(self, query, relevant_sentences, target_words):
        """Generate concise question response."""
        response_parts = []
        current_word_count = 0
        
        # Select sentences up to target word count
        for sentence_data in relevant_sentences:
            sentence = sentence_data['text']
            sentence_words = sentence_data['word_count']
            
            if current_word_count + sentence_words <= target_words:
                response_parts.append(sentence)
                current_word_count += sentence_words
            else:
                break
        
        if response_parts:
            response = ". ".join(response_parts) + "."
            return self._clean_academic_text(response)
        else:
            return self._generate_fallback_response(query, "question", None)

    def _format_with_connectors(self, sentences):
        """Format sentences with academic connectors."""
        if not sentences:
            return ""
        
        formatted = []
        for i, sentence in enumerate(sentences):
            if i == 0:
                formatted.append(sentence)
            elif i < len(self.academic_connectors):
                connector = self.academic_connectors[i-1]
                formatted.append(f"{connector} {sentence.lower()}")
            else:
                formatted.append(sentence)
        
        return ". ".join(formatted) + "."

    def _clean_academic_text(self, text):
        """Clean and standardize academic text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper capitalization
        text = text.strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure proper ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text

    def _ensure_quality_and_length(self, response, target_words, query):
        """Ensure response meets quality and length requirements."""
        word_count = len(response.split())
        
        # Minimum word count check
        if word_count < 50:
            fallback = self._generate_fallback_response(query, "learning", None)
            return fallback
        
        # Quality check - ensure academic content
        if not self._is_academic_quality(response):
            return self._generate_fallback_response(query, "learning", None)
        
        return response

    def _is_academic_quality(self, text):
        """Check if text meets academic quality standards."""
        text_lower = text.lower()
        
        # Check for academic indicators
        academic_indicators = [
            'definition', 'concept', 'principle', 'method', 'approach',
            'analysis', 'example', 'application', 'theory', 'research'
        ]
        
        has_academic_content = any(indicator in text_lower for indicator in academic_indicators)
        
        # Check minimum length and structure
        has_minimum_length = len(text.split()) >= 20
        has_proper_structure = '.' in text or '**' in text
        
        return has_academic_content and has_minimum_length and has_proper_structure

    def _generate_fallback_response(self, query, mode, marks):
        """Generate fallback academic response when context is insufficient."""
        target_words = self._get_target_word_count(mode, marks)
        
        if mode == "learning":
            fallback = (
                "**Academic Response:**\n\n"
                "I apologize, but I require more specific textbook content to provide a comprehensive "
                "academic explanation for your query. To better assist you, please ensure that relevant "
                "academic materials have been uploaded to the knowledge base, or consider rephrasing "
                "your question to be more specific about the topic you're studying.\n\n"
                "**Recommendation:**\n\nFor optimal results, upload relevant textbooks or academic "
                "materials that cover the topic you're asking about. This will enable me to provide "
                "detailed, evidence-based academic explanations with proper citations and examples."
            )
        else:
            fallback = (
                "I apologize, but I require more specific context from uploaded academic materials "
                "to provide a comprehensive answer to your question. Please ensure relevant textbooks "
                "or academic sources have been added to the knowledge base, or rephrase your question "
                "with more specific details about the topic you're studying."
            )
        
        return fallback


class DialogGPTAcademicLLM:
    """
    Academic-focused DialogGPT implementation using ONLY microsoft/DialoGPT-medium.
    Includes robust quality controls and academic post-processing.
    """
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        if model_name != "microsoft/DialoGPT-medium":
            raise ValueError(f"Only microsoft/DialoGPT-medium is allowed. Attempted: {model_name}")
        
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger("dialogpt_academic")
        
        self.logger.info(f"Loading DialogGPT Academic LLM: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                device_map=None
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"✅ DialogGPT loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load DialogGPT: {e}")
            raise

    def generate_academic_response(self, query, mode="learning", marks=None, context_chunks=None):
        """Generate academic response using DialogGPT with quality controls."""
        
        # Create academic prompt
        prompt = self._create_academic_prompt(query, mode, marks, context_chunks)
        
        # Generate with DialogGPT
        response = self._generate_with_dialogpt(prompt, marks)
        
        # Post-process for academic quality
        response = self._post_process_academic(response, query, mode, marks)
        
        # Final quality check
        response = self._final_quality_check(response, query, mode, marks)
        
        return response

    def _create_academic_prompt(self, query, mode, marks, context_chunks):
        """Create academic-focused prompt for DialogGPT."""
        prompt_parts = []
        
        # Context if available
        if context_chunks:
            context = " ".join(context_chunks[:2])[:500]  # Limit context length
            prompt_parts.append(f"Academic Context: {context}")
        
        # Mode-specific instructions
        if mode == "learning":
            instruction = (
                "Provide a comprehensive academic explanation with clear definitions, "
                "key principles, examples, and applications. Use formal academic language "
                "and structure. Write approximately 200-300 words."
            )
        else:
            target_words = {2: 100, 5: 250, 10: 500}.get(marks, 100) if marks else 100
            instruction = (
                f"Provide a precise academic answer in exactly {target_words} words. "
                f"Include definition, key points, and examples. Use formal academic tone."
            )
        
        prompt_parts.extend([
            f"Instructions: {instruction}",
            f"Academic Question: {query}",
            "Academic Response:"
        ])
        
        return "\n\n".join(prompt_parts)

    def _generate_with_dialogpt(self, prompt, marks):
        """Generate response using DialogGPT with improved parameters."""
        try:
            # Tokenize prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            
            # Calculate max tokens
            if marks:
                target_words = {2: 100, 5: 250, 10: 500}.get(marks, 100)
                max_new_tokens = min(target_words * 2, 400)
            else:
                max_new_tokens = 300
            
            # Generate with DialogGPT
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=30,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if "Academic Response:" in full_text:
                response = full_text.split("Academic Response:")[-1].strip()
            else:
                # Extract everything after the prompt
                prompt_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
                if prompt_text in full_text:
                    response = full_text.replace(prompt_text, "").strip()
                else:
                    response = full_text.strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"DialogGPT generation failed: {e}")
            return ""

    def _post_process_academic(self, response, query, mode, marks):
        """Post-process DialogGPT response for academic quality."""
        if not response:
            return ""
        
        # Clean up DialogGPT artifacts
        response = re.sub(r'<\|.*?\|>', '', response)
        response = re.sub(r'<pad>.*$', '', response, flags=re.DOTALL)
        response = re.sub(r'<unk>.*$', '', response, flags=re.DOTALL)
        response = re.sub(r'<\|endoftext\|>.*$', '', response, flags=re.DOTALL)
        
        # Remove repetitive patterns
        response = self._remove_repetitions(response)
        
        # Clean excessive whitespace
        response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
        response = re.sub(r'\s+', ' ', response)
        
        # Ensure proper capitalization
        response = response.strip()
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Ensure proper ending
        if response and not response.endswith(('.', '!', '?')):
            response = response.rstrip() + '.'
        
        return response

    def _remove_repetitions(self, text):
        """Remove repetitive sentences and phrases."""
        sentences = re.split(r'[.!?]+', text)
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Normalize for comparison
            normalized = re.sub(r'\s+', ' ', sentence.lower())
            
            if normalized not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(normalized)
        
        return '. '.join(unique_sentences) + '.' if unique_sentences else text

    def _final_quality_check(self, response, query, mode, marks):
        """Final quality check with fallback if needed - less preemptive."""
        word_count = len(response.split())
        
        # Check minimum length
        if word_count < 20:
            return self._generate_fallback_response(query, mode, marks)
        
        # Check for gibberish patterns (more strict criteria)
        if self._is_severe_gibberish(response):
            return self._generate_fallback_response(query, mode, marks)
        
        # Only fallback if response is completely non-academic and very short
        if word_count < 30 and not self._has_any_meaningful_content(response):
            return self._generate_fallback_response(query, mode, marks)
        
        return response

    def _is_severe_gibberish(self, text):
        """Check if text contains severe gibberish patterns - more strict than before."""
        # Check for repeated characters (5+ repetitions)
        if re.search(r'(.)\1{5,}', text):
            return True
        
        # Check for excessive nonsensical character patterns
        words = text.split()
        nonsensical_words = 0
        for word in words:
            if len(word) > 4 and not re.match(r'^[a-zA-Z]+$', word):
                nonsensical_words += 1
        
        # If more than 50% of words are nonsensical, it's severe gibberish
        if words and nonsensical_words / len(words) > 0.5:
            return True
        
        # Check for completely incoherent patterns
        if re.search(r'[a-zA-Z]{15,}', text) and not re.search(r'\s', text):
            return True
        
        return False

    def _has_any_meaningful_content(self, text):
        """Check if text has any meaningful content - less strict than academic check."""
        text_lower = text.lower()
        
        # Basic meaningful indicators
        meaningful_indicators = [
            'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or',
            'definition', 'concept', 'principle', 'method', 'approach',
            'theory', 'analysis', 'example', 'application', 'research',
            'study', 'academic', 'knowledge', 'understanding', 'used',
            'works', 'means', 'refers', 'includes', 'such', 'like'
        ]
        
        # Must have at least some basic English structure
        meaningful_count = sum(1 for indicator in meaningful_indicators if indicator in text_lower)
        words = text_lower.split()
        
        if not words:
            return False
        
        # At least 10% of words should be meaningful
        return meaningful_count / len(words) >= 0.1

    def _is_gibberish(self, text):
        """Check if text contains gibberish patterns."""
        # Check for repeated characters
        if re.search(r'(.)\1{4,}', text):
            return True
        
        # Check for nonsensical character patterns
        words = text.split()
        nonsensical_words = 0
        for word in words:
            if len(word) > 3 and not re.match(r'^[a-zA-Z]+$', word):
                nonsensical_words += 1
        
        # If more than 30% of words are nonsensical, it's likely gibberish
        if words and nonsensical_words / len(words) > 0.3:
            return True
        
        return False

    def _has_academic_content(self, text):
        """Check if text has academic content."""
        text_lower = text.lower()
        
        academic_indicators = [
            'definition', 'concept', 'principle', 'method', 'approach',
            'theory', 'analysis', 'example', 'application', 'research',
            'study', 'academic', 'knowledge', 'understanding'
        ]
        
        return any(indicator in text_lower for indicator in academic_indicators)

    def _generate_fallback_response(self, query, mode, marks):
        """Generate academic fallback response."""
        if mode == "learning":
            return (
                "I apologize, but I'm unable to generate a comprehensive academic response "
                "to your query at this time. This may be due to the complexity of the question "
                "or limitations in the available context. Please consider rephrasing your question "
                "with more specific details, or ensure that relevant academic materials have been "
                "uploaded to provide better context for your inquiry."
            )
        else:
            return (
                "I apologize, but I cannot provide a complete academic answer to your question "
                "at this time. Please rephrase your question with more specific details or "
                "ensure relevant academic context is available."
            )


class QuillAILLM:
    """
    Main QuillAI LLM system with dual output: DialogGPT + Extractive Custom LLM.
    """
    
    def __init__(self, model_name="microsoft/DialoGPT-medium", force_model_check=True, debug_mode=False):
        if force_model_check and model_name != "microsoft/DialoGPT-medium":
            raise ValueError(f"Only microsoft/DialoGPT-medium is allowed. Attempted: {model_name}")
        
        self.model_name = model_name
        self.debug_mode = debug_mode
        self.logger = logging.getLogger("quillai_llm")
        
        # Initialize DialogGPT component
        self.dialogpt_llm = DialogGPTAcademicLLM(model_name)
        
        # Initialize extractive custom component
        self.custom_llm = None  # Will be set when retrieval system is available
        
        self.logger.info(f"✅ QuillAI LLM initialized with {model_name}")

    def set_retrieval_system(self, retrieval_system):
        """Set the retrieval system for the extractive custom LLM."""
        try:
            self.custom_llm = ExtractiveCustomLLM(retrieval_system)
            self.logger.info("✅ Extractive Custom LLM initialized with retrieval system")
        except Exception as e:
            self.logger.warning(f"⚠ Warning: Could not initialize Custom LLM: {e}")
            self.custom_llm = None

    def generate_dual_response(self, query, mode="learning", marks=None, context_chunks=None):
        """
        Generate both DialogGPT and extractive custom responses in parallel.
        
        Returns:
            Dict with both outputs and metadata
        """
        start_time = datetime.utcnow()
        context_chunks = context_chunks or []
        
        self.logger.info(f"=== DUAL RESPONSE GENERATION ===")
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Mode: {mode}, Marks: {marks}")
        self.logger.info(f"Context chunks: {len(context_chunks)}")
        
        # Use ThreadPoolExecutor for parallel generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Future for DialogGPT generation
            dialogpt_future = executor.submit(
                self.dialogpt_llm.generate_academic_response,
                query, mode, marks, context_chunks
            )
            
            # Future for extractive custom generation
            custom_future = executor.submit(
                self._generate_custom_response,
                query, mode, marks, context_chunks
            )
            
            # Collect results
            try:
                dialogpt_start = datetime.utcnow()
                dialogpt_output = dialogpt_future.result()
                dialogpt_time = (datetime.utcnow() - dialogpt_start).total_seconds()
                dialogpt_words = len(dialogpt_output.split())
                self.logger.info(f"DialogGPT: {dialogpt_words} words in {dialogpt_time:.2f}s")
            except Exception as e:
                self.logger.error(f"DialogGPT generation failed: {e}")
                dialogpt_output = "I apologize, but I'm unable to generate a response at this time. Please try rephrasing your question."
                dialogpt_time = 0.0
                dialogpt_words = len(dialogpt_output.split())
            
            try:
                custom_start = datetime.utcnow()
                custom_output = custom_future.result()
                custom_time = (datetime.utcnow() - custom_start).total_seconds()
                custom_words = len(custom_output.split())
                self.logger.info(f"Custom: {custom_words} words in {custom_time:.2f}s")
            except Exception as e:
                self.logger.error(f"Custom generation failed: {e}")
                custom_output = "I apologize, but I require more specific context from academic materials to provide a comprehensive response."
                custom_time = 0.0
                custom_words = len(custom_output.split())
        
        total_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.logger.info(f"=== DUAL GENERATION COMPLETE ===")
        self.logger.info(f"Total time: {total_time:.2f}s")
        
        # Return structured response
        return {
            "llm_output": dialogpt_output,
            "custom_output": custom_output,
            "intent": "academic_query",
            "domain": "general",
            "topics": [],
            "word_counts": {
                "llm": dialogpt_words,
                "custom": custom_words
            },
            "generation_times": {
                "llm": dialogpt_time,
                "custom": custom_time,
                "total": total_time
            }
        }

    def _generate_custom_response(self, query, mode, marks, context_chunks):
        """Generate custom extractive response."""
        if self.custom_llm:
            return self.custom_llm.generate_academic_response(query, mode, marks, context_chunks)
        else:
            return (
                "I apologize, but the extractive response system requires relevant academic "
                "materials to be uploaded to the knowledge base. Please add textbooks or "
                "academic sources to enable comprehensive academic responses."
            )

    def generate_answer(self, query, mode="learning", marks=None, context_chunks=None, **kwargs):
        """Legacy compatibility method - returns only DialogGPT output."""
        return self.dialogpt_llm.generate_academic_response(query, mode, marks, context_chunks)

    def get_model_info(self):
        """Return information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_type": "DialogGPT + Extractive Custom",
            "device": self.dialogpt_llm.device,
            "current_user": "QuillAI",
            "session_time": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') + " UTC",
            "custom_llm_available": self.custom_llm is not None
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Initializing QuillAI LLM with Dual Output System...")
    print(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)
    
    try:
        # Initialize model
        model = QuillAILLM(model_name="microsoft/DialoGPT-medium", force_model_check=True)
        print("✅ QuillAI LLM initialized successfully")
        
        # Test dual response
        test_query = "What is machine learning?"
        result = model.generate_dual_response(test_query, mode="learning")
        
        print(f"\nDialogGPT Output: {result['llm_output'][:100]}...")
        print(f"Custom Output: {result['custom_output'][:100]}...")
        
    except Exception as ex:
        print(f"❌ FAIL: {ex}")
