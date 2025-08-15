import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st
from data.skills_database import SKILLS_DATABASE
import string
from collections import defaultdict

class SkillExtractor:
    """Extract skills from resume text using NER and rule-based matching"""
    
    def __init__(self):
        """Initialize the skill extractor"""
        self.nlp = self._load_spacy_model()
        self.skills_db = SKILLS_DATABASE
        self.stop_words = set(stopwords.words('english'))
        self._prepare_skill_patterns()
    
    def _load_spacy_model(self):
        """Load spaCy model with error handling"""
        try:
            # Try to load the English model
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except OSError:
            try:
                # Fallback to basic English model
                nlp = spacy.load("en_core_web_md")
                return nlp
            except OSError:
                # If no model is available, use blank model
                st.warning("spaCy English model not found. Using basic text processing.")
                nlp = spacy.blank("en")
                return nlp
    
    def _prepare_skill_patterns(self):
        """Prepare skill patterns for matching"""
        self.skill_patterns = {}
        
        for category, skills in self.skills_db.items():
            patterns = []
            for skill in skills:
                # Create regex patterns for each skill
                skill_lower = skill.lower()
                # Handle multi-word skills
                if ' ' in skill_lower:
                    pattern = r'\b' + re.escape(skill_lower) + r'\b'
                else:
                    pattern = r'\b' + re.escape(skill_lower) + r'\b'
                patterns.append((pattern, skill))
            self.skill_patterns[category] = patterns
    
    def extract_skills(self, text):
        """Extract skills from text using multiple approaches"""
        if not text:
            return {
                'skills': [],
                'categories': [],
                'confidence_scores': [],
                'context': []
            }
        
        # Combine NER and rule-based extraction
        ner_skills = self._extract_ner_skills(text)
        rule_based_skills = self._extract_rule_based_skills(text)
        
        # Merge and deduplicate results
        merged_skills = self._merge_skill_results(ner_skills, rule_based_skills, text)
        
        return merged_skills
    
    def _extract_ner_skills(self, text):
        """Extract skills using Named Entity Recognition"""
        skills_found = []
        
        try:
            doc = self.nlp(text)
            
            # Extract entities that might be skills
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'LANGUAGE', 'PERSON']:
                    # Check if entity matches known skills
                    entity_text = ent.text.strip()
                    skill_info = self._classify_skill(entity_text)
                    if skill_info:
                        skills_found.append({
                            'skill': skill_info['skill'],
                            'category': skill_info['category'],
                            'confidence': 0.8,  # High confidence for NER matches
                            'context': self._get_context(text, entity_text),
                            'method': 'NER'
                        })
            
            # Extract noun phrases that might be skills
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                if len(chunk_text.split()) <= 3:  # Focus on shorter phrases
                    skill_info = self._classify_skill(chunk_text)
                    if skill_info:
                        skills_found.append({
                            'skill': skill_info['skill'],
                            'category': skill_info['category'],
                            'confidence': 0.6,  # Medium confidence for noun chunks
                            'context': self._get_context(text, chunk_text),
                            'method': 'NER_CHUNK'
                        })
        
        except Exception as e:
            st.warning(f"NER extraction failed: {e}")
        
        return skills_found
    
    def _extract_rule_based_skills(self, text):
        """Extract skills using rule-based pattern matching"""
        skills_found = []
        text_lower = text.lower()
        
        for category, patterns in self.skill_patterns.items():
            for pattern, skill in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group()
                    skills_found.append({
                        'skill': skill,
                        'category': category,
                        'confidence': 0.9,  # High confidence for exact matches
                        'context': self._get_context(text, matched_text),
                        'method': 'RULE_BASED'
                    })
        
        return skills_found
    
    def _classify_skill(self, text):
        """Classify a text snippet as a skill"""
        text_lower = text.lower().strip()
        
        # Skip if too short or contains only punctuation
        if len(text_lower) < 2 or text_lower in self.stop_words:
            return None
        
        # Direct lookup in skills database
        for category, skills in self.skills_db.items():
            for skill in skills:
                if text_lower == skill.lower():
                    return {'skill': skill, 'category': category}
                # Check for partial matches for multi-word skills
                if ' ' in skill.lower() and text_lower in skill.lower():
                    return {'skill': skill, 'category': category}
        
        # Fuzzy matching for common variations
        text_normalized = self._normalize_skill_name(text_lower)
        for category, skills in self.skills_db.items():
            for skill in skills:
                skill_normalized = self._normalize_skill_name(skill.lower())
                if text_normalized == skill_normalized:
                    return {'skill': skill, 'category': category}
        
        return None
    
    def _normalize_skill_name(self, skill):
        """Normalize skill names for better matching"""
        # Remove common prefixes/suffixes
        skill = re.sub(r'\b(framework|library|language|technology|tool)\b', '', skill)
        # Remove version numbers
        skill = re.sub(r'\d+\.?\d*', '', skill)
        # Remove special characters
        skill = re.sub(r'[^\w\s]', ' ', skill)
        # Remove extra spaces
        skill = ' '.join(skill.split())
        return skill.strip()
    
    def _get_context(self, text, skill):
        """Get context around a skill mention"""
        try:
            # Find the position of the skill in text
            skill_pos = text.lower().find(skill.lower())
            if skill_pos == -1:
                return ""
            
            # Extract context (50 characters before and after)
            start = max(0, skill_pos - 50)
            end = min(len(text), skill_pos + len(skill) + 50)
            context = text[start:end]
            
            return context.strip()
        except:
            return ""
    
    def _merge_skill_results(self, ner_skills, rule_based_skills, text):
        """Merge and deduplicate skill extraction results"""
        all_skills = ner_skills + rule_based_skills
        
        # Group by skill name (case-insensitive)
        skill_groups = defaultdict(list)
        for skill_item in all_skills:
            key = skill_item['skill'].lower()
            skill_groups[key].append(skill_item)
        
        # Deduplicate and calculate final confidence scores
        final_skills = []
        final_categories = []
        final_confidences = []
        final_contexts = []
        
        for skill_key, skill_items in skill_groups.items():
            # Use the highest confidence version
            best_skill = max(skill_items, key=lambda x: x['confidence'])
            
            # Calculate final confidence based on multiple detections
            confidence_boost = min(0.2, (len(skill_items) - 1) * 0.1)
            final_confidence = min(1.0, best_skill['confidence'] + confidence_boost)
            
            final_skills.append(best_skill['skill'])
            final_categories.append(best_skill['category'])
            final_confidences.append(final_confidence)
            final_contexts.append(best_skill['context'])
        
        # Sort by confidence
        sorted_data = sorted(
            zip(final_skills, final_categories, final_confidences, final_contexts),
            key=lambda x: x[2],
            reverse=True
        )
        
        if sorted_data:
            skills, categories, confidences, contexts = zip(*sorted_data)
            return {
                'skills': list(skills),
                'categories': list(categories),
                'confidence_scores': list(confidences),
                'context': list(contexts)
            }
        else:
            return {
                'skills': [],
                'categories': [],
                'confidence_scores': [],
                'context': []
            }
    
    def get_skill_statistics(self, skills_data):
        """Get statistics about extracted skills"""
        if not skills_data['skills']:
            return {}
        
        categories = skills_data['categories']
        confidences = skills_data['confidence_scores']
        
        # Category distribution
        category_counts = defaultdict(int)
        category_confidences = defaultdict(list)
        
        for category, confidence in zip(categories, confidences):
            category_counts[category] += 1
            category_confidences[category].append(confidence)
        
        # Calculate average confidence per category
        category_avg_confidence = {}
        for category, conf_list in category_confidences.items():
            category_avg_confidence[category] = sum(conf_list) / len(conf_list)
        
        return {
            'total_skills': len(skills_data['skills']),
            'category_distribution': dict(category_counts),
            'category_avg_confidence': category_avg_confidence,
            'overall_avg_confidence': sum(confidences) / len(confidences)
        }
