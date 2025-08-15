import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import streamlit as st
from data.role_mappings import ROLE_MAPPINGS, ROLE_DESCRIPTIONS

class RolePredictor:
    """Predict suitable roles based on extracted skills"""
    
    def __init__(self):
        """Initialize the role predictor"""
        self.role_mappings = ROLE_MAPPINGS
        self.role_descriptions = ROLE_DESCRIPTIONS
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self._prepare_role_vectors()
    
    def _prepare_role_vectors(self):
        """Prepare TF-IDF vectors for role descriptions"""
        try:
            # Combine role skills and descriptions for vectorization
            role_texts = []
            self.role_names = []
            
            for role, skills in self.role_mappings.items():
                # Combine skills with role description
                skills_text = ' '.join(skills)
                description = self.role_descriptions.get(role, '')
                combined_text = f"{skills_text} {description}"
                
                role_texts.append(combined_text)
                self.role_names.append(role)
            
            # Fit TF-IDF vectorizer
            self.role_vectors = self.vectorizer.fit_transform(role_texts)
            
        except Exception as e:
            st.warning(f"Error preparing role vectors: {e}")
            self.role_vectors = None
    
    def predict_roles(self, candidate_skills, top_k=10):
        """Predict suitable roles for a candidate based on their skills"""
        if not candidate_skills:
            return {
                'roles': [],
                'match_scores': [],
                'confidence_scores': [],
                'skill_matches': [],
                'missing_skills': []
            }
        
        # Method 1: Rule-based matching
        rule_based_results = self._rule_based_prediction(candidate_skills, top_k)
        
        # Method 2: TF-IDF similarity matching
        similarity_results = self._similarity_based_prediction(candidate_skills, top_k)
        
        # Method 3: Skill overlap scoring
        overlap_results = self._overlap_based_prediction(candidate_skills, top_k)
        
        # Combine and rank results
        combined_results = self._combine_predictions(
            rule_based_results, 
            similarity_results, 
            overlap_results, 
            top_k
        )
        
        return combined_results
    
    def _rule_based_prediction(self, candidate_skills, top_k):
        """Rule-based role prediction using exact skill matches"""
        role_scores = defaultdict(float)
        role_matches = defaultdict(list)
        
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        
        for role, required_skills in self.role_mappings.items():
            matches = 0
            matched_skills = []
            
            for required_skill in required_skills:
                if required_skill.lower() in candidate_skills_lower:
                    matches += 1
                    matched_skills.append(required_skill)
            
            if matches > 0:
                # Calculate score based on match ratio and total matches
                match_ratio = matches / len(required_skills)
                bonus = min(matches / 10, 0.3)  # Bonus for having many matches
                role_scores[role] = match_ratio + bonus
                role_matches[role] = matched_skills
        
        # Sort by score
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        
        roles = [role for role, score in sorted_roles[:top_k]]
        scores = [score for role, score in sorted_roles[:top_k]]
        matches = [role_matches[role] for role in roles]
        
        return {
            'roles': roles,
            'scores': scores,
            'matches': matches,
            'method': 'rule_based'
        }
    
    def _similarity_based_prediction(self, candidate_skills, top_k):
        """TF-IDF similarity-based role prediction"""
        if self.role_vectors is None:
            return {'roles': [], 'scores': [], 'matches': [], 'method': 'similarity'}
        
        try:
            # Create candidate profile text
            candidate_text = ' '.join(candidate_skills)
            candidate_vector = self.vectorizer.transform([candidate_text])
            
            # Calculate similarities
            similarities = cosine_similarity(candidate_vector, self.role_vectors).flatten()
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            roles = [self.role_names[i] for i in top_indices]
            scores = [similarities[i] for i in top_indices]
            
            # Find actual skill matches for each role
            matches = []
            for role in roles:
                role_skills = self.role_mappings.get(role, [])
                candidate_skills_lower = [s.lower() for s in candidate_skills]
                role_skills_lower = [s.lower() for s in role_skills]
                matched = [s for s in role_skills if s.lower() in candidate_skills_lower]
                matches.append(matched)
            
            return {
                'roles': roles,
                'scores': scores,
                'matches': matches,
                'method': 'similarity'
            }
        
        except Exception as e:
            st.warning(f"Similarity-based prediction failed: {e}")
            return {'roles': [], 'scores': [], 'matches': [], 'method': 'similarity'}
    
    def _overlap_based_prediction(self, candidate_skills, top_k):
        """Skill overlap-based role prediction with weighted scoring"""
        role_scores = defaultdict(float)
        role_matches = defaultdict(list)
        
        candidate_skills_set = set(skill.lower() for skill in candidate_skills)
        
        for role, required_skills in self.role_mappings.items():
            required_skills_set = set(skill.lower() for skill in required_skills)
            
            # Calculate Jaccard similarity
            intersection = candidate_skills_set.intersection(required_skills_set)
            union = candidate_skills_set.union(required_skills_set)
            
            if union:
                jaccard_score = len(intersection) / len(union)
                
                # Weight by the number of matching skills
                skill_weight = min(len(intersection) / 5, 1.0)  # Cap at 5 skills
                
                # Final score combines Jaccard similarity with skill weight
                final_score = jaccard_score * (0.7 + 0.3 * skill_weight)
                
                role_scores[role] = final_score
                role_matches[role] = list(intersection)
        
        # Sort by score
        sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
        
        roles = [role for role, score in sorted_roles[:top_k]]
        scores = [score for role, score in sorted_roles[:top_k]]
        matches = [role_matches[role] for role in roles]
        
        return {
            'roles': roles,
            'scores': scores,
            'matches': matches,
            'method': 'overlap'
        }
    
    def _combine_predictions(self, rule_based, similarity, overlap, top_k):
        """Combine multiple prediction methods"""
        # Collect all unique roles
        all_roles = set()
        all_roles.update(rule_based['roles'])
        all_roles.update(similarity['roles'])
        all_roles.update(overlap['roles'])
        
        # Calculate combined scores
        final_scores = {}
        final_matches = {}
        final_confidences = {}
        
        for role in all_roles:
            scores = []
            weights = []
            
            # Rule-based score
            if role in rule_based['roles']:
                idx = rule_based['roles'].index(role)
                scores.append(rule_based['scores'][idx])
                weights.append(0.4)  # High weight for exact matches
                final_matches[role] = rule_based['matches'][idx]
            
            # Similarity score
            if role in similarity['roles']:
                idx = similarity['roles'].index(role)
                scores.append(similarity['scores'][idx])
                weights.append(0.3)  # Medium weight for similarity
                if role not in final_matches:
                    final_matches[role] = similarity['matches'][idx]
            
            # Overlap score
            if role in overlap['roles']:
                idx = overlap['roles'].index(role)
                scores.append(overlap['scores'][idx])
                weights.append(0.3)  # Medium weight for overlap
                if role not in final_matches:
                    final_matches[role] = overlap['matches'][idx]
            
            # Calculate weighted average
            if scores:
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                final_scores[role] = weighted_score
                
                # Calculate confidence based on method agreement
                method_count = len(scores)
                confidence = min(0.5 + (method_count - 1) * 0.25, 1.0)
                final_confidences[role] = confidence
        
        # Sort by final score
        sorted_roles = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare final results
        roles = [role for role, score in sorted_roles[:top_k]]
        match_scores = [score * 100 for role, score in sorted_roles[:top_k]]  # Convert to percentage
        confidence_scores = [final_confidences[role] for role in roles]
        skill_matches = [final_matches.get(role, []) for role in roles]
        
        # Calculate missing skills for top roles
        missing_skills = []
        for role in roles:
            required_skills = set(self.role_mappings.get(role, []))
            candidate_skills_set = set(skill.lower() for skill in final_matches.get(role, []))
            required_skills_lower = set(skill.lower() for skill in required_skills)
            missing = required_skills - {s for s in required_skills if s.lower() in candidate_skills_set}
            missing_skills.append(list(missing))
        
        return {
            'roles': roles,
            'match_scores': match_scores,
            'confidence_scores': confidence_scores,
            'skill_matches': skill_matches,
            'missing_skills': missing_skills
        }
    
    def get_role_insights(self, role_name, candidate_skills):
        """Get detailed insights for a specific role"""
        if role_name not in self.role_mappings:
            return None
        
        required_skills = self.role_mappings[role_name]
        candidate_skills_lower = [skill.lower() for skill in candidate_skills]
        
        # Calculate skill gaps
        matched_skills = []
        missing_skills = []
        
        for skill in required_skills:
            if skill.lower() in candidate_skills_lower:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)
        
        # Calculate fit score
        fit_score = len(matched_skills) / len(required_skills) if required_skills else 0
        
        # Get role description
        description = self.role_descriptions.get(role_name, "No description available")
        
        return {
            'role': role_name,
            'description': description,
            'required_skills': required_skills,
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'fit_score': fit_score,
            'recommendation': self._generate_recommendation(fit_score, missing_skills)
        }
    
    def _generate_recommendation(self, fit_score, missing_skills):
        """Generate recommendation based on fit score and missing skills"""
        if fit_score >= 0.8:
            return "Excellent fit! This candidate has most of the required skills."
        elif fit_score >= 0.6:
            return f"Good fit with some skill gaps. Consider training in: {', '.join(missing_skills[:3])}"
        elif fit_score >= 0.4:
            return f"Moderate fit. Significant training needed in: {', '.join(missing_skills[:5])}"
        else:
            return "Poor fit. This role may not be suitable for this candidate's current skill set."
