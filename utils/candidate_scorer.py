import numpy as np
import re
from collections import defaultdict
from data.skills_database import SKILLS_DATABASE

class CandidateScorer:
    """Calculate comprehensive candidate scores based on multiple factors"""
    
    def __init__(self):
        """Initialize the candidate scorer"""
        self.skills_db = SKILLS_DATABASE
        self._prepare_skill_weights()
    
    def _prepare_skill_weights(self):
        """Prepare skill category weights for scoring"""
        self.category_weights = {
            'Programming Languages': 1.0,
            'Web Technologies': 0.9,
            'Databases': 0.8,
            'Cloud Platforms': 0.9,
            'Machine Learning': 1.1,
            'Data Science': 1.0,
            'DevOps': 0.8,
            'Mobile Development': 0.9,
            'Frameworks': 0.9,
            'Tools': 0.7,
            'Soft Skills': 0.6,
            'Certifications': 0.8
        }
        
        # Skill rarity weights (higher weight for rare/valuable skills)
        self.skill_rarity_weights = {
            'Machine Learning': 1.3,
            'Artificial Intelligence': 1.3,
            'Deep Learning': 1.4,
            'Data Science': 1.2,
            'Cloud Architecture': 1.2,
            'Kubernetes': 1.1,
            'Docker': 1.0,
            'React': 1.0,
            'Python': 1.0,
            'JavaScript': 0.9,
            'HTML': 0.7,
            'CSS': 0.7
        }
    
    def calculate_score(self, skills_data, role_predictions):
        """Calculate comprehensive candidate score"""
        # Calculate individual component scores
        skills_score = self._calculate_skills_score(skills_data)
        experience_score = self._calculate_experience_score(skills_data)
        role_fit_score = self._calculate_role_fit_score(role_predictions)
        diversity_score = self._calculate_diversity_score(skills_data)
        
        # Calculate weighted overall score
        overall_score = (
            skills_score * 0.35 +
            experience_score * 0.25 +
            role_fit_score * 0.25 +
            diversity_score * 0.15
        )
        
        return {
            'overall_score': round(overall_score, 1),
            'skills_score': round(skills_score, 1),
            'experience_score': round(experience_score, 1),
            'role_fit_score': round(role_fit_score, 1),
            'diversity_score': round(diversity_score, 1),
            'score_breakdown': self._generate_score_breakdown(skills_data, role_predictions),
            'recommendations': self._generate_recommendations(skills_data, role_predictions)
        }
    
    def _calculate_skills_score(self, skills_data):
        """Calculate score based on skills quality and quantity"""
        if not skills_data['skills']:
            return 0
        
        skills = skills_data['skills']
        categories = skills_data['categories']
        confidences = skills_data['confidence_scores']
        
        total_score = 0
        skill_count = len(skills)
        
        for skill, category, confidence in zip(skills, categories, confidences):
            # Base score from confidence
            skill_score = confidence * 100
            
            # Apply category weight
            category_weight = self.category_weights.get(category, 0.8)
            skill_score *= category_weight
            
            # Apply rarity weight
            rarity_weight = self.skill_rarity_weights.get(skill, 1.0)
            skill_score *= rarity_weight
            
            total_score += skill_score
        
        # Calculate average and apply skill count bonus
        if skill_count > 0:
            average_score = total_score / skill_count
            
            # Bonus for having many skills (up to 20% bonus)
            skill_count_bonus = min(skill_count / 20 * 20, 20)
            
            final_score = min(average_score + skill_count_bonus, 100)
        else:
            final_score = 0
        
        return final_score
    
    def _calculate_experience_score(self, skills_data):
        """Calculate score based on inferred experience level"""
        if not skills_data['skills']:
            return 0
        
        skills = skills_data['skills']
        categories = skills_data['categories']
        
        # Experience indicators
        experience_indicators = {
            'Senior': ['senior', 'lead', 'architect', 'principal', 'staff'],
            'Advanced': ['advanced', 'expert', 'professional', 'enterprise'],
            'Framework/Library Mastery': ['react', 'angular', 'vue', 'django', 'flask', 'spring'],
            'Architecture': ['microservices', 'distributed', 'scalable', 'architecture'],
            'Leadership': ['team lead', 'technical lead', 'project management', 'mentoring']
        }
        
        experience_score = 40  # Base score
        
        # Count technical skills (higher count suggests more experience)
        technical_categories = ['Programming Languages', 'Web Technologies', 'Databases', 'Cloud Platforms']
        technical_skills = [s for s, c in zip(skills, categories) if c in technical_categories]
        
        # Experience bonus based on skill count
        if len(technical_skills) >= 10:
            experience_score += 30
        elif len(technical_skills) >= 7:
            experience_score += 20
        elif len(technical_skills) >= 5:
            experience_score += 15
        elif len(technical_skills) >= 3:
            experience_score += 10
        
        # Experience bonus based on skill sophistication
        skill_text = ' '.join(skills).lower()
        
        for indicator_type, indicators in experience_indicators.items():
            for indicator in indicators:
                if indicator in skill_text:
                    if indicator_type == 'Senior':
                        experience_score += 15
                    elif indicator_type == 'Advanced':
                        experience_score += 10
                    elif indicator_type == 'Architecture':
                        experience_score += 12
                    elif indicator_type == 'Leadership':
                        experience_score += 8
                    else:
                        experience_score += 5
                    break
        
        # Category diversity bonus (more categories = more experience)
        unique_categories = len(set(categories))
        if unique_categories >= 5:
            experience_score += 15
        elif unique_categories >= 4:
            experience_score += 10
        elif unique_categories >= 3:
            experience_score += 5
        
        return min(experience_score, 100)
    
    def _calculate_role_fit_score(self, role_predictions):
        """Calculate score based on role prediction quality"""
        if not role_predictions['roles']:
            return 0
        
        # Use the top role prediction
        top_match_score = role_predictions['match_scores'][0] if role_predictions['match_scores'] else 0
        top_confidence = role_predictions['confidence_scores'][0] if role_predictions['confidence_scores'] else 0
        
        # Combine match score and confidence
        role_fit_score = (top_match_score + top_confidence * 100) / 2
        
        # Bonus for having multiple good role matches
        good_matches = len([score for score in role_predictions['match_scores'] if score >= 60])
        if good_matches >= 3:
            role_fit_score += 10
        elif good_matches >= 2:
            role_fit_score += 5
        
        return min(role_fit_score, 100)
    
    def _calculate_diversity_score(self, skills_data):
        """Calculate score based on skill diversity"""
        if not skills_data['skills']:
            return 0
        
        categories = skills_data['categories']
        skills = skills_data['skills']
        
        # Category diversity
        unique_categories = len(set(categories))
        category_score = min(unique_categories * 15, 60)  # Max 60 from categories
        
        # Technology stack completeness
        stack_completeness = self._assess_stack_completeness(skills, categories)
        stack_score = stack_completeness * 40  # Max 40 from stack completeness
        
        diversity_score = category_score + stack_score
        
        return min(diversity_score, 100)
    
    def _assess_stack_completeness(self, skills, categories):
        """Assess how complete the candidate's technology stack is"""
        skills_lower = [skill.lower() for skill in skills]
        
        # Define complete stack indicators
        stack_indicators = {
            'Frontend': ['react', 'angular', 'vue', 'javascript', 'html', 'css'],
            'Backend': ['python', 'java', 'node.js', 'c#', 'php', 'ruby'],
            'Database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis'],
            'Cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'DevOps': ['git', 'ci/cd', 'jenkins', 'docker', 'terraform']
        }
        
        stack_coverage = 0
        for stack_type, indicators in stack_indicators.items():
            has_skill_in_stack = any(indicator in skill for skill in skills_lower for indicator in indicators)
            if has_skill_in_stack:
                stack_coverage += 1
        
        return stack_coverage / len(stack_indicators)
    
    def _generate_score_breakdown(self, skills_data, role_predictions):
        """Generate detailed score breakdown"""
        breakdown = {
            'skills_analysis': {
                'total_skills': len(skills_data['skills']),
                'skill_categories': len(set(skills_data['categories'])) if skills_data['categories'] else 0,
                'avg_confidence': sum(skills_data['confidence_scores']) / len(skills_data['confidence_scores']) if skills_data['confidence_scores'] else 0,
                'top_categories': self._get_top_categories(skills_data['categories'])
            },
            'role_fit_analysis': {
                'top_role': role_predictions['roles'][0] if role_predictions['roles'] else 'None',
                'top_role_score': role_predictions['match_scores'][0] if role_predictions['match_scores'] else 0,
                'role_confidence': role_predictions['confidence_scores'][0] if role_predictions['confidence_scores'] else 0,
                'good_role_matches': len([s for s in role_predictions['match_scores'] if s >= 60])
            }
        }
        
        return breakdown
    
    def _get_top_categories(self, categories):
        """Get top skill categories by count"""
        if not categories:
            return []
        
        category_counts = defaultdict(int)
        for category in categories:
            category_counts[category] += 1
        
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_categories[:5]
    
    def _generate_recommendations(self, skills_data, role_predictions):
        """Generate recommendations for candidate improvement"""
        recommendations = []
        
        # Skill-based recommendations
        if len(skills_data['skills']) < 5:
            recommendations.append("Consider expanding skill set - aim for at least 5-7 core technical skills")
        
        if not skills_data['categories'] or len(set(skills_data['categories'])) < 3:
            recommendations.append("Diversify skills across multiple categories (Frontend, Backend, Database, etc.)")
        
        # Role-based recommendations
        if role_predictions['roles']:
            top_role = role_predictions['roles'][0]
            if role_predictions['missing_skills'] and role_predictions['missing_skills'][0]:
                missing_skills = role_predictions['missing_skills'][0][:3]  # Top 3 missing skills
                recommendations.append(f"For {top_role} role, consider learning: {', '.join(missing_skills)}")
        
        # Experience recommendations
        skills_text = ' '.join(skills_data['skills']).lower()
        if 'senior' not in skills_text and 'lead' not in skills_text:
            recommendations.append("Gain leadership experience or advanced technical expertise to increase seniority level")
        
        # Stack completeness recommendations
        skills_lower = [skill.lower() for skill in skills_data['skills']]
        
        has_frontend = any(skill in skills_lower for skill in ['react', 'angular', 'vue', 'javascript'])
        has_backend = any(skill in skills_lower for skill in ['python', 'java', 'node.js', 'c#'])
        has_database = any(skill in skills_lower for skill in ['sql', 'mysql', 'postgresql', 'mongodb'])
        
        if not has_frontend:
            recommendations.append("Consider learning a frontend framework (React, Angular, or Vue)")
        if not has_backend:
            recommendations.append("Consider learning a backend technology (Python, Java, Node.js)")
        if not has_database:
            recommendations.append("Consider learning database technologies (SQL, NoSQL)")
        
        return recommendations[:5]  # Return top 5 recommendations
