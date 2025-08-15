"""
Machine Learning models for enhanced resume analysis
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class MLModels:
    """Machine Learning models for resume analysis enhancement"""
    
    def __init__(self):
        """Initialize ML models"""
        self.skill_classifier = None
        self.experience_predictor = None
        self.role_classifier = None
        self.salary_predictor = None
        self.vectorizers = {}
        self.label_encoders = {}
        self.is_trained = False
    
    def create_synthetic_training_data(self):
        """Create synthetic training data for model training"""
        # This method creates synthetic data for demonstration
        # In a real application, you would use actual resume data
        
        # Sample resume texts with varying skill levels
        resume_samples = [
            "Python developer with 5 years experience in Django Flask machine learning pandas numpy scikit-learn",
            "Senior Java developer Spring Boot microservices AWS Docker Kubernetes 8 years experience",
            "Frontend developer React Angular JavaScript TypeScript HTML CSS responsive design 3 years",
            "Data scientist with Python R SQL machine learning deep learning TensorFlow PyTorch 4 years",
            "Full stack developer MERN stack MongoDB Express React Node.js REST API 6 years experience",
            "DevOps engineer AWS Azure Docker Kubernetes Jenkins Terraform CI/CD automation 7 years",
            "Mobile developer iOS Swift Android Kotlin React Native Flutter 5 years experience",
            "Backend developer Python Django PostgreSQL Redis Celery microservices 6 years experience",
            "Machine learning engineer TensorFlow PyTorch scikit-learn MLflow Kubernetes 4 years experience",
            "Cloud architect AWS solutions architect EC2 S3 Lambda CloudFormation 10 years experience",
            "Junior Python developer Flask SQLAlchemy basic machine learning 1 year experience",
            "Entry level frontend developer HTML CSS JavaScript React beginner 6 months experience",
            "Senior data scientist advanced statistics deep learning computer vision NLP 8 years",
            "Lead software engineer team management architecture design Java Spring 12 years experience",
            "QA engineer automation testing Selenium Cypress API testing 4 years experience"
        ]
        
        # Corresponding labels
        experience_levels = [3, 4, 2, 3, 3, 4, 3, 3, 3, 5, 1, 1, 4, 5, 3]  # 1-5 scale
        roles = [
            'Backend Developer', 'Backend Developer', 'Frontend Developer', 'Data Scientist',
            'Full Stack Developer', 'DevOps Engineer', 'Mobile Developer', 'Backend Developer',
            'Machine Learning Engineer', 'Cloud Architect', 'Backend Developer', 'Frontend Developer',
            'Data Scientist', 'Software Engineer', 'QA Engineer'
        ]
        skill_levels = [4, 5, 3, 4, 4, 5, 4, 4, 4, 5, 2, 1, 5, 5, 3]  # 1-5 scale
        
        return {
            'resume_texts': resume_samples,
            'experience_levels': experience_levels,
            'roles': roles,
            'skill_levels': skill_levels
        }
    
    def train_models(self):
        """Train all ML models"""
        try:
            # Get training data
            training_data = self.create_synthetic_training_data()
            
            # Prepare features
            X = training_data['resume_texts']
            
            # Train experience level predictor
            self._train_experience_predictor(X, training_data['experience_levels'])
            
            # Train role classifier
            self._train_role_classifier(X, training_data['roles'])
            
            # Train skill level predictor
            self._train_skill_classifier(X, training_data['skill_levels'])
            
            self.is_trained = True
            print("All models trained successfully!")
            
        except Exception as e:
            print(f"Error training models: {e}")
            self.is_trained = False
    
    def _train_experience_predictor(self, X, y):
        """Train experience level prediction model"""
        # Create pipeline
        self.experience_predictor = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train model
        self.experience_predictor.fit(X, y)
        
        # Store vectorizer for later use
        self.vectorizers['experience'] = self.experience_predictor.named_steps['tfidf']
    
    def _train_role_classifier(self, X, y):
        """Train role classification model"""
        # Encode labels
        self.label_encoders['role'] = LabelEncoder()
        y_encoded = self.label_encoders['role'].fit_transform(y)
        
        # Create pipeline
        self.role_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train model
        self.role_classifier.fit(X, y_encoded)
        
        # Store vectorizer
        self.vectorizers['role'] = self.role_classifier.named_steps['tfidf']
    
    def _train_skill_classifier(self, X, y):
        """Train skill level classification model"""
        # Create pipeline
        self.skill_classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Train model
        self.skill_classifier.fit(X, y)
        
        # Store vectorizer
        self.vectorizers['skill'] = self.skill_classifier.named_steps['tfidf']
    
    def predict_experience_level(self, resume_text):
        """Predict experience level from resume text"""
        if not self.is_trained or self.experience_predictor is None:
            return 2  # Default to mid-level
        
        try:
            prediction = self.experience_predictor.predict([resume_text])[0]
            confidence = max(self.experience_predictor.predict_proba([resume_text])[0])
            
            return {
                'level': int(prediction),
                'confidence': float(confidence),
                'description': self._get_experience_description(prediction)
            }
        except Exception as e:
            print(f"Error predicting experience level: {e}")
            return {'level': 2, 'confidence': 0.5, 'description': 'Mid-level'}
    
    def predict_role_probabilities(self, resume_text):
        """Predict role probabilities from resume text"""
        if not self.is_trained or self.role_classifier is None:
            return {}
        
        try:
            probabilities = self.role_classifier.predict_proba([resume_text])[0]
            classes = self.label_encoders['role'].classes_
            
            # Create probability dictionary
            role_probs = {}
            for i, prob in enumerate(probabilities):
                role_probs[classes[i]] = float(prob)
            
            # Sort by probability
            sorted_roles = sorted(role_probs.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'predictions': sorted_roles[:5],  # Top 5 predictions
                'top_role': sorted_roles[0][0],
                'top_confidence': sorted_roles[0][1]
            }
        except Exception as e:
            print(f"Error predicting roles: {e}")
            return {}
    
    def predict_skill_level(self, resume_text):
        """Predict overall skill level from resume text"""
        if not self.is_trained or self.skill_classifier is None:
            return 3  # Default to mid-level
        
        try:
            prediction = self.skill_classifier.predict([resume_text])[0]
            confidence = max(self.skill_classifier.predict_proba([resume_text])[0])
            
            return {
                'level': int(prediction),
                'confidence': float(confidence),
                'description': self._get_skill_description(prediction)
            }
        except Exception as e:
            print(f"Error predicting skill level: {e}")
            return {'level': 3, 'confidence': 0.5, 'description': 'Intermediate'}
    
    def _get_experience_description(self, level):
        """Get experience level description"""
        descriptions = {
            1: 'Entry Level (0-2 years)',
            2: 'Junior Level (2-4 years)',
            3: 'Mid Level (4-6 years)',
            4: 'Senior Level (6-10 years)',
            5: 'Expert Level (10+ years)'
        }
        return descriptions.get(level, 'Unknown')
    
    def _get_skill_description(self, level):
        """Get skill level description"""
        descriptions = {
            1: 'Beginner',
            2: 'Basic',
            3: 'Intermediate',
            4: 'Advanced',
            5: 'Expert'
        }
        return descriptions.get(level, 'Unknown')
    
    def enhance_candidate_analysis(self, resume_text, extracted_skills):
        """Enhance candidate analysis using ML models"""
        if not self.is_trained:
            # Try to train models if not already trained
            self.train_models()
        
        enhancements = {}
        
        # Predict experience level
        exp_prediction = self.predict_experience_level(resume_text)
        enhancements['ml_experience'] = exp_prediction
        
        # Predict roles
        role_prediction = self.predict_role_probabilities(resume_text)
        enhancements['ml_roles'] = role_prediction
        
        # Predict skill level
        skill_prediction = self.predict_skill_level(resume_text)
        enhancements['ml_skill_level'] = skill_prediction
        
        # Calculate enhanced scores
        enhancements['enhanced_scores'] = self._calculate_enhanced_scores(
            extracted_skills, exp_prediction, skill_prediction
        )
        
        return enhancements
    
    def _calculate_enhanced_scores(self, extracted_skills, exp_prediction, skill_prediction):
        """Calculate enhanced scores using ML predictions"""
        scores = {}
        
        # Experience-based scoring
        exp_level = exp_prediction.get('level', 2)
        exp_confidence = exp_prediction.get('confidence', 0.5)
        scores['experience_ml_score'] = (exp_level / 5) * 100 * exp_confidence
        
        # Skill-based scoring
        skill_level = skill_prediction.get('level', 3)
        skill_confidence = skill_prediction.get('confidence', 0.5)
        scores['skill_ml_score'] = (skill_level / 5) * 100 * skill_confidence
        
        # Combined ML score
        scores['combined_ml_score'] = (scores['experience_ml_score'] + scores['skill_ml_score']) / 2
        
        return scores
    
    def save_models(self, directory='models'):
        """Save trained models to disk"""
        if not self.is_trained:
            print("No trained models to save")
            return
        
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save models
            if self.experience_predictor:
                joblib.dump(self.experience_predictor, f'{directory}/experience_predictor.pkl')
            
            if self.role_classifier:
                joblib.dump(self.role_classifier, f'{directory}/role_classifier.pkl')
            
            if self.skill_classifier:
                joblib.dump(self.skill_classifier, f'{directory}/skill_classifier.pkl')
            
            # Save label encoders
            if self.label_encoders:
                joblib.dump(self.label_encoders, f'{directory}/label_encoders.pkl')
            
            print(f"Models saved to {directory}")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self, directory='models'):
        """Load trained models from disk"""
        try:
            # Load models
            if os.path.exists(f'{directory}/experience_predictor.pkl'):
                self.experience_predictor = joblib.load(f'{directory}/experience_predictor.pkl')
            
            if os.path.exists(f'{directory}/role_classifier.pkl'):
                self.role_classifier = joblib.load(f'{directory}/role_classifier.pkl')
            
            if os.path.exists(f'{directory}/skill_classifier.pkl'):
                self.skill_classifier = joblib.load(f'{directory}/skill_classifier.pkl')
            
            # Load label encoders
            if os.path.exists(f'{directory}/label_encoders.pkl'):
                self.label_encoders = joblib.load(f'{directory}/label_encoders.pkl')
            
            self.is_trained = True
            print(f"Models loaded from {directory}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_trained = False

# Global instance for use across the application
ml_models = MLModels()
