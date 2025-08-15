# Overview

This is an AI-powered resume analysis application built with Streamlit that processes uploaded resumes and provides comprehensive candidate evaluation. The system extracts skills using Natural Language Processing (NLP), predicts suitable job roles, and calculates scoring metrics to help with recruitment decisions. It supports multiple file formats (PDF, DOCX, TXT) and can process resumes individually or in batches.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Web Interface**: Single-page application with sidebar navigation for file uploads and parameter configuration
- **Interactive Visualizations**: Plotly-based charts and graphs for displaying analysis results, skill distributions, and comparative metrics
- **Session State Management**: Maintains analysis results and batch processing data across user interactions
- **Multi-file Upload Support**: Handles individual resume analysis and batch processing with ZIP file support

## Backend Architecture
- **Modular Component Design**: Separate utility classes for distinct functionality (parsing, extraction, prediction, scoring)
- **Pipeline Architecture**: Sequential processing flow from file upload → text extraction → skill analysis → role prediction → scoring
- **Machine Learning Integration**: Scikit-learn based models for enhanced predictions and classification tasks
- **Error Handling**: Comprehensive exception handling with graceful fallbacks for missing dependencies

## Core Processing Components

### Resume Parser (`utils/resume_parser.py`)
- Multi-format text extraction (PDF via PyPDF2/pdfplumber, DOCX via python-docx, plain text)
- NLTK integration for text preprocessing and tokenization
- Fallback mechanisms for different PDF parsing libraries

### Skill Extractor (`utils/skill_extractor.py`)
- spaCy NLP model for Named Entity Recognition
- Rule-based pattern matching against comprehensive skills database
- Categorized skill classification with confidence scoring
- Graceful degradation when advanced NLP models are unavailable

### Role Predictor (`utils/role_predictor.py`)
- TF-IDF vectorization for role-skill matching
- Cosine similarity calculations for role recommendations
- Confidence scoring based on skill overlap and market demand
- Support for multiple role predictions with ranking

### Candidate Scorer (`utils/candidate_scorer.py`)
- Multi-factor scoring algorithm incorporating skill rarity, category weights, and role fit
- Weighted scoring system with configurable parameters
- Experience level estimation and salary prediction capabilities

## Data Management

### Skills Database (`data/skills_database.py`)
- Hierarchical skill categorization (Programming Languages, Web Technologies, Databases, etc.)
- Comprehensive technology stack coverage including modern frameworks and tools
- Extensible structure for adding new skills and categories

### Role Mappings (`data/role_mappings.py`)
- Predefined role-to-skill mappings for common tech positions
- Role descriptions and requirements for context-aware matching
- Configurable role definitions for different industries or requirements

## Machine Learning Framework (`models/ml_models.py`)
- Scikit-learn pipeline for model training and prediction
- Multiple algorithm support (Random Forest, Gradient Boosting, Logistic Regression)
- Synthetic data generation for training when real data is unavailable
- Model persistence and loading capabilities

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis for handling structured results
- **NumPy**: Numerical computing for scoring algorithms and ML operations

## Natural Language Processing
- **spaCy**: Advanced NLP library for text processing and Named Entity Recognition
- **NLTK**: Natural Language Toolkit for tokenization, stopwords, and text preprocessing

## Machine Learning
- **Scikit-learn**: Machine learning library for classification, regression, and model evaluation
- **Joblib**: Model serialization and parallel processing

## Document Processing
- **PyPDF2**: PDF text extraction (primary method)
- **pdfplumber**: Alternative PDF processing for complex layouts
- **python-docx**: Microsoft Word document processing

## Data Visualization
- **Plotly**: Interactive plotting library for charts and graphs
- **Plotly Express**: High-level interface for quick visualizations

## File Handling
- **zipfile**: Batch processing of multiple resume files
- **io**: In-memory file operations and stream handling

## Text Processing
- **re**: Regular expressions for pattern matching and text cleaning
- **string**: String manipulation utilities

The system is designed to be modular and extensible, with clear separation of concerns between parsing, analysis, and presentation layers. The architecture supports both real-time individual analysis and batch processing workflows, making it suitable for various recruitment scenarios.