import streamlit as st
import io
import PyPDF2
import pdfplumber
from docx import Document
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class ResumeParser:
    """Parse and extract text from resume files"""
    
    def __init__(self):
        """Initialize the resume parser"""
        self.download_nltk_data()
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def extract_text(self, file):
        """Extract text from uploaded file"""
        if file.type == "application/pdf":
            return self._extract_pdf_text(file)
        elif file.type == "text/plain":
            return self._extract_txt_text(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self._extract_docx_text(file)
        else:
            raise ValueError(f"Unsupported file type: {file.type}")
    
    def _extract_pdf_text(self, file):
        """Extract text from PDF file"""
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    return self._clean_text(text)
        except Exception as e:
            st.warning(f"pdfplumber failed: {e}. Trying PyPDF2...")
        
        try:
            # Fallback to PyPDF2
            file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return self._clean_text(text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_txt_text(self, file):
        """Extract text from TXT file"""
        try:
            text = file.read().decode('utf-8')
            return self._clean_text(text)
        except UnicodeDecodeError:
            try:
                file.seek(0)
                text = file.read().decode('latin-1')
                return self._clean_text(text)
            except Exception as e:
                raise ValueError(f"Failed to decode text file: {str(e)}")
    
    def _extract_docx_text(self, file):
        """Extract text from DOCX file"""
        try:
            doc = Document(file)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return self._clean_text(text)
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {str(e)}")
    
    def _clean_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.\,\-\@\(\)\[\]\:\/]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_sections(self, text):
        """Extract common resume sections"""
        sections = {
            'contact': '',
            'summary': '',
            'experience': '',
            'education': '',
            'skills': '',
            'projects': '',
            'certifications': ''
        }
        
        # Define section keywords
        section_patterns = {
            'contact': r'(contact|phone|email|address|linkedin)',
            'summary': r'(summary|objective|profile|about)',
            'experience': r'(experience|employment|work|career|professional)',
            'education': r'(education|academic|degree|university|college)',
            'skills': r'(skills|competencies|technologies|tools|expertise)',
            'projects': r'(projects|portfolio|work samples)',
            'certifications': r'(certifications|certificates|licenses)'
        }
        
        lines = text.split('\n')
        current_section = 'summary'  # Default section
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            section_found = False
            for section, pattern in section_patterns.items():
                if re.search(pattern, line_lower) and len(line.split()) <= 5:
                    current_section = section
                    section_found = True
                    break
            
            # Add content to current section
            if not section_found and line.strip():
                sections[current_section] += line + ' '
        
        # Clean sections
        for section in sections:
            sections[section] = sections[section].strip()
        
        return sections
    
    def extract_contact_info(self, text):
        """Extract contact information from text"""
        contact_info = {
            'email': [],
            'phone': [],
            'linkedin': [],
            'github': []
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        contact_info['email'] = re.findall(email_pattern, text)
        
        # Phone pattern
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        contact_info['phone'] = re.findall(phone_pattern, text)
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        contact_info['linkedin'] = re.findall(linkedin_pattern, text, re.IGNORECASE)
        
        # GitHub pattern
        github_pattern = r'github\.com/[\w-]+'
        contact_info['github'] = re.findall(github_pattern, text, re.IGNORECASE)
        
        return contact_info
    
    def extract_years_of_experience(self, text):
        """Extract years of experience from text"""
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'(\d+)\+?\s*years?\s*experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\s*experience',
            r'(\d+)\+?\s*years?\s*in'
        ]
        
        years = []
        text_lower = text.lower()
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text_lower)
            years.extend([int(match) for match in matches])
        
        return max(years) if years else 0
