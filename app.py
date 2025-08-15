import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
from datetime import datetime
import json

from utils.resume_parser import ResumeParser
from utils.skill_extractor import SkillExtractor
from utils.role_predictor import RolePredictor
from utils.candidate_scorer import CandidateScorer

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

def initialize_components():
    """Initialize all AI components"""
    try:
        parser = ResumeParser()
        extractor = SkillExtractor()
        predictor = RolePredictor()
        scorer = CandidateScorer()
        return parser, extractor, predictor, scorer
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.stop()

def analyze_single_resume(file, parser, extractor, predictor, scorer):
    """Analyze a single resume file"""
    try:
        # Parse resume
        text_content = parser.extract_text(file)
        
        # Extract skills using NER
        skills_data = extractor.extract_skills(text_content)
        
        # Predict roles
        role_predictions = predictor.predict_roles(skills_data['skills'])
        
        # Calculate scores
        candidate_score = scorer.calculate_score(skills_data, role_predictions)
        
        return {
            'filename': file.name,
            'text_content': text_content,
            'skills_data': skills_data,
            'role_predictions': role_predictions,
            'candidate_score': candidate_score,
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            'filename': file.name,
            'error': str(e),
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def display_analysis_results(results):
    """Display analysis results with visualizations"""
    if 'error' in results:
        st.error(f"Error analyzing {results['filename']}: {results['error']}")
        return
    
    st.subheader(f"📄 Analysis Results: {results['filename']}")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Skills visualization
        st.write("### 🎯 Extracted Skills")
        skills_data = results['skills_data']
        
        if skills_data['skills']:
            # Skills by category
            skills_df = pd.DataFrame([
                {'Skill': skill, 'Category': category, 'Confidence': confidence}
                for skill, category, confidence in zip(
                    skills_data['skills'],
                    skills_data['categories'],
                    skills_data['confidence_scores']
                )
            ])
            
            # Skills distribution chart
            category_counts = skills_df['Category'].value_counts()
            fig_skills = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Skills Distribution by Category"
            )
            st.plotly_chart(fig_skills, use_container_width=True)
            
            # Skills table
            st.dataframe(skills_df, use_container_width=True)
        else:
            st.warning("No skills extracted from the resume.")
    
    with col2:
        # Role predictions
        st.write("### 🎯 Role Predictions")
        role_predictions = results['role_predictions']
        
        if role_predictions:
            roles_df = pd.DataFrame([
                {'Role': role, 'Match Score': score, 'Confidence': confidence}
                for role, score, confidence in zip(
                    role_predictions['roles'],
                    role_predictions['match_scores'],
                    role_predictions['confidence_scores']
                )
            ])
            
            # Role match visualization
            fig_roles = px.bar(
                roles_df.head(5),
                x='Match Score',
                y='Role',
                orientation='h',
                title="Top 5 Role Matches",
                color='Confidence',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_roles, use_container_width=True)
            
            st.dataframe(roles_df, use_container_width=True)
        else:
            st.warning("No role predictions available.")
    
    # Candidate scoring
    st.write("### 📊 Candidate Scoring")
    candidate_score = results['candidate_score']
    
    # Score metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Score",
            f"{candidate_score['overall_score']:.1f}/100",
            delta=f"{candidate_score['overall_score'] - 70:.1f}"
        )
    
    with col2:
        st.metric(
            "Skills Score",
            f"{candidate_score['skills_score']:.1f}/100"
        )
    
    with col3:
        st.metric(
            "Experience Score",
            f"{candidate_score['experience_score']:.1f}/100"
        )
    
    with col4:
        st.metric(
            "Role Fit Score",
            f"{candidate_score['role_fit_score']:.1f}/100"
        )
    
    # Score breakdown chart
    score_components = {
        'Skills': candidate_score['skills_score'],
        'Experience': candidate_score['experience_score'],
        'Role Fit': candidate_score['role_fit_score'],
        'Education': candidate_score.get('education_score', 0)
    }
    
    fig_scores = go.Figure(data=go.Scatterpolar(
        r=list(score_components.values()),
        theta=list(score_components.keys()),
        fill='toself',
        name='Candidate Score'
    ))
    
    fig_scores.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Score Breakdown"
    )
    
    st.plotly_chart(fig_scores, use_container_width=True)

def main():
    # Header
    st.title("🎯 AI-Powered Resume Analyzer")
    st.markdown("**Smart HR Screening with NER, Role Prediction & Candidate Scoring**")
    
    # Initialize components
    parser, extractor, predictor, scorer = initialize_components()
    
    # Sidebar
    st.sidebar.header("📋 Analysis Options")
    
    # Analysis mode selection
    analysis_mode = st.sidebar.selectbox(
        "Select Analysis Mode",
        ["Single Resume", "Batch Processing", "Analysis Dashboard"]
    )
    
    if analysis_mode == "Single Resume":
        st.header("📄 Single Resume Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=['pdf', 'txt', 'docx'],
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if uploaded_file is not None:
            if st.button("🔍 Analyze Resume", type="primary"):
                with st.spinner("Analyzing resume..."):
                    results = analyze_single_resume(uploaded_file, parser, extractor, predictor, scorer)
                    st.session_state.analysis_results.append(results)
                    display_analysis_results(results)
    
    elif analysis_mode == "Batch Processing":
        st.header("📊 Batch Resume Processing")
        
        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Upload Multiple Resumes",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload multiple resume files for batch processing"
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} files uploaded")
            
            if st.button("🚀 Process All Resumes", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    results = analyze_single_resume(file, parser, extractor, predictor, scorer)
                    batch_results.append(results)
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.session_state.batch_results = batch_results
                status_text.text("✅ Batch processing completed!")
                
                # Display batch summary
                st.subheader("📈 Batch Analysis Summary")
                
                # Create summary statistics
                successful_analyses = [r for r in batch_results if 'error' not in r]
                failed_analyses = [r for r in batch_results if 'error' in r]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Resumes", len(uploaded_files))
                with col2:
                    st.metric("Successfully Analyzed", len(successful_analyses))
                with col3:
                    st.metric("Failed Analyses", len(failed_analyses))
                
                if successful_analyses:
                    # Create batch summary dataframe
                    summary_data = []
                    for result in successful_analyses:
                        summary_data.append({
                            'Filename': result['filename'],
                            'Overall Score': result['candidate_score']['overall_score'],
                            'Skills Count': len(result['skills_data']['skills']),
                            'Top Role': result['role_predictions']['roles'][0] if result['role_predictions']['roles'] else 'N/A',
                            'Role Match Score': result['role_predictions']['match_scores'][0] if result['role_predictions']['match_scores'] else 0
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Summary visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_scores = px.histogram(
                            summary_df,
                            x='Overall Score',
                            title="Distribution of Overall Scores",
                            nbins=20
                        )
                        st.plotly_chart(fig_scores, use_container_width=True)
                    
                    with col2:
                        fig_skills = px.scatter(
                            summary_df,
                            x='Skills Count',
                            y='Overall Score',
                            title="Skills Count vs Overall Score",
                            hover_data=['Filename']
                        )
                        st.plotly_chart(fig_skills, use_container_width=True)
                    
                    # Summary table
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Export functionality
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=csv,
                        file_name=f"resume_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    elif analysis_mode == "Analysis Dashboard":
        st.header("📊 Analysis Dashboard")
        
        if st.session_state.analysis_results or st.session_state.batch_results:
            # Combine all results
            all_results = st.session_state.analysis_results + st.session_state.batch_results
            successful_results = [r for r in all_results if 'error' not in r]
            
            if successful_results:
                st.subheader("🎯 Performance Metrics")
                
                # Overall statistics
                total_candidates = len(successful_results)
                avg_score = sum(r['candidate_score']['overall_score'] for r in successful_results) / total_candidates
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Candidates", total_candidates)
                with col2:
                    st.metric("Average Score", f"{avg_score:.1f}")
                with col3:
                    high_performers = len([r for r in successful_results if r['candidate_score']['overall_score'] >= 80])
                    st.metric("High Performers (≥80)", high_performers)
                with col4:
                    top_score = max(r['candidate_score']['overall_score'] for r in successful_results)
                    st.metric("Top Score", f"{top_score:.1f}")
                
                # Detailed analytics
                st.subheader("📈 Detailed Analytics")
                
                # Score distribution
                scores = [r['candidate_score']['overall_score'] for r in successful_results]
                fig_dist = px.histogram(
                    x=scores,
                    title="Score Distribution",
                    labels={'x': 'Overall Score', 'y': 'Frequency'},
                    nbins=20
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Skills analysis
                all_skills = []
                for result in successful_results:
                    all_skills.extend(result['skills_data']['skills'])
                
                if all_skills:
                    skill_counts = pd.Series(all_skills).value_counts().head(20)
                    fig_skills = px.bar(
                        x=skill_counts.values,
                        y=skill_counts.index,
                        orientation='h',
                        title="Top 20 Most Common Skills",
                        labels={'x': 'Frequency', 'y': 'Skills'}
                    )
                    st.plotly_chart(fig_skills, use_container_width=True)
                
                # Role predictions analysis
                all_roles = []
                for result in successful_results:
                    if result['role_predictions']['roles']:
                        all_roles.append(result['role_predictions']['roles'][0])  # Top predicted role
                
                if all_roles:
                    role_counts = pd.Series(all_roles).value_counts().head(10)
                    fig_roles = px.pie(
                        values=role_counts.values,
                        names=role_counts.index,
                        title="Distribution of Predicted Roles"
                    )
                    st.plotly_chart(fig_roles, use_container_width=True)
                
                # Individual results
                st.subheader("🔍 Individual Results")
                selected_result = st.selectbox(
                    "Select a resume to view detailed analysis:",
                    options=range(len(successful_results)),
                    format_func=lambda x: successful_results[x]['filename']
                )
                
                if selected_result is not None:
                    display_analysis_results(successful_results[selected_result])
            else:
                st.info("No successful analysis results to display.")
        else:
            st.info("No analysis results available. Please analyze some resumes first.")
    
    # Footer
    st.markdown("---")
    st.markdown("**AI Resume Analyzer** - Powered by spaCy NER, scikit-learn ML, and Streamlit")

if __name__ == "__main__":
    main()
