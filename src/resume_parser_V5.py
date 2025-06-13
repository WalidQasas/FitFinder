import argparse
import os
import re
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class UniversalResumeRanker:
    def __init__(self):
        """Initialize the Universal Resume Ranking System"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True,
            min_df=2
        )
        
        self.universal_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'scala', 'kotlin', 'swift'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'bootstrap'],
            'data_analysis': ['excel', 'sql', 'tableau', 'power bi', 'sas', 'spss', 'r', 'pandas', 'numpy', 'statistics'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis', 'elasticsearch'],
            'cloud_tech': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'project_management': ['pmp', 'agile', 'scrum', 'waterfall', 'jira', 'project planning', 'risk management', 'stakeholder management']
        }
        
        self.certifications = {
            'it': ['cissp', 'ccna', 'ccnp', 'aws certified', 'azure certified', 'comptia'],
            'project_management': ['pmp', 'prince2', 'capm', 'agile certified']
        }

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.\,\(\)]', '', text)
        return text.lower().strip()

    def extract_dynamic_skills(self, text, reference_skills=None):
        """Extract skills dynamically using NLP techniques"""
        if not text:
            return {}
        
        text_lower = text.lower()
        found_skills = {}
        
        skill_categories_to_check = {k: v for k, v in self.universal_skills.items() 
                                     if reference_skills and (k in reference_skills or any(skill in text_lower for skill in v))} or self.universal_skills
        
        for category, skills in skill_categories_to_check.items():
            found_skills[category] = [skill for skill in skills if skill in text_lower]
        
        return found_skills

    def extract_certifications(self, text):
        """Extract professional certifications"""
        text_lower = text.lower()
        found_certs = []
        for category, certs in self.certifications.items():
            for cert in certs:
                if cert in text_lower:
                    found_certs.append(cert)
        return found_certs

    def extract_experience_years(self, text):
        """Extract years of experience from text"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*years?\s*in',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?'
        ]
        years = [int(match) for pattern in patterns for match in re.findall(pattern, text.lower()) if match.isdigit()]
        return max(years) if years else 0

    def extract_education(self, text):
        """Extract education information"""
        education_keywords = {
            'masters': ['masters', 'master', 'mba', 'ms', 'ma', 'msc'],
            'bachelors': ['bachelors', 'bachelor', 'bs', 'ba', 'bsc']
        }
        text_lower = text.lower()
        found_education = []
        for level, keywords in education_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_education.append(level)
                    break
        education_scores = {'masters': 4, 'bachelors': 3}
        return max(education_scores.get(edu, 1) for edu in found_education) if found_education else 1

    def parse_resume(self, resume_text, filename, job_skills=None):
        """Parse resume and extract structured information"""
        cleaned_text = self.clean_text(resume_text)
        return {
            'filename': filename,
            'text': cleaned_text,
            'skills': self.extract_dynamic_skills(resume_text, job_skills),
            'certifications': self.extract_certifications(resume_text),
            'experience_years': self.extract_experience_years(resume_text),
            'education_score': self.extract_education(resume_text)
        }

    def calculate_advanced_skill_match(self, job_skills, resume_skills):
        """Advanced skill matching"""
        if not job_skills or not resume_skills:
            return 0.0
        
        total_score = 0
        total_weight = 0
        for job_category, job_category_skills in job_skills.items():
            if not job_category_skills:
                continue
            category_weight = len(job_category_skills)
            resume_category_skills = resume_skills.get(job_category, [])
            if job_category_skills and resume_category_skills:
                job_set = set(job_category_skills)
                resume_set = set(resume_category_skills)
                intersection = len(job_set.intersection(resume_set))
                union = len(job_set.union(resume_set))
                category_score = intersection / union if union > 0 else 0
                total_score += category_score * category_weight
                total_weight += category_weight
        
        return total_score / total_weight if total_weight > 0 else 0

    def calculate_text_similarity(self, job_description, resume_texts):
        """Calculate TF-IDF similarity between job description and resumes"""
        all_texts = [job_description] + resume_texts
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            job_vector = tfidf_matrix[0:1]
            resume_vectors = tfidf_matrix[1:]
            return cosine_similarity(job_vector, resume_vectors).flatten()
        except Exception as e:
            print(f"Error calculating text similarity: {e}")
            return np.zeros(len(resume_texts))

    def rank_resumes(self, job_description, resumes_data):
        """Rank resumes based on a single job description"""
        job_skills = self.extract_dynamic_skills(job_description)
        job_exp_years = self.extract_experience_years(job_description)
        job_education = self.extract_education(job_description)
        job_certs = self.extract_certifications(job_description)
        cleaned_job_desc = self.clean_text(job_description)
        
        processed_resumes = [self.parse_resume(resume.get('text', ''), resume['filename'], job_skills) for resume in resumes_data]
        for resume, original in zip(processed_resumes, resumes_data):
            resume['sector'] = original.get('sector', 'Unknown')
        
        resume_texts = [resume['text'] for resume in processed_resumes]
        text_similarities = self.calculate_text_similarity(cleaned_job_desc, resume_texts)
        
        results = []
        for i, resume in enumerate(processed_resumes):
            skill_score = self.calculate_advanced_skill_match(job_skills, resume['skills'])
            text_similarity = text_similarities[i]
            exp_ratio = resume['experience_years'] / job_exp_years if job_exp_years > 0 else min(resume['experience_years'] / 5, 1.0)
            exp_score = exp_ratio if exp_ratio <= 1 else 1.0 - min((exp_ratio - 1) * 0.1, 0.3)
            edu_score = min(resume['education_score'] / max(job_education, 3), 1.0)
            cert_score = min(len(set(job_certs).intersection(set(resume['certifications']))) / len(job_certs), 1.0) if job_certs else 0
            
            weights = self._calculate_adaptive_weights(job_skills, job_exp_years, job_certs)
            combined_score = (
                weights['text'] * text_similarity +
                weights['skills'] * skill_score +
                weights['experience'] * exp_score +
                weights['education'] * edu_score +
                weights['certifications'] * cert_score
            )
            
            results.append({
                'filename': resume['filename'],
                'sector': resume['sector'],
                'combined_score': combined_score,
                'text_similarity': text_similarity,
                'skill_score': skill_score,
                'experience_score': exp_score,
                'education_score': edu_score,
                'certification_score': cert_score,
                'experience_years': resume['experience_years'],
                'education_level': resume['education_score'],
                'skills_found': resume['skills'],
                'certifications': resume['certifications']
            })
        
        return sorted(results, key=lambda x: x['combined_score'], reverse=True)

    def _calculate_adaptive_weights(self, job_skills, job_exp_years, job_certs):
        """Calculate adaptive weights based on job requirements"""
        weights = {'text': 0.3, 'skills': 0.35, 'experience': 0.2, 'education': 0.1, 'certifications': 0.05}
        total_skills = sum(len(skills) for skills in job_skills.values())
        if total_skills > 10:
            weights['skills'] += 0.1
            weights['text'] -= 0.05
            weights['experience'] -= 0.05
        if job_exp_years > 5:
            weights['experience'] += 0.1
            weights['education'] -= 0.05
            weights['text'] -= 0.05
        if job_certs:
            weights['certifications'] += 0.1
            weights['text'] -= 0.05
            weights['skills'] -= 0.05
        return weights

    def process_resume_folder(self, data_path):
        """Process all resumes in the data folder"""
        resumes_data = []
        for sector_folder in Path(data_path).iterdir():
            if sector_folder.is_dir():
                print(f"Processing {sector_folder.name}...")
                for pdf_file in sector_folder.glob("*.pdf"):
                    try:
                        text = self.extract_text_from_pdf(pdf_file)
                        if text and len(text.strip()) > 50:
                            resumes_data.append({
                                'filename': pdf_file.name,
                                'sector': sector_folder.name,
                                'text': text
                            })
                        else:
                            print(f"  Warning: Insufficient text extracted from {pdf_file.name}")
                    except Exception as e:
                        print(f"  Error processing {pdf_file.name}: {e}")
        return resumes_data

    def display_results(self, ranked_results, top_n=10):
        """Display ranking results for a single job description"""
        print(f"\n{'='*80}")
        print(f"TOP {min(top_n, len(ranked_results))} RANKED RESUMES")
        print(f"{'='*80}")
        for i, result in enumerate(ranked_results[:top_n], 1):
            print(f"\n{i}. {result['filename']} (Sector: {result['sector']})")
            print(f"   Combined Score: {result['combined_score']:.3f}")
            print(f"   Text Similarity: {result['text_similarity']:.3f}")
            print(f"   Skill Match: {result['skill_score']:.3f}")
            print(f"   Experience: {result['experience_years']} years")
            skills_summary = [f"{cat}: {', '.join(skills[:3])}" for cat, skills in result['skills_found'].items() if skills]
            if skills_summary:
                print(f"   Key Skills: {' | '.join(skills_summary[:3])}")
            if result['certifications']:
                print(f"   Certifications: {', '.join(result['certifications'])}")

    def export_detailed_results(self, ranked_results, filename='detailed_resume_ranking.csv'):
        """Export detailed results to CSV"""
        export_data = []
        for result in ranked_results:
            all_skills = [f"{category}:{skill}" for category, skills in result['skills_found'].items() for skill in skills]
            export_data.append({
                'Filename': result['filename'],
                'Sector': result['sector'],
                'Overall_Score': result['combined_score'],
                'Text_Similarity': result['text_similarity'],
                'Skill_Score': result['skill_score'],
                'Experience_Score': result['experience_score'],
                'Education_Score': result['education_score'],
                'Certification_Score': result['certification_score'],
                'Experience_Years': result['experience_years'],
                'Education_Level': result['education_level'],
                'All_Skills': ' | '.join(all_skills),
                'Certifications': ' | '.join(result['certifications']) if result['certifications'] else ''
            })
        pd.DataFrame(export_data).to_csv(filename, index=False)
        print(f"\nDetailed results saved to '{filename}'")

def main():
    parser = argparse.ArgumentParser(description="Resume Ranking System")
    parser.add_argument("job_file", help="Path to the job description text file")
    parser.add_argument("--data_path", default="../data/data/data", help="Path to the resumes folder")
    args = parser.parse_args()
    
    try:
        with open(args.job_file, 'r', encoding='utf-8') as f:
            job_description = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Job description file '{args.job_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading job description file: {e}")
        return
    
    ranker = UniversalResumeRanker()
    resumes_data = ranker.process_resume_folder(args.data_path)
    if not resumes_data:
        print("No resumes processed. Check the data path and resume files.")
        return
    
    ranked_results = ranker.rank_resumes(job_description, resumes_data)
    ranker.display_results(ranked_results)
    ranker.export_detailed_results(ranked_results)

if __name__ == "__main__":
    main()