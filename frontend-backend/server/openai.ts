import OpenAI from "openai";
import type { CandidateScore } from "@shared/schema";

// the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
const openai = new OpenAI({ 
  apiKey: process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY_ENV_VAR || "default_key"
});

export async function analyzeResumes(
  jobDescription: string, 
  resumes: Array<{ id: string; filename: string; content: string }>
): Promise<CandidateScore[]> {
  const prompt = `
You are an expert HR analyst. Analyze the following resumes against the job description and provide detailed scoring.

Job Description:
${jobDescription}

Resumes to analyze:
${resumes.map((resume, index) => `
Resume ${index + 1} (${resume.filename}):
${resume.content}
---
`).join('\n')}

For each resume, provide a detailed analysis in JSON format with the following structure:
{
  "candidates": [
    {
      "id": "resume_id",
      "filename": "filename",
      "name": "extracted_candidate_name",
      "title": "extracted_job_title_or_position",
      "overallScore": 85,
      "skillsScore": 90,
      "experienceScore": 80,
      "educationScore": 85,
      "highlights": ["React Expert", "5+ Years Node.js", "AWS Certified"],
      "strengths": "Detailed strengths paragraph",
      "areasForGrowth": "Areas for improvement paragraph"
    }
  ]
}

Scoring criteria:
- Skills (40%): Technical skills match, programming languages, frameworks, tools
- Experience (35%): Years of experience, relevant projects, leadership experience
- Education (25%): Degree relevance, certifications, continuous learning

Scores should be 0-100 integers. Extract the candidate's name and current/most recent job title from their resume.
Provide 3-5 concise highlights as tags. Write detailed paragraphs for strengths and areas for growth.
Rank candidates by overall score (highest first).
`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You are an expert HR analyst. Analyze resumes against job descriptions and provide detailed scoring in JSON format."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      response_format: { type: "json_object" },
      temperature: 0.3,
    });

    const result = JSON.parse(response.choices[0].message.content || "{}");
    
    // Add the resume content to each candidate for reference
    const candidatesWithContent = result.candidates.map((candidate: any) => {
      const resume = resumes.find(r => r.id === candidate.id || r.filename === candidate.filename);
      return {
        ...candidate,
        content: resume?.content || "",
        id: resume?.id || candidate.id
      };
    });

    return candidatesWithContent;
  } catch (error) {
    console.error("OpenAI analysis error:", error);
    throw new Error("Failed to analyze resumes with AI: " + (error as Error).message);
  }
}
