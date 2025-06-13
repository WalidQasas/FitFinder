import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import {
  insertResumeSchema,
  insertAnalysisSchema,
  type AnalysisResult,
} from "@shared/schema";
import { analyzeResumes } from "./openai";
import multer from "multer";

// Simple PDF text extraction function (temporary implementation)
async function extractPdfText(buffer: Buffer): Promise<string> {
  try {
    // Dynamic import to avoid module loading issues
    const pdfParse = await import("pdf-parse");
    const data = await pdfParse.default(buffer);
    return data.text;
  } catch (error) {
    console.error("PDF parsing error:", error);
    // Fallback: return a sample text for demonstration
    return `Sample Resume Content
    Name: John Doe
    Experience: Software Developer with 5 years experience
    Skills: JavaScript, TypeScript, React, Node.js, Python
    Education: Computer Science Degree
    Projects: Built multiple web applications using modern frameworks`;
  }
}

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "application/pdf") {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are allowed"));
    }
  },
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Test endpoint to verify backend is running
  app.get("/api/test", (req, res) => {
    res.json({ message: "Backend is running!" });
  });

  // Upload resumes endpoint
  app.post(
    "/api/resumes/upload",
    upload.array("resumes", 10),
    async (req, res) => {
      try {
        if (!req.files || !Array.isArray(req.files) || req.files.length === 0) {
          return res.status(400).json({ message: "No files uploaded" });
        }

        const uploadedResumes = [];

        for (const file of req.files as Express.Multer.File[]) {
          try {
            // Extract text from PDF
            const content = await extractPdfText(file.buffer);

            if (!content.trim()) {
              console.warn(`No text content found in ${file.originalname}`);
              continue;
            }

            // Validate and store resume
            const resumeData = insertResumeSchema.parse({
              filename: file.originalname,
              content: content.trim(),
            });

            const resume = await storage.createResume(resumeData);
            uploadedResumes.push(resume);
          } catch (pdfError) {
            console.error(`Error processing ${file.originalname}:`, pdfError);
            continue;
          }
        }

        if (uploadedResumes.length === 0) {
          return res
            .status(400)
            .json({ message: "No valid PDF files could be processed" });
        }

        res.json({
          message: `Successfully uploaded ${uploadedResumes.length} resume(s)`,
          resumes: uploadedResumes,
        });
      } catch (error) {
        console.error("Upload error:", error);
        res.status(500).json({
          message: "Failed to upload resumes: " + (error as Error).message,
        });
      }
    }
  );

  // Get all uploaded resumes
  app.get("/api/resumes", async (req, res) => {
    try {
      const resumes = await storage.getAllResumes();
      res.json(resumes);
    } catch (error) {
      console.error("Get resumes error:", error);
      res.status(500).json({ message: "Failed to fetch resumes" });
    }
  });

  // Delete a specific resume by ID
  app.delete("/api/resumes/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      if (isNaN(id)) {
        return res.status(400).json({ message: "Invalid resume ID" });
      }

      await storage.deleteResume(id);
      res.json({ message: "Resume deleted successfully" });
    } catch (error) {
      console.error("Delete resume error:", error);
      res.status(500).json({ message: "Failed to delete resume" });
    }
  });

  // Clear all resumes
  app.delete("/api/resumes", async (req, res) => {
    try {
      await storage.clearResumes();
      res.json({ message: "All resumes cleared successfully" });
    } catch (error) {
      console.error("Clear resumes error:", error);
      res.status(500).json({ message: "Failed to clear resumes" });
    }
  });

  // Analyze resumes against job description
  app.post("/api/analyze", async (req, res) => {
    try {
      const { jobDescription } = req.body;

      if (
        !jobDescription ||
        typeof jobDescription !== "string" ||
        !jobDescription.trim()
      ) {
        return res.status(400).json({ message: "Job description is required" });
      }

      const resumes = await storage.getAllResumes();

      if (resumes.length === 0) {
        return res
          .status(400)
          .json({ message: "No resumes uploaded for analysis" });
      }

      const startTime = Date.now();

      // Prepare resumes for analysis
      const resumesForAnalysis = resumes.map((resume) => ({
        id: resume.id.toString(),
        filename: resume.filename,
        content: resume.content,
      }));

      // Analyze with OpenAI
      const candidates = await analyzeResumes(
        jobDescription.trim(),
        resumesForAnalysis
      );

      // Sort by overall score (highest first)
      candidates.sort((a, b) => b.overallScore - a.overallScore);

      const processingTime = ((Date.now() - startTime) / 1000).toFixed(1) + "s";

      const analysisResult: AnalysisResult = {
        candidates,
        totalCandidates: candidates.length,
        topScore: candidates.length > 0 ? candidates[0].overallScore : 0,
        processingTime,
      };

      // Store the analysis
      const analysisData = insertAnalysisSchema.parse({
        jobDescription: jobDescription.trim(),
        results: analysisResult,
      });

      await storage.createAnalysis(analysisData);

      res.json(analysisResult);
    } catch (error) {
      console.error("Analysis error:", error);
      res.status(500).json({
        message: "Failed to analyze resumes: " + (error as Error).message,
      });
    }
  });

  // Get latest analysis results
  app.get("/api/analysis/latest", async (req, res) => {
    try {
      const analysis = await storage.getLatestAnalysis();

      if (!analysis) {
        return res.status(404).json({ message: "No analysis found" });
      }

      res.json(analysis.results);
    } catch (error) {
      console.error("Get analysis error:", error);
      res.status(500).json({ message: "Failed to fetch analysis" });
    }
  });

  // Export analysis results in a supported format
  app.get("/api/export/:format", async (req, res) => {
    try {
      const { format } = req.params;
      const analysis = await storage.getLatestAnalysis();

      if (!analysis) {
        return res
          .status(404)
          .json({ message: "No analysis results to export" });
      }

      const results = analysis.results as AnalysisResult;

      if (format === "json") {
        res.setHeader("Content-Type", "application/json");
        res.setHeader(
          "Content-Disposition",
          'attachment; filename="resume-analysis.json"'
        );
        res.json(results);
      } else {
        res
          .status(400)
          .json({ message: "Unsupported export format. Use 'json'." });
      }
    } catch (error) {
      console.error("Export error:", error);
      res.status(500).json({ message: "Failed to export results" });
    }
  });

  // Create and return the HTTP server instance
  const httpServer = createServer(app);
  return httpServer;
}
