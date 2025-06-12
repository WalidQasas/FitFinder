import { pgTable, text, serial, integer, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const resumes = pgTable("resumes", {
  id: serial("id").primaryKey(),
  filename: text("filename").notNull(),
  content: text("content").notNull(),
  uploadedAt: text("uploaded_at").notNull(),
});

export const analyses = pgTable("analyses", {
  id: serial("id").primaryKey(),
  jobDescription: text("job_description").notNull(),
  results: jsonb("results").notNull(), // Store array of candidate results
  createdAt: text("created_at").notNull(),
});

export const insertResumeSchema = createInsertSchema(resumes).omit({
  id: true,
  uploadedAt: true,
});

export const insertAnalysisSchema = createInsertSchema(analyses).omit({
  id: true,
  createdAt: true,
});

export type InsertResume = z.infer<typeof insertResumeSchema>;
export type Resume = typeof resumes.$inferSelect;
export type InsertAnalysis = z.infer<typeof insertAnalysisSchema>;
export type Analysis = typeof analyses.$inferSelect;

// Types for the analysis results
export type CandidateScore = {
  id: string;
  filename: string;
  name: string;
  title: string;
  overallScore: number;
  skillsScore: number;
  experienceScore: number;
  educationScore: number;
  highlights: string[];
  strengths: string;
  areasForGrowth: string;
  content: string;
};

export type AnalysisResult = {
  candidates: CandidateScore[];
  totalCandidates: number;
  topScore: number;
  processingTime: string;
};
