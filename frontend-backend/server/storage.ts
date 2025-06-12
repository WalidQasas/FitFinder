import { resumes, analyses, type Resume, type InsertResume, type Analysis, type InsertAnalysis, type AnalysisResult } from "@shared/schema";

export interface IStorage {
  // Resume operations
  createResume(resume: InsertResume): Promise<Resume>;
  getResume(id: number): Promise<Resume | undefined>;
  getAllResumes(): Promise<Resume[]>;
  deleteResume(id: number): Promise<void>;
  clearResumes(): Promise<void>;

  // Analysis operations
  createAnalysis(analysis: InsertAnalysis): Promise<Analysis>;
  getAnalysis(id: number): Promise<Analysis | undefined>;
  getLatestAnalysis(): Promise<Analysis | undefined>;
}

export class MemStorage implements IStorage {
  private resumes: Map<number, Resume>;
  private analyses: Map<number, Analysis>;
  private currentResumeId: number;
  private currentAnalysisId: number;

  constructor() {
    this.resumes = new Map();
    this.analyses = new Map();
    this.currentResumeId = 1;
    this.currentAnalysisId = 1;
  }

  // Resume operations
  async createResume(insertResume: InsertResume): Promise<Resume> {
    const id = this.currentResumeId++;
    const resume: Resume = {
      ...insertResume,
      id,
      uploadedAt: new Date().toISOString(),
    };
    this.resumes.set(id, resume);
    return resume;
  }

  async getResume(id: number): Promise<Resume | undefined> {
    return this.resumes.get(id);
  }

  async getAllResumes(): Promise<Resume[]> {
    return Array.from(this.resumes.values());
  }

  async deleteResume(id: number): Promise<void> {
    this.resumes.delete(id);
  }

  async clearResumes(): Promise<void> {
    this.resumes.clear();
  }

  // Analysis operations
  async createAnalysis(insertAnalysis: InsertAnalysis): Promise<Analysis> {
    const id = this.currentAnalysisId++;
    const analysis: Analysis = {
      ...insertAnalysis,
      id,
      createdAt: new Date().toISOString(),
    };
    this.analyses.set(id, analysis);
    return analysis;
  }

  async getAnalysis(id: number): Promise<Analysis | undefined> {
    return this.analyses.get(id);
  }

  async getLatestAnalysis(): Promise<Analysis | undefined> {
    const analyses = Array.from(this.analyses.values());
    return analyses.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())[0];
  }
}

export const storage = new MemStorage();
