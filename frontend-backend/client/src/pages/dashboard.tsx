import { useState } from "react";
import { FileUpload } from "@/components/file-upload";
import { AnalysisResults } from "@/components/analysis-results";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Bell, WandSparkles } from "lucide-react";
import type { Resume, AnalysisResult } from "@shared/schema";
import logoPath from "../../../attached_assets/Logo.jpg"; // Adjust the path as necessary

export default function Dashboard() {
  const [jobDescription, setJobDescription] = useState("");
  const { toast } = useToast();

  // Fetch uploaded resumes
  const { data: resumes = [], isLoading: resumesLoading } = useQuery<Resume[]>({
    queryKey: ["/api/resumes"],
  });

  // Fetch latest analysis
  const { data: analysisResults, isLoading: analysisLoading } =
    useQuery<AnalysisResult>({
      queryKey: ["/api/analysis/latest"],
      retry: false,
    });

  // Analyze resumes mutation
  const analyzeMutation = useMutation({
    mutationFn: async () => {
      if (!jobDescription.trim()) {
        throw new Error("Job description is required");
      }
      if (resumes.length === 0) {
        throw new Error("Please upload at least one resume");
      }

      const response = await apiRequest("POST", "/api/analyze", {
        jobDescription: jobDescription.trim(),
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/analysis/latest"] });
      toast({
        title: "Analysis Complete",
        description: "Resumes have been successfully analyzed and ranked.",
      });
    },
    onError: (error) => {
      toast({
        title: "Analysis Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleAnalyze = () => {
    analyzeMutation.mutate();
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card border-b border-border sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <img
                src={logoPath}
                alt="FitFinder Logo"
                className="h-10 w-auto mr-3"
              />
              <h1 className="text-xl font-bold text-foreground">FitFinder</h1>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="sm">
                <Bell className="h-5 w-5 text-muted-foreground" />
              </Button>
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                  <span className="text-white text-sm font-medium">HR</span>
                </div>
                <span className="text-sm font-medium text-foreground">
                  HR Manager
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle>Upload CVs & Job Description</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Job Description Input */}
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Job Description
                  </label>
                  <Textarea
                    placeholder="Paste the job description here..."
                    value={jobDescription}
                    onChange={(e) => setJobDescription(e.target.value)}
                    className="h-32 resize-none"
                  />
                </div>

                {/* File Upload */}
                <FileUpload />

                {/* Uploaded Files Count */}
                <div className="text-sm text-muted-foreground">
                  {resumesLoading
                    ? "Loading resumes..."
                    : `${resumes.length} resume${
                        resumes.length !== 1 ? "s" : ""
                      } uploaded`}
                </div>

                {/* Analyze Button */}
                <Button
                  onClick={handleAnalyze}
                  disabled={
                    analyzeMutation.isPending ||
                    !jobDescription.trim() ||
                    resumes.length === 0
                  }
                  className="w-full"
                >
                  <WandSparkles className="mr-2 h-4 w-4" />
                  {analyzeMutation.isPending
                    ? "Analyzing..."
                    : "Analyze Resumes"}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            <AnalysisResults
              results={analysisResults}
              isLoading={analyzeMutation.isPending || analysisLoading}
              resumes={resumes}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
