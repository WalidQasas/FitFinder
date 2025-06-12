import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CandidateCard } from "./candidate-card";
import { Download, RefreshCw, Users, Star, Clock, FileText, Share } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import type { AnalysisResult, Resume } from "@shared/schema";

interface AnalysisResultsProps {
  results?: AnalysisResult;
  isLoading: boolean;
  resumes: Resume[];
}

export function AnalysisResults({ results, isLoading, resumes }: AnalysisResultsProps) {
  const { toast } = useToast();

  const handleExport = async (format: string) => {
    try {
      const response = await fetch(`/api/export/${format}`, {
        credentials: 'include'
      });
      
      if (!response.ok) {
        throw new Error('Export failed');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `resume-analysis.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast({
        title: "Export Successful",
        description: `Results exported as ${format.toUpperCase()} file.`,
      });
    } catch (error) {
      toast({
        title: "Export Failed",
        description: "Failed to export results. Please try again.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="space-y-6">
      {/* Analysis Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Analysis Results</CardTitle>
            <div className="flex items-center space-x-3">
              <Button variant="ghost" size="sm" onClick={() => handleExport('json')}>
                <Download className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="sm">
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* Processing Indicator */}
          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
                <p className="text-gray-600">Analyzing resumes with AI...</p>
                <div className="w-64 bg-gray-200 rounded-full h-2 mt-4 mx-auto">
                  <div className="bg-primary h-2 rounded-full animate-pulse" style={{ width: "45%" }}></div>
                </div>
              </div>
            </div>
          )}

          {/* Results Summary */}
          {results && !isLoading && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="flex items-center">
                  <Users className="text-blue-500 text-xl mr-3 h-6 w-6" />
                  <div>
                    <p className="text-2xl font-bold text-blue-600">
                      {results.totalCandidates}
                    </p>
                    <p className="text-sm text-blue-700">Total Candidates</p>
                  </div>
                </div>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <div className="flex items-center">
                  <Star className="text-green-500 text-xl mr-3 h-6 w-6" />
                  <div>
                    <p className="text-2xl font-bold text-green-600">
                      {results.topScore}%
                    </p>
                    <p className="text-sm text-green-700">Highest Match</p>
                  </div>
                </div>
              </div>
              <div className="bg-amber-50 p-4 rounded-lg">
                <div className="flex items-center">
                  <Clock className="text-amber-500 text-xl mr-3 h-6 w-6" />
                  <div>
                    <p className="text-2xl font-bold text-amber-600">
                      {results.processingTime}
                    </p>
                    <p className="text-sm text-amber-700">Processing Time</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Empty State */}
          {!results && !isLoading && (
            <div className="text-center py-8">
              <FileText className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <p className="text-gray-600 mb-2">No analysis results yet</p>
              <p className="text-sm text-gray-500">
                Upload resumes and add a job description to get started
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Ranked Candidates List */}
      {results && !isLoading && (
        <div className="space-y-4">
          {results.candidates.map((candidate, index) => (
            <CandidateCard
              key={candidate.id}
              candidate={candidate}
              rank={index + 1}
            />
          ))}
        </div>
      )}

      {/* Export Section */}
      {results && !isLoading && (
        <Card>
          <CardHeader>
            <CardTitle>Export Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-3">
              <Button
                onClick={() => handleExport('json')}
                className="bg-green-600 hover:bg-green-700"
              >
                <Download className="mr-2 h-4 w-4" />
                Export to JSON
              </Button>
              <Button
                variant="outline"
                onClick={() => toast({
                  title: "Coming Soon",
                  description: "PDF and Excel export will be available soon.",
                })}
              >
                <FileText className="mr-2 h-4 w-4" />
                Export to PDF
              </Button>
              <Button
                variant="outline"
                onClick={() => toast({
                  title: "Coming Soon",
                  description: "Share functionality will be available soon.",
                })}
              >
                <Share className="mr-2 h-4 w-4" />
                Share Results
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
