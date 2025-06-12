import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation, useQuery } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { CloudUpload, FileText, X } from "lucide-react";
import type { Resume } from "@shared/schema";

export function FileUpload() {
  const { toast } = useToast();

  // Fetch uploaded resumes
  const { data: resumes = [] } = useQuery<Resume[]>({
    queryKey: ["/api/resumes"],
  });

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async (files: File[]) => {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('resumes', file);
      });

      const response = await fetch('/api/resumes/upload', {
        method: 'POST',
        body: formData,
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Upload failed');
      }

      return response.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["/api/resumes"] });
      toast({
        title: "Upload Successful",
        description: data.message,
      });
    },
    onError: (error) => {
      toast({
        title: "Upload Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  // Delete resume mutation
  const deleteMutation = useMutation({
    mutationFn: async (id: number) => {
      await apiRequest("DELETE", `/api/resumes/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/resumes"] });
      toast({
        title: "Resume Deleted",
        description: "Resume has been removed successfully.",
      });
    },
    onError: (error) => {
      toast({
        title: "Delete Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const pdfFiles = acceptedFiles.filter(file => file.type === 'application/pdf');
    
    if (pdfFiles.length !== acceptedFiles.length) {
      toast({
        title: "Invalid Files",
        description: "Only PDF files are allowed.",
        variant: "destructive",
      });
    }

    if (pdfFiles.length > 0) {
      uploadMutation.mutate(pdfFiles);
    }
  }, [uploadMutation, toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: true,
  });

  return (
    <div className="space-y-4">
      {/* File Upload Zone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
          isDragActive
            ? "border-primary bg-primary/5"
            : "border-gray-300 hover:border-primary/40"
        }`}
      >
        <input {...getInputProps()} />
        <CloudUpload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <p className="text-lg font-medium text-gray-700 mb-2">
          {isDragActive ? "Drop the files here..." : "Drag & drop PDF files here"}
        </p>
        <p className="text-sm text-gray-500 mb-4">or click to browse</p>
        <Button type="button" variant="outline">
          <FileText className="mr-2 h-4 w-4" />
          Browse Files
        </Button>
      </div>

      {/* Upload Progress */}
      {uploadMutation.isPending && (
        <div className="text-center py-4">
          <div className="inline-flex items-center">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary mr-2"></div>
            <span className="text-sm text-gray-600">Uploading and processing files...</span>
          </div>
        </div>
      )}

      {/* Uploaded Files List */}
      {resumes.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-3">
            Uploaded Files ({resumes.length})
          </h3>
          <div className="space-y-2">
            {resumes.map((resume) => (
              <div
                key={resume.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-md"
              >
                <div className="flex items-center">
                  <FileText className="h-5 w-5 text-red-500 mr-3" />
                  <span className="text-sm font-medium text-gray-700">
                    {resume.filename}
                  </span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => deleteMutation.mutate(resume.id)}
                  disabled={deleteMutation.isPending}
                  className="text-gray-400 hover:text-red-500"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
