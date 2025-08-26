import React, { useState, useEffect, useCallback } from 'react';
import { ChevronRight, FileText, Upload, CheckCircle, User, BookOpen, Settings, Search, Edit3, Save, X, Eye, Download, Send } from 'lucide-react';
import { api } from './api'; // Import the API service
import "../styles/situated.css";
import { Star} from "lucide-react";

const SituatedLearningApp = () => {
  const [currentScreen, setCurrentScreen] = useState('dashboard');
  const [generatedAssignment, setGeneratedAssignment] = useState('');
  const [generatedRubric, setGeneratedRubric] = useState('');
  const [editingAssignment, setEditingAssignment] = useState(false);
  const [editingRubric, setEditingRubric] = useState(false);
  const [editingEvaluation, setEditingEvaluation] = useState(false);
  const [courses, setCourses] = useState([]);
  const [assignments, setAssignments] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [assignmentId, setAssignmentId] = useState(null);
  const [rubricData, setRubricData] = useState(null);
  const [evaluation, setEvaluation] = useState('');
  
  // EvaluateSubmissions specific states
  const [selectedAssignment, setSelectedAssignment] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [submissionText, setSubmissionText] = useState('');
  
  // CreateAssignment specific states
  const [courseOptions, setCourseOptions] = useState([]);

  const [rating, setRating] = useState(0);
  
  const [suggestion, setSuggestion] = useState("");
  const [assignmentData, setAssignmentData] = useState({
    courseName: "",
    courseCode: "",
    instructorName: "",
    domain: "",
    topic: "",
    customInstructions: ""
  });

  // inside SituatedLearningApp component
const [showFeedback, setShowFeedback] = useState(false);
const [feedbackType, setFeedbackType] = useState("");
const [feedbackContent, setFeedbackContent] = useState("");

// 🔹 helper to open feedback
const openFeedback = (type, content) => {
  setFeedbackType(type);
  setFeedbackContent(content);
  setShowFeedback(true);
};

  // Add scroll position preservation
  const preserveScrollPosition = useCallback(() => {
    const scrollY = window.scrollY;
    requestAnimationFrame(() => {
      window.scrollTo(0, scrollY);
    });
  }, []);

function ProgressBar({ percentage }) {
  return (
    <div className="progress-bar">
      <div
        className="progress-fill"
        style={{ width: `${percentage}%` }}
      >
        {percentage}%
      </div>
    </div>
  );
}

const FeedbackForm = ({ feedbackType, content, onClose }) => {
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [suggestion, setSuggestion] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);

  const handleSubmit = async () => {
    if (rating === 0) {
      alert("Please select a rating before submitting.");
      return;
    }

    setIsSubmitting(true);
    
    try {
      await api.submitFeedback({
        feedback_type: feedbackType,
      generated_content: JSON.stringify(content),
        rating:rating,
        suggestion: suggestion || null,
      });
      
      setSubmitSuccess(true);
      
      // Auto-close after showing success
      setTimeout(() => {
        onClose?.();
      }, 1500);
      
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      alert('Failed to submit feedback. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (submitSuccess) {
    return (
      <div className="p-6 border rounded-2xl shadow-lg bg-white max-w-md">
        <div className="text-center">
          <CheckCircle className="mx-auto mb-3 text-green-500" size={48} />
          <h3 className="text-lg font-semibold text-green-700 mb-2">
            Feedback Submitted!
          </h3>
          <p className="text-gray-600">Thank you for your feedback.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 border rounded-2xl shadow-lg bg-white max-w-md">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-lg font-semibold">
          Give Feedback on {feedbackType}
        </h3>
        <button 
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600 text-xl font-bold"
        >
          ×
        </button>
      </div>

      {/* Star Rating */}
      <div className="mb-4">
        <p className="text-sm font-medium text-gray-700 mb-2">
          Rate this {feedbackType}:
        </p>
        <div className="flex gap-1">
          {[1, 2, 3, 4, 5].map((star) => (
            <Star
              key={star}
              size={32}
              className={`cursor-pointer transition-all duration-200 ${
                star <= (hover || rating)
                  ? "text-yellow-500 fill-yellow-500 scale-110"
                  : "text-gray-300 hover:text-yellow-400"
              }`}
              fill={hover||rating?"Yellow":"none"}
              onClick={() => setRating(star)}
              onMouseEnter={() => setHover(star)}
              onMouseLeave={() => setHover(0)}
            />
          ))}
        </div>
        {(rating > 0 || hover > 0) && (
          <p className="text-sm text-gray-600 mt-1">
            {hover > 0 ? hover : rating} out of 5 stars
          </p>
        )}
      </div>

      {/* Suggestion textarea */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Suggestions (optional)
        </label>
        <textarea
          className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="Share your thoughts or suggestions for improvement..."
          value={suggestion}
          onChange={(e) => setSuggestion(e.target.value)}
          rows={4}
        />
      </div>

      {/* Submit button */}
      <div className="flex gap-2">
        <button
          onClick={handleSubmit}
          disabled={isSubmitting || rating === 0}
          className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
            isSubmitting || rating === 0
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-500 hover:bg-blue-600 text-white shadow hover:shadow-md'
          }`}
        >
          <Send size={16} />
          {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
        </button>
        
        <button
          onClick={onClose}
          className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
};


function EvaluationReport({ evaluation }) {
  if (!evaluation) return null;

  const data = typeof evaluation === "string" ? JSON.parse(evaluation) : evaluation;

  return (
    <div>
      <h2>{data.rubric_name}</h2>
      <p>
        <strong>Overall:</strong> {data.evaluation_summary.total_score}/
        {data.evaluation_summary.total_questions} (
        {data.evaluation_summary.percentage}%)
      </p>
      <ProgressBar percentage={data.evaluation_summary.percentage} />

      {Object.entries(data.category_breakdown).map(([cat, val]) => {
        const pct = (val.score / val.total) * 100;
        return (
          <div key={cat} style={{ marginTop: "12px" }}>
            <p>
              <strong>{cat}:</strong> {val.score}/{val.total} ({pct.toFixed(0)}%)
            </p>
            <ProgressBar percentage={pct.toFixed(0)} />
          </div>
        );
      })}
    </div>
  );
}

function RubricViewer({ rubricData }) {
  if (!rubricData) return <p>No rubric available.</p>;

  const { rubric_name, doc_type, rubrics } = rubricData;

  return (
    <div className="rubric-viewer">
      <h2>{rubric_name}</h2>
      <h4 style={{ color: "#666" }}>{doc_type}</h4>

      {rubrics.map((section, idx) => (
        <Category key={idx} section={section} />
      ))}
    </div>
  );
}

function Category({ section }) {
  const [open, setOpen] = React.useState(false);

  return (
    <div className="rubric-category">
      <div
        className="category-header"
        onClick={() => setOpen(!open)}
        style={{
          cursor: "pointer",
          background: "#f5f5f5",
          padding: "8px",
          borderRadius: "6px",
          marginBottom: "4px"
        }}
      >
        <strong>{section.category}</strong> {open ? "▲" : "▼"}
      </div>
      {open && (
        <ul style={{ marginLeft: "16px" }}>
          {section.questions.map((q, i) => (
            <li key={i} style={{ marginBottom: "6px" }}>
              {q}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}


  // Memoized handlers to prevent unnecessary re-renders
  const handleInputChange = useCallback((field, value) => {
    preserveScrollPosition();
    setAssignmentData(prev => ({
      ...prev,
      [field]: value
    }));
  }, [preserveScrollPosition]);

  const handleEvaluationChange = useCallback((value) => {
    preserveScrollPosition();
    setEvaluation(value);
  }, [preserveScrollPosition]); 

  // Fetch courses on component mount
  useEffect(() => {
    fetchCourses();
  }, []);

  // CreateAssignment specific useEffect
  useEffect(() => {
    if (courses.length > 0) {
      setCourseOptions(courses);
    }
  }, [courses]);

  const fetchCourses = async () => {
    try {
      setIsLoading(true);
      const data = await api.getCourses();
      setCourses(data.courses || []);
    } catch (err) {
      setError(`Failed to fetch courses: ${err.message}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchAssignments = async (courseTitle) => {
    try {
      setIsLoading(true);
      const data = await api.getAssignments(courseTitle);
      setAssignments(data || []);
    } catch (err) {
      setError(`Failed to fetch assignments: ${err.message}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateAssignment = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const payload = {
        course_title: assignmentData.courseName,
        topic: assignmentData.topic,
        user_domain: assignmentData.domain,
        extra_instructions: assignmentData.customInstructions
      };
      
      const data = await api.generateAssignment(payload);
      setGeneratedAssignment(data.generated_assignment || '');
      setAssignmentId(data.assignment_id || null);
      
          const newAssignment = {
      id: data.assignment_id || Date.now(), // Use a unique ID
      title: `${assignmentData.courseCode} - ${assignmentData.topic}`,
      courseName: assignmentData.courseName,
      courseCode: assignmentData.courseCode,
      instructorName: assignmentData.instructorName,
      topic: assignmentData.topic
    };
    
    // Check if this assignment already exists
    const assignmentExists = assignments.some(a => 
      a.id === newAssignment.id || 
      a.title === newAssignment.title
    );
    
    if (!assignmentExists) {
      setAssignments(prev => [...prev, newAssignment]);
    }

      openFeedback("assignment", data.generated_assignment || '');

    } catch (err) {
      setError(`Failed to generate assignment: ${err.message}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateRubric = async () => {
    try {
      if (!assignmentId) {
        setError('No assignment ID available. Please generate an assignment first.');
        return;
      }
      
      setIsLoading(true);
      setError(null);
      
      const data = await api.generateRubric(assignmentId);
      setGeneratedRubric(data.rubric || '');
      setRubricData(data.rubric ? JSON.parse(data.rubric) : null);
      
        openFeedback("rubric", data.rubric || '');

    } catch (err) {
      setError(`Failed to generate rubric: ${err.message}`);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveAssignment = async () => {
    // In a real implementation, you would save the edited assignment to the backend
    setEditingAssignment(false);
    console.log('Assignment saved:', generatedAssignment);
  };

  const handleSaveRubric = async () => {
    // In a real implementation, you would save the edited rubric to the backend
    setEditingRubric(false);
    console.log('Rubric saved:', generatedRubric);
  };

  // Add a function to handle course selection
  const handleCourseSelect = useCallback((courseTitle) => {
    setAssignmentData(prev => {
      const courseCodeMatch = courseTitle.match(/\(([^)]+)\)/);
      return {
        ...prev,
        courseName: courseTitle,
        courseCode: courseCodeMatch && courseCodeMatch[1] ? courseCodeMatch[1] : prev.courseCode
      };
    });
    
    // Fetch assignments for this course
    fetchAssignments(courseTitle);
  }, []);

  // EvaluateSubmissions specific functions
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setUploadedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setSubmissionText(e.target.result);
      };
      reader.readAsText(file);
    }
  };

  const handleEvaluate = async () => {
    console.log("=== EVALUATE FUNCTION STARTED ===");
    
    try {
      setIsLoading(true);
      setError(null);
      
      if (!rubricData || !generatedAssignment || !submissionText) {
        const missingItems = [];
        if (!rubricData) missingItems.push("rubricData");
        if (!generatedAssignment) missingItems.push("generatedAssignment");
        if (!submissionText) missingItems.push("submissionText");
        
        const errorMsg = `Missing required data: ${missingItems.join(", ")}`;
        setError(errorMsg);
        return;
      }
      
      const payload = {
        rubric: rubricData,
        assignment_description: generatedAssignment,
        submission: submissionText
      };
      console.log("Sending payload to API:", payload);

      const data = await api.evaluateSubmission(payload);
      console.log("=== API RESPONSE ===");
      console.log("Raw data:", data);
      
      if (data) {
        let evaluationText;
        if (typeof data === 'string') {
          evaluationText = data;
        } else if (typeof data === 'object') {
          evaluationText = JSON.stringify(data, null, 2);
        } else {
          evaluationText = String(data);
        }
        
        console.log("Setting parent evaluation state...");
        setEvaluation(evaluationText); // This now sets the parent state
        console.log("Evaluation set successfully!");
    openFeedback("evaluation", evaluationText || '');


      } else {
        setError('No evaluation data received from server');
      }

    } catch (err) {
      console.error("Evaluation error:", err);
      setError(`Failed to evaluate submission: ${err.message}`);
    } finally { 
      setIsLoading(false);
      console.log("=== EVALUATE FUNCTION COMPLETED ===");
    }
  };

  const handleSaveEvaluation = () => {
    setEditingEvaluation(false);
    console.log('Evaluation saved:', evaluation);
  };

  
  // Render functions for different screens
  const renderDashboard = () => (


    <div className="main-content">
  <div className="dashboard">
        <div className="page-container">
  <div className="max-w-6xl w-full">
    <div className="dashboard-container">
      <div className="dashboard-content">
        <div className="dashboard-header">
          <h1>Situated Learning Assignment System</h1>
          <div className="header-actions">
            <div className="search-box">
              <Search className="search-icon" />
              <input type="text" placeholder="Search assignments..." />
            </div>
            <div className="user-profile">
              <User className="profile-icon" />
              <span>Dr. Smith</span>
            </div>
          </div>
        </div>
        
        <div className="dashboard-grid">
          <div 
            onClick={() => setCurrentScreen('create')}
            className="dashboard-card create-card"
          >
            <div className="card-content">
              <div className="card-info">
                <h2>Create Assignment</h2>
                <p>Generate new situated learning assignments using AI</p>
                <div className="card-stats">
                  <span className="stat">{assignments.length} Created</span>
                  <span className="stat">{assignments.length} Active</span>
                </div>
              </div>
              <div className="card-icon create-icon">
                <FileText />
              </div>
            </div>
          </div>
          
          <div 
            onClick={() => setCurrentScreen('evaluate')}
            className="dashboard-card evaluate-card"
          >
            <div className="card-content">
              <div className="card-info">
                <h2>Evaluate Submissions</h2>
                <p>Upload and grade student submissions with AI assistance</p>
                <div className="card-stats">
                  <span className="stat">0 Submissions</span>
                  <span className="stat">0 Pending</span>
                </div>
              </div>
              <div className="card-icon evaluate-icon">
                <Upload />
              </div>
            </div>
          </div>
        </div>
        
        <div className="recent-activity">
          <h3>Recent Activity</h3>
          <div className="activity-list">
            {assignments.slice(0, 3).map((assignment, index) => (
              <div key={index} className="activity-item">
                <div className="activity-icon success">
                  <CheckCircle />
                </div>
                <div className="activity-content">
                  <span className="activity-title">{assignment.title || 'Assignment'} - Created</span>
                  <span className="activity-time">Recently</span>
                </div>
                <div className="activity-status success">Completed</div>
              </div>
            ))}
            {assignments.length === 0 && (
              <div className="activity-item">
                <div className="activity-content">
                  <span className="activity-title">No recent activity</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
    </div>
    </div>
    </div>
    </div>
  );

  const renderCreateAssignment = () => (
    <div className="page-container">
      <div className="page-content">
        <div className="page-header">
          <button 
            onClick={() => setCurrentScreen('dashboard')}
            className="back-button"
          >
            ← Back to Dashboard
          </button>
          <h1>Create Situated Learning Assignment</h1>
        </div>
        
        <div className="form-card">
          <div className="form-section">
            <h3>Course Information</h3>
            <div className="form-grid">
              <div className="flex">
              <div className="form-group">
                <label>Course Name</label>
                <select 
                  value={assignmentData.courseName}
                  onChange={(e) => handleCourseSelect(e.target.value)}
                >
                  <option value="">Select a course</option>
                  {courseOptions.map((course, index) => (
                    <option key={index} value={course}>
                      {course}
                    </option>
                  ))}
                </select>
              </div>
              
              <div className="form-group">
                <label>Course Code</label>
                <input 
                  type="text"
                  value={assignmentData.courseCode}
                  onChange={(e) => handleInputChange('courseCode', e.target.value)}
                  placeholder="e.g., CS F415"
                />
              </div>
              </div>
              
              <div className="flex">
              <div className="form-group">
                <label>Instructor Name</label>
                <input 
                  type="text"
                  value={assignmentData.instructorName}
                  onChange={(e) => handleInputChange('instructorName', e.target.value)}
                  placeholder="e.g., Dr. Smith"
                />
              </div>
              
              <div className="form-group">
                <label>Domain</label>
                <select 
                  value={assignmentData.domain}
                  onChange={(e) => handleInputChange('domain', e.target.value)}
                >
                  <option value="">Select Domain</option>
                  <option value="Technology Industry">Technology Industry</option>
                  <option value="Manufacturing">Manufacturing</option>
                  <option value="Finance">Finance</option>
                  <option value="Healthcare">Healthcare</option>
                  <option value="Education">Education</option>
                </select>
              </div>
            </div>
          </div>
          </div>
          
          <div className="form-section">
            <h3>Assignment Details</h3>
            <div className="form-group">
              <label>Topic</label>
              <input 
                type="text"
                value={assignmentData.topic}
                onChange={(e) => handleInputChange('topic', e.target.value)}
                placeholder="e.g., Artificial Intelligence, Machine Learning, etc."
                required
              />
            </div>
            
            <div className="form-group">
              <label>Custom Instructions</label>
              <textarea 
                value={assignmentData.customInstructions}
                onChange={(e) => handleInputChange('customInstructions', e.target.value)}
                rows={4}
                placeholder="Any specific requirements, constraints, or additional context for the assignment..."
              />
            </div>
          </div>
          
          <div className="form-actions">
            <button 
              onClick={handleGenerateAssignment}
              className="btn-primary"
              disabled={isLoading || !assignmentData.courseName || !assignmentData.topic}
            >
              <Settings />
              {isLoading ? 'Generating...' : 'Generate Assignment'}
            </button>
          </div>
          
          {generatedAssignment && (
            <div className="result-section">
              <div className="result-header">
                <h3>Generated Assignment</h3>
                <div className="result-actions">
                  <button 
                    onClick={() => setEditingAssignment(!editingAssignment)}
                    className={editingAssignment ? "btn-secondary active" : "btn-secondary"}
                  >
                    <Edit3 />
                    {editingAssignment ? 'Cancel Edit' : 'Edit'}
                  </button>
                  {editingAssignment && (
                    <button onClick={handleSaveAssignment} className="btn-success">
                      <Save />
                      Save Changes
                    </button>
                  )}
                </div>
              </div>
              
              <div className="result-content">
                {editingAssignment ? (
                  <textarea
                    value={generatedAssignment}
                    onChange={(e) => setGeneratedAssignment(e.target.value)}
                    className="edit-textarea"
                    rows={20}
                  />
                ) : (
                  <pre className="result-display">{generatedAssignment}</pre>
                )}
              </div>
              
              <div className="result-footer">
                <button 
                  onClick={handleGenerateRubric}
                  className="btn-success"
                  disabled={isLoading || !assignmentId}
                >
                  <CheckCircle />
                  {isLoading ? 'Generating...' : 'Generate Rubrics'}
                </button>
                
                <button className="btn-outline">
                  <Download />
                  Export Assignment
                </button>
                
                <button className="btn-primary">
                  <Save />
                  Approve & Save
                </button>
              </div>          
            </div>
          )}
          
          {generatedRubric && (
            <div className="result-section">
              <div className="result-header">
                <h3>Generated Evaluation Rubrics</h3>
                <div className="result-actions">
                  <button 
                    onClick={() => setEditingRubric(!editingRubric)}
                    className={editingRubric ? "btn-secondary active" : "btn-secondary"}
                  >
                    <Edit3 />
                    {editingRubric ? 'Cancel Edit' : 'Edit'}
                  </button>
                  {editingRubric && (
                    <button onClick={handleSaveRubric} className="btn-success">
                      <Save />
                      Save Changes
                    </button>
                  )}
                </div>
              </div>
              
<div className="result-content rubric">
  {editingRubric ? (
    <textarea
      value={generatedRubric}
      onChange={(e) => setGeneratedRubric(e.target.value)}
      className="edit-textarea"
      rows={15}
    />
  ) : (
    <RubricViewer rubricData={rubricData} />
  )}
</div>


              
              <div className="result-footer">
                <button className="btn-success">
                  <Save />
                  Save Rubrics
                </button>
                <button className="btn-outline">
                  <Download />
                  Export Rubrics
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderEvaluateSubmissions = () => (
    <div className="page-container">
      <div className="page-content">
        <div className="page-header">
          <button 
            onClick={() => setCurrentScreen('dashboard')}
            className="back-button"
          >
            ← Back to Dashboard
          </button>
          <h1>Evaluate Student Submissions</h1>
        </div>
        
      <div className="form-card">
        <div className="form-section">
          <h3>Select Assignment</h3>
          <div className="assignment-selector">
            <select 
              value={selectedAssignment}
              onChange={(e) => {
                const selectedId = e.target.value;
                setSelectedAssignment(selectedId);
                
                // Find the selected assignment and update assignmentData
                const selected = assignments.find(a => a.id === selectedId);
                if (selected) {
                  setAssignmentData(prev => ({
                    ...prev,
                    courseName: selected.courseName || prev.courseName,
                    courseCode: selected.courseCode || prev.courseCode,
                    instructorName: selected.instructorName || prev.instructorName,
                    topic: selected.topic || prev.topic
                  }));
                }
              }}
              className="assignment-select"
            >
              <option value="">Choose an assignment...</option>
              {assignments
                .filter((assignment, index, self) => 
                  // Filter out duplicates by checking if this is the first occurrence of this assignment
                  index === self.findIndex(a => 
                    a.id === assignment.id || 
                    a.title === assignment.title
                  )
                )
                .map((assignment, index) => (
                  <option key={assignment.id || index} value={assignment.id || index}>
                    {assignment.title || `Assignment ${index + 1}`}
                  </option>
                ))
              }
            </select>
          </div>
          {selectedAssignment && (
            <div className="assignment-details">
              <div className="assignment-info">
                <h4>{assignments.find(a => a.id === selectedAssignment)?.title || 'Selected Assignment'}</h4>
                <p>{assignmentData.courseName} - {assignmentData.topic}</p>
                <div className="assignment-meta">
                  <span className="meta-item">Course: {assignmentData.courseCode}</span>
                  <span className="meta-item">Instructor: {assignmentData.instructorName}</span>
                  <span className="meta-item status-ready">Rubrics Ready</span>
                </div>
              </div>
            </div>
          )}
        </div>
          
          <div className="form-section">
            <h3>Upload Student Submission</h3>
            <div className="upload-area">
              <div className="upload-content">
                <Upload className="upload-icon" />
                <p className="upload-text">Drag and drop files here, or browse</p>
                <div className="upload-actions">
                  <label className="btn-primary upload-btn">
                    <span>Browse Files</span>
                    <input 
                      type="file" 
                      className="hidden-input" 
                      onChange={handleFileUpload}
                      accept=".pdf,.doc,.docx,.txt"
                    />
                  </label>
                  <button
                    onClick={() => {
                      const mockFile = new File(
                        ["This is a mock submission for testing purposes."],
                        "mock_submission.txt",
                        { type: "text/plain" }
                      );
                      setUploadedFile(mockFile);
                      setSubmissionText("This is a mock submission for testing purposes.");
                    }}
                    className="btn-secondary"
                  >
                    Use Mock File
                  </button>
                </div>
                <p className="upload-hint">Supported: PDF, DOC, DOCX, TXT (Max 10MB)</p>
              </div>
              
              {uploadedFile && (
                <div className="upload-success">
                  <div className="success-content">
                    <CheckCircle className="success-icon" />
                    <div className="file-info">
                      <strong>{uploadedFile.name}</strong>
                      <span>{(uploadedFile.size / 1024).toFixed(2)} KB</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {uploadedFile && selectedAssignment && (
            <div className="form-actions">
              <button 
                onClick={handleEvaluate}
                className="btn-primary evaluate-btn"
                disabled={isLoading}
              >
                <Settings />
                {isLoading ? 'Evaluating...' : 'Evaluate Submission'}
              </button>
            </div>
          )}
          
          {evaluation && (
            <div className="result-section">
              <div className="result-header">
                <h3>AI Evaluation Results</h3>
                <div className="result-actions">
                  <button 
                    onClick={() => setEditingEvaluation(!editingEvaluation)}
                    className={editingEvaluation ? "btn-secondary active" : "btn-secondary"}
                  >
                    <Edit3 />
                    {editingEvaluation ? 'Cancel Edit' : 'Edit Results'}
                  </button>
                  {editingEvaluation && (
                    <button onClick={handleSaveEvaluation} className="btn-success">
                      <Save />
                      Save Changes
                    </button>
                  )}
                </div>
              </div>
              
              <div className="result-content evaluation">
                {editingEvaluation ? (
                  <textarea
                    value={evaluation}
                    onChange={(e) => handleEvaluationChange(e.target.value)}
                    className="edit-textarea"
                    rows={20}
                  />
                ) : (
                  <pre className="result-display">{(() => {
  let evalData;
  try {
    evalData = JSON.parse(evaluation);
  } catch {
    return <pre className="result-display">{evaluation}</pre>;
  }

  return (
    <div className="evaluation-report">
      <h4>{evalData.rubric_name}</h4>

      {/* Overall Score */}
      <div className="evaluation-overall">
        <p>
          Overall: {evalData.evaluation_summary.total_score}/
          {evalData.evaluation_summary.total_questions} (
          {evalData.evaluation_summary.percentage}%)
        </p>
        <ProgressBar percentage={evalData.evaluation_summary.percentage} />
      </div>

      {/* Category breakdown */}
      <div className="evaluation-categories">
        {Object.entries(evalData.category_breakdown).map(([cat, data]) => (
          <div key={cat} className="category-block">
            <p>
              <strong>{cat}</strong>: {data.score}/{data.total} (
              {data.percentage}%)
            </p>
            <ProgressBar percentage={data.percentage} />
          </div>
        ))}
      </div>
    </div>
  );
})()}
</pre>
                )}
              </div>
              
              <div className="result-footer">
                <button className="btn-primary">
                  <Save />
                  Save Evaluation
                </button>
                <button className="btn-success">
                  <Send />
                  Send to Student
                </button>
                <button className="btn-outline">
                  <Download />
                  Export Report
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderErrorDisplay = () => (
    error && (
      <div className="error-message">
        <p>{error}</p>
        <button onClick={() => setError(null)}>Dismiss</button>
      </div>
    )
  );

  const renderLoadingIndicator = () => (
    isLoading && (
      <div className="loading-indicator">
        <div className="spinner"></div>
        <p>Processing...</p>
      </div>
    )
  );

  return (
    <div className="app">
      {renderErrorDisplay()}
      {renderLoadingIndicator()}
      
      {currentScreen === 'dashboard' && renderDashboard()}
      {currentScreen === 'create' && renderCreateAssignment()}
      {currentScreen === 'evaluate' && renderEvaluateSubmissions()}
    
       {/* 🔽 Feedback modal goes here */}
    {showFeedback && (
      <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
        <FeedbackForm
          feedbackType={feedbackType}
          content={feedbackContent}
          onClose={() => setShowFeedback(false)}
        />
      </div>
    )}
    
    </div>
  );
};

export default SituatedLearningApp;