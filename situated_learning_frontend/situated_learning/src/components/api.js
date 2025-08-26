import axios from "axios";

// api.js
const BASE_URL = "http://localhost:8090";
const FEEDBACK_API = "http://localhost:8001"; 


// Helper function for API calls
async function apiCall(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    console.error(`API call failed to ${endpoint}:`, error);
    throw error;
  }
}

// API endpoints
export const api = {
  // Get all courses
  getCourses: () => apiCall('/courses/all'),
  
  // Get assignments by course title
  getAssignments: (courseTitle) => apiCall(`/assignments/by_course_title/${encodeURIComponent(courseTitle)}`),
  
  // Generate assignment from course title
  generateAssignment: (payload) => apiCall('/generate_from_course_title', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  
  // Generate rubric for assignment
  generateRubric: (assignmentId) => apiCall(`/assignments/${assignmentId}/generate_rubric`, {
    method: 'POST',
  }),
  
  // Evaluate submission
  evaluateSubmission: (payload) => apiCall('/api/evaluate_assignment', {
    method: 'POST',
        headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  }),
  
  // Health check
  healthCheck: () => apiCall('/api/health'),
  
  // LLM status
  llmStatus: () => apiCall('/llm_status'),
  
  // DB status
  dbStatus: () => apiCall('/db_status'),

  submitFeedback: async (payload) => {
    const res = await axios.post(`${FEEDBACK_API}/feedback`, payload);
    return res.data;
  },
  getFeedback: async (type) => {
    const res = await axios.get(`${FEEDBACK_API}/feedback`, {
      params: type ? { feedback_type: type } : {}
    });
    return res.data;
  }
};


