import React, { useState } from "react";

export default function AssignmentGenerator() {
  const [courseId, setCourseId] = useState("");
  const [topic, setTopic] = useState("");
  const [userDomain, setUserDomain] = useState("");
  const [extraInstructions, setExtraInstructions] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [generatedAssignment, setGeneratedAssignment] = useState(null);
  const [assignmentId, setAssignmentId] = useState(null); // NEW: assignment id
  const [examples, setExamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [rubric, setRubric] = useState(null); // NEW: rubric state

  const handleStartSession = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8090/start_assignment_session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ course_id: courseId }),
      });
      const data = await response.json();
      setSessionId(data.session_id);
      setExamples(data.examples);
    } catch (error) {
      console.error("Failed to start session:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateAssignment = async () => {
    if (!sessionId) {
      alert("Please start a session first.");
      return;
    }

    setLoading(true);
    try {
      const url = `http://localhost:8090/generate_from_topic`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          topic,
          user_domain: userDomain,
          extra_instructions: extraInstructions,
        }),
      });
      const data = await response.json();
      setGeneratedAssignment(data.generated_assignment);
      setAssignmentId(data.assignment_id || null); // store assignment id
    } catch (error) {
      console.error("Failed to generate assignment:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateRubric = async () => {
    if (!assignmentId) {
      alert("Assignment ID not found.");
      return;
    }

    setLoading(true);
    try {
      const url = `http://localhost:8090/assignments/${assignmentId}/generate_rubric`;
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await response.json();
      setRubric(data.rubric || "No rubric generated.");
    } catch (error) {
      console.error("Failed to generate rubric:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Situated Learning</h1>

      <div className="space-y-4">
        <input
          className="w-full p-2 border rounded"
          type="text"
          placeholder="Course ID"
          value={courseId}
          onChange={(e) => setCourseId(e.target.value)}
        />

        <button
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          onClick={handleStartSession}
          disabled={loading}
        >
          Enter
        </button>

        {sessionId && (
          <div className="space-y-4 mt-6">
            <div className="space-y-6">
              <div className="grid grid-cols-3 gap-4 items-center">
                <label className="font-medium text-right">Topic</label>
                <input
                  className="col-span-2 p-2 border rounded w-full"
                  type="text"
                  placeholder="Topic"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-3 gap-4 items-center">
                <label className="font-medium text-right">Domain</label>
                <input
                  className="col-span-2 p-2 border rounded w-full"
                  type="text"
                  placeholder="Your Domain (e.g., Healthcare, Finance)"
                  value={userDomain}
                  onChange={(e) => setUserDomain(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-3 gap-4 items-start">
                <label className="font-medium text-right mt-2">
                  Extra Instructions
                </label>
                <textarea
                  className="col-span-2 p-2 border rounded w-full"
                  placeholder="Extra Instructions (optional)"
                  rows="4"
                  value={extraInstructions}
                  onChange={(e) => setExtraInstructions(e.target.value)}
                />
              </div>
            </div>

            <button
              className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
              onClick={handleGenerateAssignment}
              disabled={loading}
            >
              Generate Assignment
            </button>
          </div>
        )}

        {loading && <p className="text-gray-500">Processing...</p>}

        {generatedAssignment && (
          <div className="mt-6 space-y-4">
            <div className="p-4 border rounded bg-gray-100">
              <h2 className="text-xl font-semibold mb-2">Generated Assignment</h2>
              <pre className="whitespace-pre-wrap text-sm">{generatedAssignment}</pre>
            </div>

            {/* Rubric Button (appears after assignment is generated) */}
            <button
              className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700"
              onClick={handleGenerateRubric}
              disabled={loading}
            >
              Generate Rubric
            </button>
          </div>
        )}

        {rubric && (
          <div className="mt-6 p-4 border rounded bg-yellow-100">
            <h2 className="text-xl font-semibold mb-2">Generated Rubric</h2>
            <pre className="whitespace-pre-wrap text-sm">{rubric}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
