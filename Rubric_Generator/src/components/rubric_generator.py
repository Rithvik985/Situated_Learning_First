"""
Objective Rubric Generator

Given a text and its document type (thesis, film script, readme, situated_learning, qa),
use an LLM to generate a gold-standard rubric as a set of 10 objective evaluation questions.

Each question must:
- Be specific, observable, and minimally subjective.
- Together cover the most important quality dimensions for the given doc type.

Output format:
{
  "rubric_name": str,
  "doc_type": str,
  "questions": [str, str, ... 10 items]
}
"""

import os
import json
import requests
from typing import List, Dict, Optional

# ------------------------------
# Generic OpenAI-compatible client
# ------------------------------
class GenericLLM:
    def __init__(self, base_url: Optional[str], model: Optional[str], timeout: int = 120):
        self.base_url = (base_url or os.getenv("VLLM_URL_FOR_ANALYSIS"))
        self.model = model or os.getenv("VLLM_MODEL_FOR_ANALYSIS")
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.2, response_format: Optional[dict] = None) -> str:
        url = f"{self.base_url}"
        headers = {"Content-Type": "application/json"}
        payload: Dict = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

# ------------------------------
# Supported document types
# ------------------------------
# SUPPORTED_TYPES = {
#     "thesis": "Academic thesis: extended argument with lit review, methodology, results, discussion.",
#     "film_script": "Film or TV script: screenplay format, scenes, dialogue, stage directions.",
#     "readme": "README file documenting a software project.",
#     "situated_learning": "Situated learning assignment with scenario, objectives, reflection.",
#     "qa": "Question-answer response: answer to a prompt or exam question.",
# }

# ------------------------------
# Prompt templates
# ------------------------------
SYSTEM_PROMPT = """
You are an expert rubric designer.
Given a document type and text, create an objective evaluation rubric.

Instructions:
- Create exactly 4 rubric categories that capture key dimensions of evaluation 
- Under each rubric, write exactly 5 evaluation questions.
- Each question must be phrased as a yes/no or countable check (minimize subjectivity).
- Avoid vague words like "good", "appropriate", or "clear". Use measurable, verifiable criteria instead.
- Ensure the 4 rubrics + 20 questions together comprehensively cover evaluation for that document type.
- Most important of all - Make sure the questions in the rubric criteria are objective questions without any vague answers. Mostly yes/no type questions.


Return STRICT JSON only in the format:
{
  "rubric_name": str,
  "doc_type": str,
  "rubrics": [
    {
      "category": str,
      "questions": [str, str, str, str, str]
    },
    ...
    (exactly 4 rubric objects)
  ]
}
"""



def generate_objective_rubric(text: str, doc_type: str, client: Optional[GenericLLM] = None, model: Optional[str] = None) -> Dict:

    user_prompt = f"""
    Document type: {doc_type}

    Document snippet (for tailoring criteria, max 2000 chars):
    \"\"\"{text[:2000]}\"\"\"

    Now generate the rubric.
    """


    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        content = client.chat(messages, model=model, temperature=0.1, response_format={"type": "json_object"})
    except Exception:
        content = client.chat(messages, model=model, temperature=0.1)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        data = json.loads(content[start:end+1])

    data.setdefault("rubric_name", f"Objective Rubric for {doc_type.title()}")
    data.setdefault("doc_type", doc_type)
    return data


if __name__ == "__main__":

    client = GenericLLM(
        base_url="http://localhost:9091/v1/chat/completions",  # or your API endpoint
        model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"  # or whatever model you are using
    )
    sample_text = f"""### Assignment Title: Optimizing Manufacturing Processes with Genetic Algorithms

### Context:
In the manufacturing industry, optimizing production processes is crucial for reducing costs, improving efficiency, and enhancing product quality. Genetic algorithms, a type of optimization technique inspired by the process of natural selection, can be applied to solve complex problems in manufacturing, such as scheduling, resource allocation, and quality control. This assignment aims to equip students with the practical skills to apply genetic algorithms to real-world manufacturing challenges, providing immediate value to their organizations.

### Tasks:

1. **Process Optimization using Genetic Algorithms**:
   - Identify a specific manufacturing process in your current workplace that could benefit from optimization (e.g., production scheduling, inventory management, supply chain logistics).
   - Apply a genetic algorithm to this process using a programming language of your choice (e.g., Python, MATLAB).
   - Deliverables:
     * A detailed report on the selected process and the potential benefits of optimization.
     * A code snippet or pseudocode of the genetic algorithm applied to the process.
     * An analysis of the results, including any improvements in efficiency, cost savings, or quality enhancement.

2. **Comparative Analysis of Optimization Techniques**:
   - Research and compare at least two other optimization techniques (e.g., linear programming, simulated annealing) that could be applied to the same manufacturing process identified in Task 1.
   - Evaluate the strengths and weaknesses of each technique in the context of your selected process.
   - Deliverables:
     * A comparative table or chart highlighting the key features, advantages, and disadvantages of each optimization technique.
     * A brief case study or scenario where each technique might be preferred over the others.

3. **Implementation Plan for Genetic Algorithm Optimization**:
   - Based on the results from Task 1, develop a comprehensive plan to implement the genetic algorithm optimization in your workplace.
   - Consider factors such as data collection, computational resources, potential barriers to implementation, and strategies for overcoming these barriers.
   - Deliverables:
     * A step-by-step implementation plan, including timelines and responsible personnel.
     * A cost-benefit analysis of implementing the genetic algorithm optimization, including projected ROI and potential risks.

4. **Presentation and Reflection**:
   - Prepare a professional presentation to share your findings and implementation plan with your organization's leadership or relevant stakeholders.
   - Reflect on what you learned from this assignment and how it can be applied to future challenges in your career.
   - Deliverables:
     * A PowerPoint presentation or equivalent, summarizing your work and highlighting the practical applications and benefits of genetic algorithms in manufacturing.
     * A personal reflection essay (1-2 pages) on the professional development value of this assignment and potential future applications of genetic algorithms in your role.

### Implementation Guidance:
- Utilize industry-specific software and tools (e.g., SIMIO, AnyLogic, MATLAB) for modeling and optimizing manufacturing processes.
- Consult with colleagues and supervisors to ensure the relevance and feasibility of the selected process and optimization technique.
- Consider participating in industry webinars, workshops, or conferences to stay updated on the latest trends and best practices in manufacturing optimization.

By completing this assignment, students will develop practical skills in applying genetic algorithms to real-world manufacturing challenges, producing tangible outputs that can be implemented in their current workplace to add measurable value to their organization."""
    rubric = generate_objective_rubric(sample_text, "assignment",client=client)
    print(json.dumps(rubric, indent=2, ensure_ascii=False))
