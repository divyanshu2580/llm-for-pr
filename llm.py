import os
import sys
import json
from google import genai
from google.genai.errors import APIError

# --- CONFIGURATION ---

# ENV VARS
api_key = os.getenv("GEMINI_API_KEY")
pr_number = os.getenv("PR_NUMBER", "N/A") # Use N/A if not set
semgrep_config = os.getenv("SEMGREP_CONFIG", "auto")

# Exit immediately if API key is missing
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not set. LLM review cannot be generated.")
    sys.exit(1)

# INPUTS (Check arguments are provided)
diff_path = sys.argv[1] if len(sys.argv) > 1 else None
semgrep_path = sys.argv[2] if len(sys.argv) > 2 else None

if not diff_path:
    print("Error: PR Diff path not provided. Exiting.")
    sys.exit(1)


# --- DATA LOADING AND VALIDATION ---

# READ DIFF
diff_content = ""
if os.path.exists(diff_path):
    try:
        with open(diff_path, "r", encoding="utf-8") as f:
            diff_content = f.read()
    except Exception as e:
        print(f"Error reading diff file: {e}")
        sys.exit(1)

if len(diff_content.strip()) < 10:
    print("Diff missing or too small to review. Exiting.")
    # Exit code 0 so the workflow doesn't fail, but the review is skipped
    sys.exit(0)

# READ SEMGREP JSON
semgrep_json = {}
if semgrep_path and os.path.exists(semgrep_path):
    try:
        with open(semgrep_path, "r", encoding="utf-8") as f:
            semgrep_json = json.load(f)
    except json.JSONDecodeError:
        print("Warning: Failed to parse Semgrep JSON. Proceeding without findings.")
    except Exception as e:
        print(f"Warning: Error reading Semgrep file: {e}")

# Compile relevant Semgrep metadata
analysis_metadata = {
    "version": semgrep_json.get("version"),
    "total_findings": len(semgrep_json.get("results", [])),
    "errors": semgrep_json.get("errors", "None"),
    "paths_scanned": semgrep_json.get("paths", {}).get("scanned", "N/A"),
    # We include findings if there are any, otherwise it can be omitted
    "relevant_findings": semgrep_json.get("results", [])[:5] if semgrep_json.get("results") else "No findings."
}

# Save metadata for debugging/main branch tracking (already in your original script)
os.makedirs("analysis_output", exist_ok=True)
meta_file = "analysis_output/semgrep_metadata.json"
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(analysis_metadata, f, indent=2)

print(f"Saved Semgrep metadata → {meta_file}")


# --- GEMINI PROMPT ENGINEERING ---

# 1. System Prompt (The Expert Role and Data Context)
SystemPrompt = f"""
You are an expert, meticulous technical Pull Request (PR) reviewer.
Your goal is to analyze the provided PR diff and static analysis data, and generate a concise, structured review summary.

### INPUT DATA
- **PR Diff:**
{diff_content}

- **Semgrep Metadata (relevant findings included):**
{json.dumps(analysis_metadata, indent=2)}

### CONSTRAINTS
- The total review text must be concise (strive for under 140 words).
- Your review must adhere *strictly* to the output format provided in the User Prompt.
- **NEGATIVE CONSTRAINT:** 
-Only analyze and discuss the changes and context explicitly present in the 'PR Diff' block. Do not comment on existing code that was not modified in the diff.
- Do not invent information. Use only the provided diff and metadata.
"""

# 2. User Prompt (The Instruction and Desired Output Format)
UserPrompt = """
Generate the automated PR review summary based *only* on the input data above.

**Output Format (use markdown headings and bullets in this exact order):**

## Why It Matters
[A short paragraph explaining the impact (e.g., bug fix, new feature, performance)]

## Issues
[State any security/logic issues found in the diff or flagged by Semgrep metadata. If none, state: 'No critical issues found.']

## Changes Required
[List 1-2 essential, specific fixes needed for approval. If none, state: 'No immediate changes required.']

## Summary
- [Bullet 1: Main code change]
- [Bullet 2: Secondary code change]
"""

# --- CALL GEMINI ---
client = genai.Client(api_key=api_key)
try:
    # --- Using the explicit role-based structure ---
    contents = [
        {"role": "system", "parts": [{"text": SystemPrompt}]},
        {"role": "user", "parts": [{"text": UserPrompt}]}
    ]
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents, 
        config={"response_mime_type": "text/markdown"} # Explicitly request Markdown output
    )
    # Print the raw text output, which is captured by the GitHub Action
    print(response.text.strip())

except APIError as e:
    print(f"Error: Gemini API call failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)


