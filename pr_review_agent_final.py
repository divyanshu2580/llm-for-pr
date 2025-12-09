import os
import sys
import json
import requests
from github import Github 
from google import genai

# --- Environment Variables and Constants ---

# Retrieve critical environment variables
api_key = os.getenv("GEMINI_API_KEY")
github_token = os.getenv("GITHUB_TOKEN")
repo_full_name = os.getenv("REPO_FULL_NAME") # e.g., 'owner/repo'

try:
    pr_number = int(os.getenv("PR_NUMBER"))
except (TypeError, ValueError):
    pr_number = None

semgrep_config = os.getenv("SEMGREP_CONFIG", "auto")
GEMINI_MODEL = "gemini-2.5-flash"


# --- Command Line Inputs ---

# Inputs passed from the CI runner (GitHub Actions run command)
diff_path = sys.argv[1] if len(sys.argv) > 1 else None
semgrep_path = sys.argv[2] if len(sys.argv) > 2 else None


# --- Input Reading Functions ---

def read_inputs():
    """Reads the Git diff and Semgrep JSON output files."""
    
    diff_content = ""
    if diff_path and os.path.exists(diff_path):
        with open(diff_path, "r", encoding="utf-8") as f:
            diff_content = f.read()

    # --- Trivial Diff Check ---
    if len(diff_content) < 10:
        # If diff is too small, post a message and exit cleanly.
        post_review("## Automated Gemini Code Review\n\n✅ **Agent Status:** No significant code diff detected to review.", skip_header=True)
        sys.exit(0)

    semgrep_json = {}
    if semgrep_path and os.path.exists(semgrep_path):
        try:
            with open(semgrep_path, "r", encoding="utf-8") as f:
                semgrep_json = json.load(f)
        except Exception as e:
            print(f"Warning: Could not parse Semgrep JSON: {e}")
            semgrep_json = {"error": str(e)}

    return diff_content, semgrep_json


# --- GitHub Posting Logic ---

def post_review(review_text, skip_header=False):
    """Posts the generated review as a comment on the Pull Request."""
    
    if not (github_token and repo_full_name and pr_number):
        print("Error: Missing GitHub token or PR metadata. Cannot post review.")
        print(f"Generated Review:\n{review_text}")
        return

    try:
        # Initialize PyGithub client
        github_client = Github(github_token)
        repo_obj = github_client.get_repo(repo_full_name)
        pr = repo_obj.get_pull(pr_number)
        
        # Determine if we should add the standard header
        if skip_header:
            final_comment = review_text
        else:
            header = f"## Automated Gemini Code Review for PR #{pr_number}\n\n"
            final_comment = header + review_text

        pr.create_issue_comment(final_comment)
        print("Successfully posted consolidated review to Pull Request.")

    except Exception as e:
        print(f"Error posting comment to GitHub: {e}")
        # In case of posting failure, print review to logs anyway
        print(f"Generated Review (not posted):\n{review_text}")
        sys.exit(1)


# --- LLM System Prompt and Execution ---

LLM_PERSONA = "You are an expert technical PR reviewer. Your tone is professional and focused on actionable risks. Analyze the provided Git diff and Semgrep metadata to produce a structured review."

def create_mega_prompt(diff_content, semgrep_json):
    """Constructs the main content (User Message) sent to the LLM."""
    
    findings = semgrep_json.get("results", [])

    content_template = f"""
### DIFF
```diff
{diff_content}
```

### SEMGREP FINDINGS
Total Findings: {len(findings)}

```json
{json.dumps(findings, indent=2, sort_keys=True)}
```

### REQUIRED OUTPUT FORMAT (Strictly adhere to this structure. Max 140 words total.)

### Summary
- 2–3 bullets detailing exact PR changes.

### Why It Matters
- Real reasoning based only on diff and security findings.

### Issues
- Synthesized list of real issues found in diff and tool output.

### Changes Required
- 1–2 essential fixes that need to be made before merging.

RULES:
- Do not output the markdown headers *inside* the bullet points.
- No hallucinations (use only diff and findings).
"""
    return content_template


def call_gemini_api(mega_prompt):
    """Calls the Gemini API with the structured prompt and persona."""
    
    if not api_key:
        print("Error: GEMINI_API_KEY is not set.")
        return "Gemini API Call Failed: API Key is missing from the environment."
        
    try:
        client = genai.Client(api_key=api_key)
        
        # We separate the content into the system instruction and the user message
        contents = [
            {"role": "system", "parts": [{"text": LLM_PERSONA}]},
            {"role": "user", "parts": [{"text": mega_prompt}]}
        ]

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents
        )
        return response.text.strip()
    
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return f"Gemini API Call Failed. Exception: {str(e)[:100]}..."


# --- Main Execution ---

if __name__ == "__main__":
    
    # Check for critical variables immediately at the start of execution
    if not (api_key and github_token and repo_full_name and pr_number):
        missing = []
        if not api_key: missing.append("GEMINI_API_KEY")
        if not github_token: missing.append("GITHUB_TOKEN")
        if not repo_full_name: missing.append("REPO_FULL_NAME")
        if pr_number is None: missing.append("PR_NUMBER")
        
        print(f"CRITICAL ERROR: Missing environment variables: {', '.join(missing)}. Please check YAML configuration.")
        sys.exit(1)

    # 1. Read Inputs (calls sys.exit(0) if diff is too small)
    diff_content, semgrep_json = read_inputs()

    # 2. Construct Prompt
    mega_prompt = create_mega_prompt(diff_content, semgrep_json)

    # 3. Call LLM
    review_output = call_gemini_api(mega_prompt)

    # 4. Post Output to GitHub
    post_review(review_output, skip_header=False)

    # For debugging in the CI runner
    print("\n--- Final Review Output ---\n")
    print(review_output)
