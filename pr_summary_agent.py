import os
import sys
from github import Github
from google import genai

# --- Configuration & Setup ---

# Define the models and persona
GEMINI_MODEL = 'gemini-2.5-flash'  # Fast and cost-effective model
SYSTEM_PROMPT = (
    "You are a meticulous, senior software engineer and automated code reviewer. "
    "Your review is critical and focused on high-impact issues only. "
    "Analyze the provided Git diff for the following:\n"
    "1. **Security Vulnerabilities:** Look for exposed secrets, poor input validation, or injection risks.\n"
    "2. **Logic & Potential Bugs:** Identify possible NullPointerExceptions, infinite loops, or incorrect error handling.\n"
    "3. **Maintainability:** Comment on excessive complexity, unclear variable names, or missing documentation for new functions.\n"
    "4. **Efficiency:** Suggest performance improvements where applicable.\n\n"
    "Respond ONLY in clean Markdown. Start with a header '## 🤖 Gemini AI Review Findings' "
    "followed by a numbered list of issues. For each issue, provide the file name and the exact line number if possible. "
    "If no issues are found, state: 'No critical issues found. This PR looks clean!'"
)

# --- Environment Variable Check ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
PR_NUMBER = os.environ.get("PR_NUMBER")
REPO_FULL_NAME = os.environ.get("REPO_FULL_NAME")

if not all([GEMINI_API_KEY, GITHUB_TOKEN, PR_NUMBER, REPO_FULL_NAME]):
    print("Error: Missing required environment variables.")
    sys.exit(1)

try:
    PR_NUMBER = int(PR_NUMBER)
except ValueError:
    print("Error: PR_NUMBER is not a valid integer.")
    sys.exit(1)

# --- Main Logic ---

def fetch_pr_diff(repo, pr_number):
    """
    Step 4 & 5: Fetches the changed files and the combined diff content from GitHub API.
    """
    try:
        pr = repo.get_pull(pr_number)
        
        # 5. Get the list of files changed in the PR
        files = pr.get_files()
        
        diff_content = ""
        for file in files:
            # We use the patch property which contains the unified diff format
            if file.patch:
                diff_content += f"--- File: {file.filename} ---\n"
                diff_content += file.patch
                diff_content += "\n\n"
        
        if not diff_content:
            return "No detectable code changes (diff) found in this Pull Request."
        
        return diff_content

    except Exception as e:
        return f"Error fetching PR data from GitHub API: {e}"

def generate_review(diff):
    """
    Step 6: Runs the core LLM logic (Gemini) to analyze the diff.
    """
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # The prompt payload combines the System Instruction and the User Query (the diff)
        contents = [
            {"role": "user", "parts": [
                {"text": SYSTEM_PROMPT},
                {"text": f"Analyze the following Git Diff:\n\n```diff\n{diff}\n```"}
            ]}
        ]
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents
        )
        return response.text

    except Exception as e:
        return f"Error calling Gemini API: {e}"

def post_review(repo, pr_number, review_text):
    """
    Step 6 (Cont.): Posts the generated review as a general PR comment.
    """
    try:
        pr = repo.get_pull(pr_number)
        pr.create_issue_comment(review_text)
        print("Successfully posted review to Pull Request.")

    except Exception as e:
        print(f"Error posting comment to GitHub: {e}")
        # Fail the job if we can't post the review
        sys.exit(1)


if __name__ == "__main__":
    # Initialize GitHub client using the token provided by GitHub Actions
    github_client = Github(GITHUB_TOKEN)
    repo_obj = github_client.get_repo(REPO_FULL_NAME)

    # 1. Fetch the diff
    pr_diff = fetch_pr_diff(repo_obj, PR_NUMBER)
    
    # 2. Check if fetching was successful and run analysis
    if pr_diff.startswith("Error"):
        final_review = f"## 🤖 Gemini AI Review Failure\n\n{pr_diff}"
    elif pr_diff.startswith("No detectable code changes"):
        # Skip LLM call if no code changed
        final_review = f"## 🤖 Gemini AI Review\n\n{pr_diff}"
    else:
        # 3. Generate the review using Gemini
        final_review = generate_review(pr_diff)

    # 4. Post the final review
    post_review(repo_obj, PR_NUMBER, final_review)
