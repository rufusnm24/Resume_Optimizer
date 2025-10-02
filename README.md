# Resume Optimizer Agent

An end-to-end automation agent that harvests LinkedIn job descriptions, optimizes a LaTeX resume for ATS performance, and compiles updated artifacts.

## Features
- Playwright-based LinkedIn harvester with optional manual mode for compliance.
- Keyword extraction and ATS scoring across coverage, format, quality, and distribution (OpenAI-assisted when available).
- LaTeX-aware rewriting that preserves macros and constrains bullet edits.
- PDF compilation via optional cloud API with local fallbacks.
- Transparent reports including diff, keyword usage map, and score breakdown.

## Prerequisites
- Python 3.9 or newer
- Google Chrome or Microsoft Edge (required by Playwright)

## Setup
1. Create and activate a virtual environment:
   - Linux/macOS: `python -m venv .venv && source .venv/bin/activate`
   - Windows PowerShell: `python -m venv .venv; .venv\Scripts\Activate.ps1`
2. Install dependencies: `pip install -r requirements.txt`
3. Install Playwright browsers if you plan to use automated harvesting: `playwright install`
4. Copy the environment template and populate it:
   - Linux/macOS: `cp .env.example .env`
   - Windows PowerShell: `Copy-Item .env.example .env`
5. Edit `.env` and provide the required keys:
   - `LINKEDIN_EMAIL` / `LINKEDIN_PASSWORD` for automated scraping (leave blank when using manual mode)
   - `OPENAI_API_KEY` for enhanced keyword extraction and bullet rewriting
   - Optional `LATEX_API_ENDPOINT` for remote PDF compilation

## Running the Agent
### Full Pipeline (harvest + optimize)
```bash
python -m cli.resume_opt pipeline   --job-titles "Data Analyst, Project Manager"   --locations "New York, Remote"   --resume-path "~/resumes/main.tex"   --output-dir "./artifacts"   --ats-threshold 82 --strict
```
- Use `--manual-mode` to skip Playwright and feed local job descriptions (see next section).
- Pass `--use-openai false` to disable OpenAI usage even when the key is present.

### Manual Optimization
```bash
python -m cli.resume_opt optimize   --resume-path "~/resumes/main.tex"   --jd-file "./samples/jd_pm.txt"   --ats-threshold 85 --strict
```
You can supply `.txt` job descriptions or JSON exports created by the harvester.

## Outputs
Each run writes artifacts into the chosen `--output-dir`:
- `main_optimized.tex`: rewritten LaTeX resume
- `diff.patch`: unified diff versus the original resume
- `keyword_map.json`: before/after keyword counts
- `report.md`: human-readable ATS summary
- `Resume_Optimized.pdf`: compiled resume (requires LaTeX toolchain or API)

## Testing and Quality
```bash
pytest
ruff check .
mypy .
```

## Troubleshooting
- Missing `OPENAI_API_KEY` falls back to rule-based keyword extraction and rewriting.
- If Playwright login fails, run in `--manual-mode` and provide job descriptions collected elsewhere.
- Use `--strict` to limit OpenAI rewrites to +/- 10 characters per bullet for tighter control.
