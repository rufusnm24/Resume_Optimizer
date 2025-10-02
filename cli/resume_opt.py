"""Typer CLI orchestrating harvesting, optimization, and compilation."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import List

import typer
try:  # pragma: no cover - optional dependency for local environments
    from dotenv import load_dotenv
except ImportError:  # fallback for test environments without python-dotenv
    def load_dotenv(*args, **kwargs):  # type: ignore[override]
        return False
try:  # pragma: no cover - optional dependency for local environments
    from rich.console import Console
except ImportError:  # Provide minimal shim when rich is absent
    class Console:  # type: ignore[override]
        def rule(self, message: str) -> None:
            print(f"--- {message} ---")

        def print(self, message: str) -> None:
            print(message)

from ats.keyword_extract import KeywordCandidate, extract_keywords
from openai_utils import validate_openai_setup as _check_openai
from ats.scorer import ATSScorer, summarise_keywords
from compile.pdf_compile import PDFCompiler
from harvest.linkedin_scraper import LinkedInScraper, ManualJob, load_manual_jobs_from_paths
from latex.ast_parser import parse_document
from latex.rewriter import optimize_resume

console = Console()
app = typer.Typer(help="AI-powered resume optimizer pipeline")


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_openai_setup() -> bool:
    """Check if OpenAI is properly configured."""
    ok, error = _check_openai()
    if ok:
        console.print('[green]OpenAI API key validated successfully[/green]')
        return True

    if error == 'OPENAI_API_KEY environment variable is not set':
        console.print('[yellow]Warning: OPENAI_API_KEY not found. Using basic extraction methods.[/yellow]')
    elif error == 'openai package is not installed':
        console.print('[yellow]Warning: OpenAI package not installed. Install with: pip install openai[/yellow]')
    else:
        console.print(f"[yellow]Warning: {error or 'OpenAI validation failed'}. Falling back to basic extraction methods.[/yellow]")
    return False


def _optimize(
    *,
    resume_path: Path,
    job: ManualJob,
    output_dir: Path,
    ats_threshold: float,
    strict: bool,
    use_openai: bool = True,
) -> dict:
    resume_text = resume_path.read_text(encoding="utf-8")
    keywords = extract_keywords(job.description, use_openai=use_openai)
    document = parse_document(resume_text)
    scorer = ATSScorer()
    before = scorer.score(
        resume_text=resume_text,
        bullet_texts=document.bullet_texts,
        keywords=keywords,
        sections_present=document.section_names,
        page_estimate=document.page_estimate(),
    )
    rewrite = optimize_resume(resume_text, keywords, strict=strict, use_openai=use_openai)
    optimized_path = output_dir / "main_optimized.tex"
    optimized_path.write_text(rewrite.optimized_tex, encoding="utf-8")

    diff_path = output_dir / "diff.patch"
    diff_path.write_text(rewrite.diff, encoding="utf-8")

    keyword_map_path = output_dir / "keyword_map.json"
    keyword_map_path.write_text(json.dumps(rewrite.keyword_map, indent=2), encoding="utf-8")

    optimized_document = parse_document(rewrite.optimized_tex)
    after = scorer.score(
        resume_text=rewrite.optimized_tex,
        bullet_texts=optimized_document.bullet_texts,
        keywords=keywords,
        sections_present=optimized_document.section_names,
        page_estimate=optimized_document.page_estimate(),
    )

    report_path = output_dir / "report.md"
    report_path.write_text(
        _render_report(job, keywords, before, after, ats_threshold),
        encoding="utf-8",
    )

    pdf_compiler = PDFCompiler()
    pdf_path = output_dir / "Resume_Optimized.pdf"
    pdf_compiler.compile(optimized_path, pdf_path)

    keyword_summary = summarise_keywords(keywords, optimized_document.bullet_texts)
    return {
        "before": asdict(before),
        "after": asdict(after),
        "keyword_summary": keyword_summary,
        "meets_threshold": after.total >= ats_threshold,
        "paths": {
            "optimized_tex": str(optimized_path),
            "diff": str(diff_path),
            "keyword_map": str(keyword_map_path),
            "report": str(report_path),
            "pdf": str(pdf_path),
        },
    }


def _render_report(job: ManualJob, keywords: List[KeywordCandidate], before, after, ats_threshold: float) -> str:
    lines = [
        f"# Resume Optimization Report for {job.title} at {job.company}",
        "",
        "## Job Summary",
        f"- Location: {job.location or 'N/A'}",
        f"- Seniority: {job.seniority or 'N/A'}",
        f"- URL: {job.url}",
        "",
        "## ATS Scores",
        f"- Target threshold: {ats_threshold}",
        f"- Before: {before.total} (coverage {before.coverage}, format {before.section}, quality {before.quality}, distribution {before.distribution})",
        f"- After: {after.total} (coverage {after.coverage}, format {after.section}, quality {after.quality}, distribution {after.distribution})",
        "",
        "## Keyword Focus",
    ]
    for keyword in keywords:
        lines.append(f"- {keyword.token} (synonyms: {', '.join(keyword.synonyms) or 'none'})")
    lines.append("")
    lines.append("Generated by resume-optimizer-agent.")
    return "\n".join(lines)


@app.command()
def pipeline(
    *,
    job_titles: str = typer.Option(..., help="Comma-separated job titles"),
    locations: str = typer.Option(..., help="Comma-separated job locations"),
    resume_path: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("artifacts")),
    ats_threshold: float = typer.Option(80.0, help="Minimum ATS score"),
    strict: bool = typer.Option(False, help="Restrict bullet edits to +/- 10 chars"),
    manual_mode: bool = typer.Option(False, help="Skip LinkedIn automation"),
    manual_jd: List[Path] = typer.Option(None, help="Manual JD files (text or JSON)"),
    use_openai: bool = typer.Option(True, help="Use OpenAI for enhanced optimization"),
) -> None:
    """Run the full harvesting + optimization pipeline."""

    load_dotenv()
    
    # Validate OpenAI setup if requested
    openai_available = False
    if use_openai:
        openai_available = _validate_openai_setup()
    
    titles = _split_csv(job_titles)
    locs = _split_csv(locations)
    output_dir.mkdir(parents=True, exist_ok=True)

    email = os.getenv("LINKEDIN_EMAIL")
    password = os.getenv("LINKEDIN_PASSWORD")
    scraper = LinkedInScraper(email, password, manual_mode=manual_mode)

    if manual_mode:
        manual_jobs = load_manual_jobs_from_paths(manual_jd)
    else:
        manual_jobs = None

    postings = scraper.harvest(titles, locs, manual_inputs=manual_jobs)

    best_result = None
    for posting in postings:
        manual_job = ManualJob(
            title=posting.title,
            company=posting.company,
            location=posting.location,
            seniority=posting.seniority,
            description=posting.description,
            url=posting.url,
        )
        console.rule(f"Optimizing for {manual_job.title} @ {manual_job.company}")
        result = _optimize(
            resume_path=resume_path,
            job=manual_job,
            output_dir=output_dir,
            ats_threshold=ats_threshold,
            strict=strict,
            use_openai=openai_available and use_openai,
        )
        if result["meets_threshold"]:
            console.print(f"[green]Achieved ATS {result['after']['total']}[/green]")
            best_result = result
            break
        if not best_result or result["after"]["total"] > best_result["after"]["total"]:
            best_result = result

    if not best_result:
        raise typer.Exit(code=1)

    console.print("Artifacts saved:")
    for name, path in best_result["paths"].items():
        console.print(f"- {name}: {path}")


@app.command()
def optimize(
    *,
    resume_path: Path = typer.Option(..., exists=True),
    jd_file: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("artifacts")),
    ats_threshold: float = typer.Option(80.0),
    strict: bool = typer.Option(False),
    use_openai: bool = typer.Option(True, help="Use OpenAI for enhanced optimization"),
) -> None:
    """Optimize a resume using a manual job description file."""

    load_dotenv()
    
    # Validate OpenAI setup if requested
    openai_available = False
    if use_openai:
        openai_available = _validate_openai_setup()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    job = LinkedInScraper.load_manual_file(jd_file)
    result = _optimize(
        resume_path=resume_path,
        job=job,
        output_dir=output_dir,
        ats_threshold=ats_threshold,
        strict=strict,
        use_openai=openai_available and use_openai,
    )
    console.print(f"Optimized resume saved to {result['paths']['optimized_tex']}")
    console.print(f"PDF saved to {result['paths']['pdf']}")


if __name__ == "__main__":
    app()
