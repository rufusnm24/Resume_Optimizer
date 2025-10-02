"""LinkedIn harvesting utilities.

The scraper favours Playwright for deterministic browser automation.  For tests we
expose a manual mode that reads job descriptions from text/JSON files so the rest of
the pipeline can run without network access.  Each collected posting is persisted to
``artifacts/jobs``.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

@dataclass
class ManualJob:
    """Structure used for manual mode input files."""

    title: str
    company: str
    description: str
    location: str = ""
    seniority: Optional[str] = None
    url: str = "manual://entry"

    def __post_init__(self) -> None:
        self.description = " ".join(self.description.split())


@dataclass
class JobPosting:
    """Normalized view of a job posting."""

    title: str
    company: str
    location: str
    seniority: Optional[str]
    description: str
    url: str
    keywords: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)





def clear_job_artifacts(target: Path | None = None) -> None:
    """Remove any previously harvested job files."""
    directory = target or Path("artifacts/jobs")
    if not directory.exists():
        return
    for path in directory.iterdir():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            continue

class LinkedInScraper:
    """Headless LinkedIn job harvester with polite scraping defaults."""

    def __init__(
        self,
        email: Optional[str],
        password: Optional[str],
        *,
        manual_mode: bool = False,
        delay_seconds: float = 2.0,
        authenticate: bool = False,
        max_postings: int = 1,
        job_output_dir: Optional[Path] = None,
    ) -> None:
        self.email = email
        self.password = password
        self.manual_mode = manual_mode
        self.delay_seconds = delay_seconds
        self.authenticate = authenticate
        self.max_postings = max(1, max_postings)
        self.job_output_dir = job_output_dir or Path("artifacts/jobs")
        self.job_output_dir.mkdir(parents=True, exist_ok=True)

    # Public API -----------------------------------------------------------------
    def harvest(
        self,
        job_titles: Sequence[str],
        locations: Sequence[str],
        *,
        manual_inputs: Optional[Sequence[ManualJob]] = None,
    ) -> List[JobPosting]:
        """Return harvested postings for titles/locations.

        When ``manual_mode`` is enabled the caller *must* provide ``manual_inputs``.
        The entries are stored as JSON artifacts alongside the fully scraped output
        so the rest of the pipeline remains identical.
        """

        if self.manual_mode:
            if not manual_inputs:
                raise ValueError("manual_mode requires manual_inputs")
            postings = [self._from_manual(job) for job in manual_inputs]
        else:
            postings = self._automated_harvest(job_titles, locations)

        for posting in postings:
            self._persist(posting)
        return postings

    # Manual mode helpers --------------------------------------------------------
    @staticmethod
    def load_manual_file(path: Path) -> ManualJob:
        """Load a manual job description from JSON or plain text."""

        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            return ManualJob(**data)
        except json.JSONDecodeError:
            return ManualJob(title=path.stem, company="Manual", description=text, location="", url=f"manual://{path.stem}")

    def _from_manual(self, manual_job: ManualJob) -> JobPosting:
        keywords = self._simple_keywords(manual_job.description)
        return JobPosting(
            title=manual_job.title,
            company=manual_job.company,
            location=manual_job.location,
            seniority=manual_job.seniority,
            description=manual_job.description,
            url=manual_job.url,
            keywords=keywords,
        )

    # Automated scraping ---------------------------------------------------------
    def _automated_harvest(self, job_titles: Sequence[str], locations: Sequence[str]) -> List[JobPosting]:
        """Scrape LinkedIn using Playwright if available, falling back to Selenium."""

        try:
            from playwright.sync_api import sync_playwright
        except Exception:  # pragma: no cover - optional dependency branch
            sync_playwright = None

        if not sync_playwright:
            raise RuntimeError(
                "Playwright is not installed. Install playwright and run `playwright install` "
                "or use manual mode via --manual-mode."
            )

        postings: List[JobPosting] = []
        with sync_playwright() as p:  # pragma: no cover - network automation not unit tested
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            if self.authenticate and self.email and self.password:
                self._login(page)
            for title in job_titles:
                for location in locations:
                    postings.extend(self._search_and_collect(page, title, location))
                    if len(postings) >= self.max_postings:
                        break
                    time.sleep(self.delay_seconds)
                if len(postings) >= self.max_postings:
                    break
            browser.close()
        return postings[: self.max_postings]

    def _login(self, page) -> None:  # pragma: no cover - requires browser
        if not self.email or not self.password:
            return
        page.goto("https://www.linkedin.com/login")
        page.fill("input[id=username]", self.email)
        page.fill("input[id=password]", self.password)
        page.click("button[type=submit]")
        print("LinkedIn login submitted. Complete any verification within 10 seconds...")
        page.wait_for_timeout(10000)

    def _dismiss_login_prompt(self, page) -> bool:  # pragma: no cover - requires browser
        selectors = [
            "button[aria-label='Dismiss']",
            "button[aria-label='Dismiss this message']",
            "button[data-test-modal-close-btn]",
            "button[data-control-name='overlay.close']",
            "button.artdeco-modal__dismiss",
            "button[aria-label='Close']",
            ".artdeco-hoverable-content__close",
            ".artdeco-toast-item__dismiss",
        ]
        for _ in range(5):
            for selector in selectors:
                try:
                    element = page.query_selector(selector)
                    if element:
                        element.click()
                        page.wait_for_timeout(500)
                        return True
                except Exception:
                    continue
            try:
                page.keyboard.press('Escape')
                page.wait_for_timeout(300)
            except Exception:
                pass
        return False

    def _search_and_collect(self, page, title: str, location: str) -> List[JobPosting]:  # pragma: no cover
        query = title.replace(" ", "%20")
        loc = location.replace(" ", "%20")
        page.goto(f"https://www.linkedin.com/jobs/search/?keywords={query}&location={loc}")
        page.wait_for_timeout(int(self.delay_seconds * 1000))
        self._dismiss_login_prompt(page)
        postings: List[JobPosting] = []
        elements = page.query_selector_all(".job-card-container--clickable")
        for element in elements:
            if len(postings) >= self.max_postings:
                break
            try:
                element.click()
                page.wait_for_timeout(int(self.delay_seconds * 1000))
                title_text = page.query_selector(".jobs-unified-top-card__job-title")
                company_el = page.query_selector(".jobs-unified-top-card__company-name")
                location_el = page.query_selector(".jobs-unified-top-card__bullet")
                company_text = company_el.inner_text().strip() if company_el else "Unknown"
                location_text = location_el.inner_text().strip() if location_el else location
                seniority_text_element = page.query_selector(".jobs-unified-top-card__job-insight")
                seniority = seniority_text_element.inner_text().strip() if seniority_text_element else None
                description_el = page.query_selector(".jobs-description__content")
                description_text = description_el.inner_text().strip() if description_el else ""
                url = page.url
                keywords = self._simple_keywords(description_text)
                postings.append(
                    JobPosting(
                        title=title_text.inner_text().strip() if title_text else title,
                        company=company_text,
                        location=location_text,
                        seniority=seniority,
                        description=description_text,
                        url=url,
                        keywords=keywords,
                    )
                )
                return postings
            except Exception:
                continue
        return postings

    # Utility --------------------------------------------------------------------
    def _persist(self, posting: JobPosting) -> None:
        target = self.job_output_dir / f"{self._slug(posting.title)}_{self._slug(posting.company)}.json"
        target.write_text(posting.to_json(), encoding="utf-8")
        (self.job_output_dir / "job_description.txt").write_text(posting.description, encoding="utf-8")

    @staticmethod
    def _slug(value: str) -> str:
        safe = value.lower().replace(" ", "-")
        return "".join(ch for ch in safe if ch.isalnum() or ch == "-")

    @staticmethod
    def _simple_keywords(description: str, limit: int = 15) -> List[str]:
        words = [w.strip(".,:- ").lower() for w in description.split() if len(w) > 3]
        freq: dict[str, int] = {}
        for word in words:
            if not word.isalpha():
                continue
            freq[word] = freq.get(word, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
        return [word for word, _ in sorted_words[:limit]]


def load_manual_jobs_from_paths(paths: Iterable[os.PathLike[str] | str]) -> List[ManualJob]:
    """Helper used by CLI to load manual job descriptions from arbitrary files."""

    results: List[ManualJob] = []
    for path_like in paths:
        path = Path(path_like)
        results.append(LinkedInScraper.load_manual_file(path))
    return results
