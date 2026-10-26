import asyncio
import os
import urllib.parse
import httpx
from bs4 import BeautifulSoup
from typing import List
from loguru import logger
import random

from models.job import Job
from filters.date_filter import is_from_today
from config.settings import LINKEDIN_KEYWORDS

# How far back to look, in seconds. Must be larger than SCRAPE_INTERVAL + cycle runtime,
# so one failed/slow cycle doesn't lose jobs. 900 = 15 minutes (good for a 5-minute interval).
# Override with the env var LINKEDIN_POSTED_WITHIN_SECONDS.
POSTED_WITHIN_SECONDS = int(os.getenv("LINKEDIN_POSTED_WITHIN_SECONDS", "900"))

# Standard browser headers to avoid basic blocks
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
]

def get_random_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

async def fetch_jobs_page(client: httpx.AsyncClient, keywords: str, location: str, start: int = 0, retries: int = 5, cooldown_event: asyncio.Event = None) -> str:
    """Hits the LinkedIn Guest Jobs API endpoint with retries on 429.

    Args:
        cooldown_event: Optional shared Event. When set, all LinkedIn tasks
                        pause until the cooldown clears — prevents the remaining
                        tasks from hammering LinkedIn after a 429.
    """
    encoded_keywords = urllib.parse.quote(keywords)
    encoded_location = urllib.parse.quote(location)

    # f_TPR=r<seconds> limits to jobs posted within that window.
    # sortBy=DD sorts newest first (default is relevance, which can bury fresh jobs).
    url = (
        f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?"
        f"keywords={encoded_keywords}&location={encoded_location}"
        f"&f_TPR=r{POSTED_WITHIN_SECONDS}&sortBy=DD&start={start}"
    )

    for attempt in range(retries):
        # If another task triggered a global cooldown, wait for it to clear
        if cooldown_event is not None and cooldown_event.is_set():
            logger.debug(f"Global 429 cooldown active — pausing '{keywords}' request...")
            while cooldown_event.is_set():
                await asyncio.sleep(1.0)
            logger.debug(f"Cooldown cleared — resuming '{keywords}'.")

        try:
            response = await client.get(url, headers=get_random_headers(), timeout=15.0)
            if response.status_code == 429:
                wait_time = random.uniform(5.0, 10.0) * (2 ** attempt)
                logger.warning(f"Rate limited (429) by LinkedIn for '{keywords}'. Attempt {attempt + 1}/{retries}, backing off {wait_time:.1f}s...")
                # Signal global cooldown so sibling tasks pause too
                if cooldown_event is not None:
                    cooldown_event.set()
                    asyncio.get_event_loop().call_later(wait_time, cooldown_event.clear)
                await asyncio.sleep(wait_time)
                continue
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < retries - 1:
                wait_time = random.uniform(5.0, 10.0) * (2 ** attempt)
                logger.warning(f"HTTP 429 scraping LinkedIn. Attempt {attempt + 1}/{retries}, backing off {wait_time:.1f}s...")
                if cooldown_event is not None:
                    cooldown_event.set()
                    asyncio.get_event_loop().call_later(wait_time, cooldown_event.clear)
                await asyncio.sleep(wait_time)
                continue
            logger.error(f"HTTP error scraping LinkedIn (Status {e.response.status_code}): {e}")
            return ""
        except Exception as e:
            logger.error(f"Request error scraping LinkedIn: {e}")
            return ""

    logger.error(f"Max retries ({retries}) reached for LinkedIn scraping '{keywords}' in '{location}'.")
    return ""

# Title terms that count as a software role even if they don't match a configured keyword.
# (Removed "qa" and "tester" — add them back here if you want QA roles too.)
GENERAL_TERMS = [
    "developer", "programmer", "software", "backend", "back-end", "back end",
    "frontend", "front-end", "front end", "full stack", "full-stack", "fullstack",
    "tech lead", "architect", ".net", "dotnet", "c#", "java", "node", "python",
    "sde", "application engineer", "platform engineer", "api engineer", "web engineer",
]

def is_title_relevant(title: str) -> bool:
    title_lower = title.lower()
    keywords_lower = [k.lower() for k in LINKEDIN_KEYWORDS]

    # Accept if the title contains ANY of our target keywords
    for k in keywords_lower:
        if k in title_lower:
            return True

    # Also accept general software terms
    for t in GENERAL_TERMS:
        if t in title_lower:
            return True

    return False

def parse_jobs(html: str) -> List[Job]:
    """Parses the HTML fragment returned by the Guest API."""
    jobs = []
    if not html:
        return jobs

    soup = BeautifulSoup(html, "lxml")
    job_cards = soup.find_all("li")

    for card in job_cards:
        try:
            # The base Job entity ID is usually in an attribute or URL
            job_div = card.find("div", class_="base-card")
            if not job_div:
                continue

            job_id = job_div.get("data-entity-urn", "").split(":")[-1]
            if not job_id:
                continue

            # Title
            title_elem = card.find("h3", class_="base-search-card__title")
            title = title_elem.text.strip() if title_elem else "Unknown Title"

            if not is_title_relevant(title):
                continue

            # Company
            company_elem = card.find("h4", class_="base-search-card__subtitle")
            company = company_elem.text.strip() if company_elem else "Unknown Company"

            # Location
            location_elem = card.find("span", class_="job-search-card__location")
            location = location_elem.text.strip() if location_elem else "Unknown Location"

            # Date
            date_elem = card.find("time")
            posted_date = date_elem.text.strip() if date_elem else ""

            # Apply date filter strictly on the client side just in case 'f_TPR' failed
            # If date format is totally unrecognized, the filter allows it to prevent misses.
            if posted_date and not is_from_today(posted_date):
                continue

            # URL
            url_elem = card.find("a", class_="base-card__full-link")
            job_url = url_elem["href"].split("?")[0] if url_elem and "href" in url_elem.attrs else f"https://www.linkedin.com/jobs/view/{job_id}"

            job = Job(
                job_id=f"linkedin_{job_id}",
                title=title,
                company=company,
                location=location,
                job_url=job_url,
                posted_date=posted_date,
                source="LinkedIn"
            )
            jobs.append(job)

        except Exception as e:
            logger.debug(f"Error parsing a job card: {e}")
            continue

    return jobs

async def scrape_linkedin_jobs(keyword: str, location: str, max_pages: int = 1, cooldown_event: asyncio.Event = None) -> List[Job]:
    """
    Main scraping orchestration for LinkedIn.
    Scrapes multiple pages until no more jobs are found or max_pages is reached.
    """
    all_jobs = []
    logger.debug(f"Scraping LinkedIn for '{keyword}' in '{location}'")

    async with httpx.AsyncClient() as client:
        for page in range(max_pages):
            start_idx = page * 25 # LinkedIn API returns 25 results per page
            logger.debug(f"Fetching page {page + 1} (start={start_idx})...")

            html = await fetch_jobs_page(client, keyword, location, start=start_idx, cooldown_event=cooldown_event)
            if not html.strip():
                logger.debug("No more HTML returned. Ending pagination.")
                break

            jobs = parse_jobs(html)
            if not jobs:
                logger.debug("No valid jobs parsed from HTML fragment. Ending pagination.")
                break

            all_jobs.extend(jobs)

            # Respectful delay between pages
            if page < max_pages - 1:
                await asyncio.sleep(random.uniform(2.0, 4.0))

    logger.debug(f"LinkedIn found {len(all_jobs)} recent jobs for '{keyword}'.")
    return all_jobs
