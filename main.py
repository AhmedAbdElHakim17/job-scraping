import asyncio
import sys
import os
import random
import uuid
from aiohttp import web
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config.settings import (
    LOG_LEVEL, SCRAPE_INTERVAL_MINUTES, LINKEDIN_KEYWORDS, LINKEDIN_LOCATIONS,
    DEDUP_TTL_DAYS, LINKEDIN_DELAY_MIN, LINKEDIN_DELAY_MAX, LINKEDIN_MAX_CONCURRENCY,
    LINKEDIN_KEYWORDS_PER_CYCLE
)
from database.repository import JobRepository
from scraper.linkedin import scrape_linkedin_jobs
from telegram_bot.notifier import TelegramNotifier
from filters.company_filter import filter_muted_companies

# Configure Loguru
logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

_linkedin_keyword_offset = 0


def get_linkedin_keywords_for_cycle():
    """Select the next rotating batch while keeping the full list in use over time."""
    global _linkedin_keyword_offset

    if not LINKEDIN_KEYWORDS:
        return []

    batch_size = min(LINKEDIN_KEYWORDS_PER_CYCLE, len(LINKEDIN_KEYWORDS))
    start = _linkedin_keyword_offset
    selected = [
        LINKEDIN_KEYWORDS[(start + index) % len(LINKEDIN_KEYWORDS)]
        for index in range(batch_size)
    ]
    _linkedin_keyword_offset = (start + batch_size) % len(LINKEDIN_KEYWORDS)
    return selected


async def run_scraper_batch(tasks, semaphore, delay=0.5, cooldown_event=None):
    """Run a batch of scraper coroutines with bounded concurrency and delay."""
    results = []

    async def bounded(coro):
        async with semaphore:
            # If a global cooldown is active (e.g. LinkedIn 429), wait for it to clear
            if cooldown_event is not None and cooldown_event.is_set():
                logger.debug("Cooldown active — waiting before starting next task...")
                while cooldown_event.is_set():
                    await asyncio.sleep(1.0)
            result = await coro
            await asyncio.sleep(delay)
            return result

    wrapped = [bounded(t) for t in tasks]
    raw = await asyncio.gather(*wrapped, return_exceptions=True)

    for res in raw:
        if isinstance(res, list):
            results.extend(res)
        elif isinstance(res, Exception):
            logger.error(f"A scraper failed: {res}")

    return results


async def scrape_and_notify(repo: JobRepository, notifier: TelegramNotifier):
    """Main pipeline execution for a single scrape cycle (LinkedIn only)."""
    cycle_id = uuid.uuid4().hex[:6]
    linkedin_keywords = get_linkedin_keywords_for_cycle()
    logger.info(
        f"[{cycle_id}] Starting scrape cycle — LinkedIn batch: "
        f"{len(linkedin_keywords)}/{len(LINKEDIN_KEYWORDS)} keywords × {len(LINKEDIN_LOCATIONS)} locations"
    )

    # ── Cleanup expired dedup entries BEFORE scraping ──
    repo.cleanup_old_jobs(days=DEDUP_TTL_DAYS)

    # ── LinkedIn (low concurrency + delay to limit 429s) ──
    linkedin_tasks = []
    cooldown_event = asyncio.Event()  # Shared 429 cooldown across all LinkedIn tasks
    for loc in LINKEDIN_LOCATIONS:
        for key in linkedin_keywords:
            linkedin_tasks.append(
                scrape_linkedin_jobs(keyword=key, location=loc, max_pages=1, cooldown_event=cooldown_event)
            )

    delay = random.uniform(LINKEDIN_DELAY_MIN, LINKEDIN_DELAY_MAX)
    logger.info(f"[{cycle_id}] LinkedIn — {len(linkedin_tasks)} tasks ({LINKEDIN_MAX_CONCURRENCY} concurrent, {delay:.1f}s delay)...")
    linkedin_sem = asyncio.Semaphore(LINKEDIN_MAX_CONCURRENCY)
    all_jobs = await run_scraper_batch(linkedin_tasks, linkedin_sem, delay=delay, cooldown_event=cooldown_event)
    logger.info(f"[{cycle_id}] LinkedIn done — {len(all_jobs)} jobs collected.")

    # ── Company blocklist filter ──
    all_jobs = filter_muted_companies(all_jobs)
    logger.info(f"[{cycle_id}] After company filter: {len(all_jobs)} jobs remain.")

    # Store & Dedupe
    new_jobs_count = 0
    for job in all_jobs:
        if repo.insert_job(job):
            new_jobs_count += 1
    logger.info(f"[{cycle_id}] Deduplication complete. {new_jobs_count} brand new jobs added to database.")

    # Retrieve Pending & Notify (also retries anything that failed to send earlier)
    unsent_jobs = repo.get_unsent_jobs()
    if unsent_jobs:
        await notifier.send_job_alerts(unsent_jobs, repo)
    else:
        logger.info(f"[{cycle_id}] No unsent jobs pending for Telegram.")


async def main():
    repo = JobRepository()
    notifier = TelegramNotifier()

    await notifier.send_startup_message()

    logger.info("Running initial scrape cycle before scheduling...")
    await scrape_and_notify(repo, notifier)

    logger.info(f"Initializing APScheduler. Interval: {SCRAPE_INTERVAL_MINUTES} minutes.")
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scrape_and_notify,
        'interval',
        minutes=SCRAPE_INTERVAL_MINUTES,
        args=[repo, notifier],
        max_instances=1,        # Prevent overlapping cycles
        coalesce=True,          # If runs are missed, only execute once
        misfire_grace_time=300  # Tolerate up to 5 min of scheduler delay
    )

    scheduler.start()

    # --- Web Server for UptimeRobot pings ---
    async def health_check(request):
        return web.Response(text="Bot is running!")

    app = web.Application()
    app.router.add_get('/', health_check)

    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.environ.get("PORT", 8080))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Web server started on port {port} for UptimeRobot pings.")
    # -----------------------------------------

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Graceful shutdown initiated.")
        scheduler.shutdown()
        await runner.cleanup()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting...")
