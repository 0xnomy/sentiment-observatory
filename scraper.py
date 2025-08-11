from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import logging
import json
import subprocess
import sys

import requests
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


def _clean_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    banner = "Skip to main content LinkedIn Top Content People Learning Jobs Games Get the app Join now Sign in"
    return cleaned.replace(banner, "").strip()


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return safe or "content"


def _slug_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    slug = path.replace("/", "-")
    return _sanitize_filename(slug)[:120] or "content"


def _guess_image_extension_from_url(img_url: str) -> str:
    path = urlparse(img_url).path
    _, _, last = path.rpartition('/')
    if '.' in last:
        ext = last.split('.')[-1].lower()
        if ext in {"jpg", "jpeg", "png", "gif", "webp"}:
            return "." + ext
    return ".jpg"


def _download_image(url: str, output_path: Path) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": url,
    }
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=30, headers=headers) as resp:
                if resp.status_code == 200:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    return
                else:
                    logging.warning("Image download non-200 (%s): %s", resp.status_code, url)
        except Exception as e:
            if attempt == 2:
                logging.warning("Image download failed: %s", e)


# ------------- LinkedIn -------------
def _is_login_page(page) -> bool:
    try:
        return (
            (page.locator("input[name='session_key']").count() > 0 and page.locator("input[name='session_password']").count() > 0)
            or (page.locator("form.login__form").count() > 0 and (page.locator("#username").count() > 0 or page.locator("input[name='session_key']").count() > 0))
        )
    except Exception:
        return False


def _login_if_needed(page, email: Optional[str], password: Optional[str], timeout_ms: int) -> None:
    # quick cookie consent dismissals commonly shown
    try:
        for sel in [
            "button:has-text('Accept')",
            "button:has-text('Accept all')",
            "button:has-text('Allow essential and optional cookies')",
        ]:
            btn = page.locator(sel).first
            if btn.count() > 0:
                btn.click(timeout=1500)
                break
    except Exception:
        pass

    # if already authenticated, nav should be present
    try:
        if page.locator("nav.global-nav__content, #global-nav").count() > 0 and page.locator("[data-test-global-nav-link='profile-nav-item'], img.global-nav__me-photo, button[aria-label*='Me']").count() > 0:
            return
    except Exception:
        pass

    # if a join/sign-in interstitial or header link is present, click Sign in
    try:
        for sel in [
            "a.nav__button-secondary[href*='/login']",
            "a[href*='/login'][data-tracking-control-name]",
            "a[href*='/login']",
            "button[aria-label='Sign in']",
            "a:has-text('Sign in')",
        ]:
            el = page.locator(sel).first
            if el.count() > 0:
                el.click(timeout=timeout_ms)
                break
    except Exception:
        pass

    # ensure we're at a login form (either full page or modal)
    if not _is_login_page(page):
        try:
            page.wait_for_selector("input[name='session_key'], form[action*='login']", timeout=timeout_ms)
        except Exception:
            pass

    if not _is_login_page(page):
        # try navigating directly
        page.goto("https://www.linkedin.com/login", timeout=timeout_ms)

    if not _is_login_page(page):
        return

    if not email or not password:
        raise RuntimeError("LinkedIn login required but LINKEDIN_EMAIL or LINKEDIN_PASSWORD missing.")

    # fill and submit using robust selectors
    try:
        # Preferred IDs from provided HTML
        page.fill("#username", email, timeout=timeout_ms)
    except Exception:
        page.fill("input[name='session_key']", email, timeout=timeout_ms)
    try:
        page.fill("#password", password, timeout=timeout_ms)
    except Exception:
        page.fill("input[name='session_password']", password, timeout=timeout_ms)

    # Click submit
    for sel in [
        "button[data-litms-control-urn='login-submit']",
        "form.login__form button[type='submit']",
        "button[aria-label='Sign in']",
        "button[type='submit']",
    ]:
        try:
            page.locator(sel).first.click(timeout=timeout_ms)
            break
        except Exception:
            continue

    # wait for authenticated UI or redirect to target
    try:
        page.wait_for_selector("nav.global-nav__content, #global-nav", timeout=timeout_ms)
    except Exception:
        pass

    # Detect OTP challenge and stop early (user action required)
    try:
        if page.locator("#otp-div:not(.hidden__imp), .otp-success-container").count() > 0:
            # Do not proceed automatically if OTP appears
            return
    except Exception:
        pass


def _scrape_linkedin(page, url: str, output_base: Path, timeout_ms: int) -> dict:
    # Wait for post area to load
    try:
        page.wait_for_selector(
            ".update-components-actor__container, div.fie-impression-container, .update-components-text",
            timeout=timeout_ms,
        )
    except Exception:
        pass

    # Expand where possible
    try:
        for sel in [
            ".feed-shared-inline-show-more-text button:has-text('See more')",
            "button:has-text('See more')",
            "button:has-text('Show more')",
            "button[aria-expanded='false']",
        ]:
            page.locator(sel).first.click(timeout=2000)
    except Exception:
        pass

    post_locator = page.locator("div.fie-impression-container")
    post_count = post_locator.count()
    first_post = post_locator.first if post_count > 0 else None

    def safe_inner_text(node_sel: str) -> str:
        try:
            if first_post is not None:
                return _clean_text(first_post.locator(node_sel).inner_text(timeout=2500))
            return _clean_text(page.locator(node_sel).first.inner_text(timeout=2500))
        except Exception:
            return ""

    # Author name
    author = safe_inner_text(".update-components-actor__title")
    if not author:
        # Try aria-label on actor avatar link, e.g. "View Khalil Ahmad’s  graphic link"
        try:
            target = (first_post or page).locator("a.update-components-actor__image").first
            aria = target.get_attribute("aria-label") or ""
            m = re.search(r"View\s*:?\s*(.*?)’", aria)
            if m:
                author = _clean_text(m.group(1))
        except Exception:
            pass

    headline = safe_inner_text(".update-components-actor__description")
    timestamp = safe_inner_text(".update-components-actor__sub-description")

    # Body text
    body = ""
    for body_sel in [
        ".feed-shared-inline-show-more-text .update-components-text",
        ".update-components-update-v2__commentary .update-components-text",
        ".update-components-text",
        "span.break-words.tvm-parent-container",
    ]:
        body = safe_inner_text(body_sel)
        if body:
            break

    image_srcs: list[str] = []
    try:
        images = (
            first_post.locator("img.update-components-image__image, .update-components-image__container img, button.update-components-image__image-link img")
            if first_post is not None
            else page.locator("img.update-components-image__image, .update-components-image__container img, button.update-components-image__image-link img")
        )
        for i in range(images.count()):
            try:
                src = images.nth(i).get_attribute("src")
                if src:
                    image_srcs.append(src)
            except Exception:
                pass
    except Exception:
        pass

    # Counts: reactions, comments, reposts
    def _first_number(text: str) -> str:
        m = re.search(r"\d[\d,\.KkMm]*", text or "")
        return m.group(0) if m else ""

    reactions = ""
    comments = ""
    reposts = ""
    try:
        # Prefer explicit fallback number when shown
        fallback = (first_post or page).locator(".social-details-social-counts__social-proof-fallback-number").first
        if fallback.count() > 0:
            reactions = _first_number(fallback.inner_text(timeout=1500))
        else:
            reactions = _first_number((first_post or page).locator(".social-details-social-counts__reactions-count").first.inner_text(timeout=1500))
    except Exception:
        pass
    try:
        comments = _first_number((first_post or page).locator(".social-details-social-counts__comments").first.inner_text(timeout=1500))
    except Exception:
        pass
    try:
        reposts = _first_number((first_post or page).locator("button[aria-label*='reposts']").first.inner_text(timeout=1500))
    except Exception:
        pass

    pieces = []
    if author:
        pieces.append(f"Author: {author}")
    if headline:
        pieces.append(f"Headline: {headline}")
    if timestamp:
        pieces.append(f"When: {timestamp}")
    if body:
        pieces.append(f"Body: {body}")
    if image_srcs:
        pieces.append("Images:\n" + "\n".join(image_srcs))
    if reactions:
        pieces.append(f"Reactions: {reactions}")
    if comments:
        pieces.append(f"Comments: {comments}")
    if reposts:
        pieces.append(f"Reposts: {reposts}")
    text = "\n\n".join([p for p in pieces if p]) or body or ""

    # name
    profile_name = _sanitize_filename(author.split("•")[0].split("Verified")[0].split("-")[0].strip()) or _slug_from_url(url)
    (output_base).mkdir(parents=True, exist_ok=True)
    (output_base / f"{profile_name}.txt").write_text(text, encoding="utf-8")
    local_images: list[str] = []
    for idx, src in enumerate(dict.fromkeys(image_srcs), start=1):
        ext = _guess_image_extension_from_url(src)
        out_fp = output_base / f"{profile_name}_image_{idx}{ext}"
        _download_image(src, out_fp)
        local_images.append(str(out_fp))

    return {
        "platform": "linkedin",
        "url": url,
        "title": "",
        "author": author,
        "headline": headline,
        "timestamp": timestamp,
        "body": body,
        "text": text,
        "images": local_images,
        "counts": {
            "reactions": reactions,
            "comments": comments,
            "reposts": reposts,
        },
        "slug": profile_name,
    }


# ------------- Reddit -------------
def _scrape_reddit(page, url: str, output_base: Path, timeout_ms: int) -> dict:
    # Try to accept cookies/overlays
    try:
        for sel in [
            "button:has-text('Accept all')",
            "button:has-text('Accept All')",
            "button:has-text('Accept')",
            "button:has-text('Continue')",
            "button:has-text('No thanks')",
            "button:has-text('Continue to site')",
        ]:
            page.locator(sel).first.click(timeout=1500)
    except Exception:
        pass

    # Try to get the post title via robust fallbacks
    title_text = ""
    try:
        # First, meta og:title
        og_title = page.locator("meta[property='og:title']").first
        if og_title.count() > 0:
            content = og_title.get_attribute("content") or ""
            title_text = _clean_text(content)
    except Exception:
        pass

    # Fallback: new Reddit full-post link anchors (with optional data-ks-id like t3_...)
    if not title_text:
        try:
            link = page.locator("a[slot='full-post-link'][data-ks-id^='t3_'], a[slot='full-post-link'][href*='/r/']").first
            if link.count() > 0:
                try:
                    title_text = _clean_text(link.locator("faceplate-screen-reader-content").inner_text(timeout=5000))
                except Exception:
                    title_text = _clean_text(link.inner_text())
        except Exception:
            pass

    # Fallback: try h1/h2 text near the post
    if not title_text:
        for sel in ["h1", "h2", "[data-test-id='post-content']"]:
            try:
                t = _clean_text(page.locator(sel).first.inner_text(timeout=6000))
                if t and len(t) > 5:
                    title_text = t
                    break
            except Exception:
                continue

    # Collect ONLY post images (avoid avatars/banners). Prefer images within post media container
    image_srcs: list[str] = []
    allowed_hosts = ("preview.redd.it", "i.redd.it", "i.reddituploads.com", "external-preview.redd.it")
    try:
        post = page.locator("shreddit-post").first
        if post.count() > 0:
            imgs = post.locator("[slot='post-media-container'] img.media-lightbox-img, gallery-carousel img.media-lightbox-img")
        else:
            # Fallback to post content container but still require media-lightbox-img where possible
            imgs = page.locator("div[data-test-id='post-content'] img.media-lightbox-img, article img.media-lightbox-img")

        count = imgs.count()
        for i in range(min(count, 4)):
            try:
                el = imgs.nth(i)
                src = el.get_attribute("src") or el.get_attribute("data-lazy-src")
                if not src:
                    continue
                # Only accept images from Reddit media hosts
                if any(host in src for host in allowed_hosts):
                    image_srcs.append(src)
            except Exception:
                pass
    except Exception:
        pass

    # If no images, try to extract text-body for self/text posts
    body_text = ""
    if not image_srcs:
        try:
            if post.count() > 0:
                md = post.locator("[slot='text-body'] .md").first
                if md.count() == 0:
                    md = post.locator("[slot='text-body']").first
                if md.count() > 0:
                    body_text = _clean_text(md.inner_text(timeout=4000))
            if not body_text:
                md = page.locator("div[data-test-id='post-content'] .md, [slot='text-body'] .md").first
                if md.count() > 0:
                    body_text = _clean_text(md.inner_text(timeout=3000))
        except Exception:
            pass

    # Extract best-effort upvote and comment counts
    def _first_number(text: str) -> str:
        m = re.search(r"\d[\d,\.KkMm]*", text or "")
        return m.group(0) if m else ""

    upvotes = ""
    comments = ""
    try:
        # Look for aria-label like "1,234 upvotes"
        aria_up = page.locator("[aria-label*='upvote']").first
        if aria_up.count() == 0:
            aria_up = page.locator("[aria-label*='upvotes']").first
        if aria_up.count() > 0:
            upvotes = _first_number(aria_up.get_attribute("aria-label") or "")
    except Exception:
        pass

    try:
        # Typical comments link
        cta = page.locator("a[data-click-id='comments']").first
        if cta.count() > 0:
            comments = _first_number(cta.inner_text())
        else:
            try:
                comments = _first_number(page.locator("a:has-text('comment')").first.inner_text(timeout=2000))
            except Exception:
                pass
        # Meta twitter:label2 may contain comments text sometimes
        if not comments:
            meta = page.locator("meta[name='twitter:label2']").first
            if meta.count() > 0:
                comments = _first_number(meta.get_attribute("content") or "")
    except Exception:
        pass

    text = title_text or ""
    name = _sanitize_filename((urlparse(url).path.split("/")[-1]) or "reddit_post")
    output_base.mkdir(parents=True, exist_ok=True)
    pieces = []
    if title_text:
        pieces.append(f"Title: {title_text}")
    if upvotes:
        pieces.append(f"Upvotes: {upvotes}")
    if comments:
        pieces.append(f"Comments: {comments}")
    # Include body only when there are no post images
    if not image_srcs and body_text:
        pieces.append(f"Body: {body_text}")
    final_text = "\n\n".join(pieces) or text
    (output_base / f"{name}.txt").write_text(final_text, encoding="utf-8")
    local_images: list[str] = []
    for idx, src in enumerate(dict.fromkeys(image_srcs), start=1):
        ext = _guess_image_extension_from_url(src)
        out_fp = output_base / f"{name}_image_{idx}{ext}"
        _download_image(src, out_fp)
        local_images.append(str(out_fp))

    return {
        "platform": "reddit",
        "url": url,
        "title": title_text,
        "author": (post.get_attribute("author") if 'post' in locals() and post and post.count() > 0 else ""),
        "timestamp": "",
        "body": body_text,
        "text": final_text,
        "images": local_images,
        "counts": {
            "upvotes": upvotes,
            "comments": comments,
        },
        "slug": name,
    }


# ------------- X / Twitter -------------
def _scrape_x(page, url: str, output_base: Path, timeout_ms: int) -> dict:
    # Locate main tweet article
    article = page.locator("article[data-testid='tweet']").first
    # Author display name and handle
    author = ""
    handle = ""
    text = ""
    try:
        author = _clean_text(article.locator("[data-testid='User-Name']").first.inner_text(timeout=2500))
    except Exception:
        pass
    try:
        handle = _clean_text(page.locator("a[href^='/' i] span:has-text('@')").first.inner_text(timeout=2000))
    except Exception:
        pass
    try:
        text = _clean_text(article.locator("[data-testid='tweetText']").first.inner_text(timeout=2500))
    except Exception:
        pass

    # Images
    image_srcs: list[str] = []
    try:
        imgs = article.locator("[data-testid='tweetPhoto'] img, img[alt='Image']")
        for i in range(imgs.count()):
            try:
                src = imgs.nth(i).get_attribute("src")
                if src and src.startswith("http"):
                    image_srcs.append(src)
            except Exception:
                pass
    except Exception:
        pass

    # Counts (reply, retweet, like)
    def _first_number(text: str) -> str:
        m = re.search(r"\d[\d,\.KkMm]*", text or "")
        return m.group(0) if m else ""

    replies = ""
    retweets = ""
    likes = ""
    try:
        replies = _first_number(article.locator("[data-testid='reply']").first.inner_text(timeout=1500))
    except Exception:
        pass
    try:
        retweets = _first_number(article.locator("[data-testid='retweet']").first.inner_text(timeout=1500))
    except Exception:
        pass
    try:
        likes = _first_number(article.locator("[data-testid='like']").first.inner_text(timeout=1500))
    except Exception:
        pass

    pieces = []
    if author:
        pieces.append(f"Author: {author}")
    if handle:
        pieces.append(f"Handle: {handle}")
    if text:
        pieces.append(f"Body: {text}")
    if likes:
        pieces.append(f"Likes: {likes}")
    if retweets:
        pieces.append(f"Reposts: {retweets}")
    if replies:
        pieces.append(f"Replies: {replies}")
    content = "\n\n".join(pieces) or text or ""

    name_base = (handle or author or urlparse(url).path.split("/")[-1] or "tweet").replace("@", "")
    name = _sanitize_filename(name_base)
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / f"{name}.txt").write_text(content, encoding="utf-8")
    local_images: list[str] = []
    for idx, src in enumerate(dict.fromkeys(image_srcs), start=1):
        ext = _guess_image_extension_from_url(src)
        out_fp = output_base / f"{name}_image_{idx}{ext}"
        _download_image(src, out_fp)
        local_images.append(str(out_fp))

    return {
        "platform": "x",
        "url": url,
        "title": "",
        "author": author,
        "handle": handle,
        "timestamp": "",
        "body": text,
        "text": content,
        "images": local_images,
        "counts": {
            "likes": likes,
            "reposts": retweets,
            "replies": replies,
        },
        "slug": name,
    }


def scrape_content(url: str, session_dir: str = ".playwright/session", headless: bool = False, timeout_ms: int = 20000) -> dict:
    load_dotenv()
    env_session = os.getenv("SESSION_DIR")
    if env_session:
        session_dir = env_session
    env_headless = os.getenv("HEADLESS")
    if env_headless is not None:
        headless = str(env_headless).lower() in {"1", "true", "yes"}
    env_timeout = os.getenv("SCRAPER_TIMEOUT_MS")
    if env_timeout and env_timeout.isdigit():
        timeout_ms = int(env_timeout)

    domain = urlparse(url).netloc.lower()
    session_path = Path(session_dir)
    session_path.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(user_data_dir=str(session_path), headless=headless)
        # Set tighter default timeouts for snappier scraping
        context.set_default_timeout(timeout_ms)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            page.goto(url, timeout=timeout_ms)

        # Explicit auth-required detection for LinkedIn
        if "linkedin.com" in domain:
            try:
                if page.locator("form.login__form, #username, input[name='session_key']").count() > 0:
                    return {
                        "platform": "linkedin",
                        "url": url,
                        "text": "",
                        "images": [],
                        "counts": {},
                        "slug": _slug_from_url(url),
                        "error": "login_required",
                        "message": "LinkedIn login required. Please sign in in the persistent browser profile."
                    }
            except Exception:
                pass

        if "linkedin.com" in domain:
            out_dir = Path("output/linkedin")
            result = _scrape_linkedin(page, url, out_dir, timeout_ms)
        elif "reddit.com" in domain:
            out_dir = Path("output/reddit")
            result = _scrape_reddit(page, url, out_dir, timeout_ms)
        elif "twitter.com" in domain or "x.com" in domain:
            out_dir = Path("output/x")
            result = _scrape_x(page, url, out_dir, timeout_ms)
        else:
            result = {"platform": "unknown", "url": url, "text": "", "images": [], "counts": {}, "slug": _slug_from_url(url)}

        context.close()
        return result


def scrape_content_subprocess(url: str, timeout_sec: int = 90) -> dict:
    """Run scraper in a separate Python process to avoid event loop conflicts."""
    cmd = [sys.executable, str(Path(__file__).resolve()), url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except Exception as e:
        raise RuntimeError(f"scraper subprocess failed to start: {e}")
    if proc.returncode != 0:
        raise RuntimeError(f"scraper subprocess error: {proc.stderr.strip() or proc.stdout.strip()}")
    out = proc.stdout.strip()
    try:
        return json.loads(out)
    except Exception as e:
        raise RuntimeError(f"failed to parse scraper output as JSON: {e}; output={out[:500]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scraper.py <url>", file=sys.stderr)
        raise SystemExit(1)
    try:
        result = scrape_content(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        raise SystemExit(2)


