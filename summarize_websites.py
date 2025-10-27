#!/usr/bin/env python3
import argparse, asyncio, json, math, re, sys, time, heapq, itertools
from typing import Optional, Dict, Any, List, Tuple

import aiohttp
from datasets import load_dataset
from bs4 import BeautifulSoup
import extruct
from w3lib.html import get_base_url
import trafilatura
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
USER_AGENT = "Mozilla/5.0 (compatible; website-summary-bot/0.1; +https://example.com/contact)"
MAX_HTML_BYTES = 1_800_000  # ~1.8MB cap per page
MIN_DESC_LEN = 40           # heuristically prefer non-trivial blurbs
DEFAULT_CONCURRENCY = 16

def to_int_safe(x) -> int:
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        if math.isnan(x):
            return 0
        return int(x)
    s = str(x).strip().replace(",", "").replace("+", "")
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else 0

def candidate_urls(domain: str) -> List[str]:
    d = domain.strip()
    d = re.sub(r"^https?://", "", d, flags=re.I)
    d = d.strip("/")

    # Try a few common variants in a sensible order
    return [
        f"https://{d}/",
        f"http://{d}/",
        f"https://www.{d}/",
        f"http://www.{d}/",
    ]

def first_nonempty(*vals) -> Optional[str]:
    for v in vals:
        if v and isinstance(v, str):
            s = v.strip()
            if s:
                return s
    return None

def extract_meta_and_ld(html: str, url: str) -> Tuple[Optional[str], Optional[str]]:
    soup = BeautifulSoup(html, "lxml")

    def meta(name=None, prop=None):
        if name:
            tag = soup.find("meta", attrs={"name": name})
            if tag and tag.get("content"):
                return tag["content"].strip()
        if prop:
            tag = soup.find("meta", attrs={"property": prop})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return None

    title = soup.title.string.strip() if soup.title and soup.title.string else None
    og_title = meta(prop="og:title")
    tw_title = meta(name="twitter:title")
    description = first_nonempty(
        meta(name="description"),
        meta(prop="og:description"),
        meta(name="twitter:description"),
    )

    ld_name, ld_desc = None, None
    try:
        jsonld = extruct.extract(html, base_url=get_base_url(html, url), syntaxes=["json-ld"]).get("json-ld", [])
        for block in jsonld:
            if isinstance(block, dict):
                # direct
                if block.get("description") or block.get("name"):
                    ld_desc = block.get("description") or ld_desc
                    ld_name = block.get("name") or ld_name
                    if ld_desc or ld_name:
                        break
                # graph
                if "@graph" in block and isinstance(block["@graph"], list):
                    for node in block["@graph"]:
                        if isinstance(node, dict) and (node.get("description") or node.get("name")):
                            ld_desc = node.get("description") or ld_desc
                            ld_name = node.get("name") or ld_name
                            break
    except Exception:
        pass

    picked_title = first_nonempty(og_title, tw_title, ld_name, title)
    picked_desc = first_nonempty(description, ld_desc)
    return picked_title, picked_desc

def extract_main_text(html: str, url: str) -> Optional[str]:
    try:
        text = trafilatura.extract(html, include_comments=False, include_tables=False, url=url)
        if not text:
            return None
        # compact to 1–3 sentences
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        snippet = " ".join(sents[:3]).strip()
        return snippet if snippet else None
    except Exception:
        return None

async def fetch_one(session: aiohttp.ClientSession, url: str, timeout: int) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    try:
        async with session.get(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": USER_AGENT}) as resp:
            ct = resp.headers.get("Content-Type", "")
            status = resp.status
            if resp.status >= 400:
                return None, status, f"http error {resp.status}"
            if "text/html" not in ct.lower():
                return None, status, f"non-html content-type: {ct}"
            # Limit the read to MAX_HTML_BYTES
            content = await resp.content.readexactly(MAX_HTML_BYTES) if resp.content_length and resp.content_length > MAX_HTML_BYTES else await resp.content.read()
            return content.decode(errors="ignore"), status, None
    except asyncio.IncompleteReadError as e:
        # Partial reads are fine—use what we got
        try:
            data = e.partial.decode(errors="ignore")
            return data, None, "partial-read"
        except Exception:
            return None, None, "decode-failed"
    except Exception as e:
        return None, None, str(e)

async def summarize_domain(session: aiohttp.ClientSession, rec: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    domain = (rec.get("domain") or "").strip()
    name = rec.get("name")
    urls = candidate_urls(domain)
    fetched_url = None
    html = None
    status = None
    err = None

    for u in urls:
        html, status, err = await fetch_one(session, u, timeout=timeout)
        if html:
            fetched_url = u
            break

    result = {
        "domain": domain,
        "name": name,
        "picked_url": fetched_url,
        "http_status": status,
        "error": err,
        "title": None,
        "summary": None,
        "source": None,
        "meta": {
            "industry": rec.get("industry"),
            "country": rec.get("country"),
            "year_founded": rec.get("year founded") or rec.get("year_founded"),
            "date": rec.get("date"),
            "employee_estimate": to_int_safe(rec.get("total employee estimate")),
        },
    }

    if not html:
        return result

    title, meta_desc = extract_meta_and_ld(html, fetched_url)
    if meta_desc and len(meta_desc) >= MIN_DESC_LEN:
        result["title"] = title
        result["summary"] = meta_desc
        result["source"] = "metadata"
        return result

    snippet = extract_main_text(html, fetched_url)
    if snippet:
        result["title"] = title
        result["summary"] = snippet
        result["source"] = "main_content"
        return result

    result["title"] = title
    result["summary"] = None
    result["source"] = "empty"
    return result

def topk_streaming(dataset_name: str, split: str, k: int) -> List[Dict[str, Any]]:
    """
    Stream the dataset and keep a min-heap of top-k by 'total employee estimate'.
    Returns the selected records as a list (largest first).
    """
    ds = load_dataset(dataset_name, split=split, streaming=True)
    heap: List[Tuple[int, int, Dict[str, Any]]] = []
    counter = itertools.count()
    for rec in tqdm(ds):
        emp = to_int_safe(rec.get("total employee estimate"))
        idx = next(counter)
        if len(heap) < k:
            heapq.heappush(heap, (emp, idx, rec))
        else:
            if emp > heap[0][0]:
                heapq.heapreplace(heap, (emp, idx, rec))
    # largest first
    heap.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for (_, _, r) in heap]

async def process_all(selected: List[Dict[str, Any]], out_path: str, concurrency: int, timeout: int):
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()

    async with aiohttp.ClientSession() as session, open(out_path, "w", encoding="utf-8") as f:
        async def bound(rec):
            async with sem:
                return await summarize_domain(session, rec, timeout)

        # Use tqdm over async tasks
        tasks = [asyncio.create_task(bound(rec)) for rec in selected]
        async for done in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Fetching+Summarizing"):
            res = await done
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
    dt = time.time() - t0
    print(f"Done. Wrote {len(selected)} rows to {out_path} in {dt:.1f}s.")

def main():
    parser = argparse.ArgumentParser(description="Summarize top websites by employee estimate from Hugging Face dataset.")
    parser.add_argument("--dataset", default="Plugiloinc/45_Million_Websites", help="HF dataset path")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--topk", type=int, default=10_000, help="Number of top rows to summarize")
    parser.add_argument("--out", default="website_summaries_top10k.jsonl", help="Output JSONL path")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent fetches")
    parser.add_argument("--timeout", type=int, default=12, help="Per-request timeout (seconds)")
    args = parser.parse_args()

    print(f"Streaming dataset {args.dataset} split={args.split} to select top {args.topk} by 'total employee estimate'…")
    selected = topk_streaming(args.dataset, args.split, args.topk)
    print(f"Selected {len(selected)} records. Starting fetch/summarize with concurrency={args.concurrency}…")

    asyncio.run(process_all(selected, args.out, args.concurrency, args.timeout))

if __name__ == "__main__":
    main()
