#!/usr/bin/env python3
import argparse, asyncio, json, re, time, itertools
from typing import Optional, Dict, Any, List, Tuple

import aiohttp
from datasets import load_dataset
from bs4 import BeautifulSoup
import extruct
from w3lib.html import get_base_url
import trafilatura
from tqdm.asyncio import tqdm_asyncio

USER_AGENT = "Mozilla/5.0 (compatible; website-summary-bot/0.1; +https://example.com/contact)"
MAX_HTML_BYTES = 1_800_000
MIN_DESC_LEN = 40
DEFAULT_CONCURRENCY = 16

def candidate_urls(domain: str) -> List[str]:
    d = (domain or "").strip()
    d = re.sub(r"^https?://", "", d, flags=re.I).strip("/")
    return [f"https://{d}/", f"http://{d}/", f"https://www.{d}/", f"http://www.{d}/"]

def first_nonempty(*vals) -> Optional[str]:
    for v in vals:
        if isinstance(v, str):
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
            if not isinstance(block, dict):
                continue
            if block.get("description") or block.get("name"):
                ld_desc = ld_desc or block.get("description")
                ld_name = ld_name or block.get("name")
            if "@graph" in block and isinstance(block["@graph"], list):
                for node in block["@graph"]:
                    if isinstance(node, dict) and (node.get("description") or node.get("name")):
                        ld_desc = ld_desc or node.get("description")
                        ld_name = ld_name or node.get("name")
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
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return " ".join(sents[:3]).strip() or None
    except Exception:
        return None

async def fetch_one(session: aiohttp.ClientSession, url: str, timeout: int) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    try:
        async with session.get(url, timeout=timeout, allow_redirects=True, headers={"User-Agent": USER_AGENT}) as resp:
            ct = (resp.headers.get("Content-Type") or "").lower()
            status = resp.status
            if status >= 400:
                return None, status, f"http error {status}"
            if "text/html" not in ct:
                return None, status, f"non-html content-type: {ct}"
            # Cap bytes
            if resp.content_length and resp.content_length > MAX_HTML_BYTES:
                chunk = await resp.content.readexactly(MAX_HTML_BYTES)
            else:
                chunk = await resp.content.read()
            return chunk.decode(errors="ignore"), status, None
    except asyncio.IncompleteReadError as e:
        try:
            return e.partial.decode(errors="ignore"), None, "partial-read"
        except Exception:
            return None, None, "decode-failed"
    except Exception as e:
        return None, None, str(e)

async def summarize_domain(session: aiohttp.ClientSession, rec: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    domain = (rec.get("domain") or "").strip()
    name = rec.get("name")
    urls = candidate_urls(domain)

    fetched_url, html, status, err = None, None, None, None
    for u in urls:
        html, status, err = await fetch_one(session, u, timeout=timeout)
        if html:
            fetched_url = u
            break

    out = {
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
            "employee_estimate": rec.get("total employee estimate"),
        },
    }

    if not html:
        return out

    title, meta_desc = extract_meta_and_ld(html, fetched_url)
    if meta_desc and len(meta_desc) >= MIN_DESC_LEN:
        out.update({"title": title, "summary": meta_desc, "source": "metadata"})
        return out

    snippet = extract_main_text(html, fetched_url)
    if snippet:
        out.update({"title": title, "summary": snippet, "source": "main_content"})
        return out

    out.update({"title": title, "summary": None, "source": "empty"})
    return out

def take_first_10k(dataset_name: str, split: str, limit: int) -> List[Dict[str, Any]]:
    # Prefer streaming to avoid full download
    try:
        ds_stream = load_dataset(dataset_name, split=split, streaming=True)
        return list(itertools.islice(ds_stream, limit))
    except Exception:
        # Fallback: non-streaming then head() if needed
        ds = load_dataset(dataset_name, split=split)
        n = min(limit, len(ds))
        return [ds[i] for i in range(n)]

async def process_all(records, out_path: str, concurrency: int, timeout: int):
    import aiohttp, time, json, asyncio
    from tqdm.asyncio import tqdm_asyncio

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()

    client_timeout = aiohttp.ClientTimeout(total=None, sock_connect=timeout, sock_read=timeout)
    connector = aiohttp.TCPConnector(ttl_dns_cache=300)

    async with aiohttp.ClientSession(timeout=client_timeout, connector=connector, headers={"User-Agent": USER_AGENT}) as session:
        with open(out_path, "w", encoding="utf-8") as f:

            async def bound(rec):
                async with sem:
                    return await summarize_domain(session, rec, timeout)

            tasks = [asyncio.create_task(bound(r)) for r in records]

            # NOTE: normal `for` loop (NOT `async for`)
            for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Fetching+Summarizing"):
                res = await fut
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()

    print(f"Done. Wrote {len(records)} rows to {out_path} in {time.time()-t0:.1f}s.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Plugiloinc/45_Million_Websites")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=10_000)
    p.add_argument("--out", default="website_summaries_first10k.jsonl")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--timeout", type=int, default=12)
    args = p.parse_args()

    print(f"Taking first {args.limit} rows from {args.dataset} ({args.split})…")
    records = take_first_10k(args.dataset, args.split, args.limit)
    print(f"Fetched {len(records)} records. Starting summarization with concurrency={args.concurrency}…")
    asyncio.run(process_all(records, args.out, args.concurrency, args.timeout))

if __name__ == "__main__":
    main()

