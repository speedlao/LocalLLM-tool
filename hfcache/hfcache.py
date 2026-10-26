#!/usr/bin/env python3
"""
hfcache -- a self-built HuggingFace Hub model downloader
========================================================
Manages download tasks precisely by (repo_id, commit). On-disk layout is fully
compatible with the official ~/.cache/huggingface/hub conventions
(refs / blobs / snapshots / trees), plus a content-addressed store (cas/)
and a SQLite task state machine.

See DESIGN.md and _hf_research/ in this directory for the measured rationale.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, CancelledError
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import (
    HTTPDefaultErrorHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    ProxyHandler,
    build_opener,
)

# ----------------------------- constants -----------------------------
VERSION = "1.0.0"
DEFAULT_ENDPOINT = "https://huggingface.co"
SEP = "--"
REGEX_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
REGEX_SHA256 = re.compile(r"[0-9a-f]{64}")
HDR_REPO_COMMIT = "X-Repo-Commit"
HDR_LINKED_ETAG = "X-Linked-Etag"
HDR_LINKED_SIZE = "X-Linked-Size"
UA = f"hfcache/{VERSION} (python-urllib; self-built hub downloader)"

TIER = "models"          # directory prefix: models--<ns>--<name>
TREE_FORMAT_VERSION = 1
CACHEDIR_TAG = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by hfcache.\n"
    "# For information about cache directory tags, see\n"
    "#\thttps://bford.info/cachedir/spec.html\n"
)


# ----------------------------- helpers -----------------------------
def norm_etag(etag):
    """Strip the W/ prefix and quotes from an ETag so it is usable as a filename."""
    if etag is None:
        return None
    return etag.lstrip("W/").strip('"')


def git_blob_oid(size, data):
    """git blob oid = sha1(b"blob <size>\\0" + data); the official etag for non-LFS files."""
    return hashlib.sha1(b"blob %d\0" % size + data).hexdigest()


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(xlen(p), "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def xlen(p):
    """On Windows, promote long paths to the extended-length form to dodge MAX_PATH."""
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def atomic_write(path, text):
    """Write via a temp file + os.replace so concurrent readers never see a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with open(xlen(tmp), "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(xlen(tmp), xlen(path))


# OS error strings are localized; map them to English so every message stays English.
_WINERROR = {
    2: "path not found",
    3: "path not found (check the cache directory)",
    5: "access denied by the file system",
    13: "permission denied",
    36: "path too long",
    121: "operation timed out",
    10013: "connection forbidden by firewall or socket policy",
    10051: "network is unreachable",
    10054: "connection reset by the peer",
    10060: "connection timed out",
    10061: "connection refused (is the proxy running?)",
    11001: "host name not found (check DNS or proxy)",
}


def _osmsg(exc):
    code = getattr(exc, "winerror", None) or getattr(exc, "errno", None)
    if code in _WINERROR:
        return _WINERROR[code]
    if code is not None:
        return f"os error {code}"
    return str(exc) or exc.__class__.__name__


def describe_exc(exc):
    """Render any exception as a single-line English description."""
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code} {exc.reason} for {exc.url}"
    if isinstance(exc, Redirect):
        return f"redirected to {exc.url}"
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, OSError):
            return f"network error: {_osmsg(reason)}"
        return f"network error: {reason}"
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return f"io error: {_osmsg(exc)}"
    return str(exc).strip() or exc.__class__.__name__


# ------------------------- HTTP layer -------------------------
class _NoRedirect(HTTPRedirectHandler):
    """Turn 3xx into an exception so the metadata headers of the 302 hop stay readable."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Redirect(code, newurl, dict(headers))


class Redirect(Exception):
    def __init__(self, code, url, headers):
        super().__init__(f"HTTP {code} -> {url}")
        self.code = code
        self.url = url
        self.headers = headers


class Http:
    """Minimal HTTP client: optional proxy, explicit redirect control, optional Bearer auth."""

    def __init__(self, proxy=None, token=None):
        self.proxy = proxy
        self.token = token
        self._local = threading.local()

    def _opener(self):
        op = getattr(self._local, "opener", None)
        if op is None:
            handlers = [HTTPSHandler()]
            if self.proxy:
                handlers.append(ProxyHandler({"http": self.proxy, "https": self.proxy}))
            handlers.append(_NoRedirect())
            handlers.append(HTTPDefaultErrorHandler())
            op = build_opener(*handlers)
            self._local.opener = op
        return op

    def headers(self):
        h = {"User-Agent": UA, "Accept-Encoding": "identity"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def open(self, req, timeout=60):
        return self._opener().open(req, timeout=timeout)

    def get_json(self, url, timeout=60):
        with self.open(Request(url, headers=self.headers()), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    @staticmethod
    def _next(url, exc):
        if isinstance(exc, Redirect):
            return exc.url, exc.headers
        if isinstance(exc, HTTPError) and exc.code in (301, 302, 303, 307, 308):
            return urljoin(url, exc.headers.get("Location", "")), dict(exc.headers)
        return None, None

    def head(self, url, timeout=60):
        """
        Issue HEAD while following redirects, but return the headers of the *first* hop
        that actually carries metadata.

        This is the single most important rule in this tool: the metadata lives only on
        the 302 from /resolve. Once you follow through to the CDN, the etag degrades into
        the xetHash and X-Repo-Commit disappears entirely.
        """
        meta = None
        for _ in range(10):
            try:
                with self.open(Request(url, method="HEAD", headers=self.headers()), timeout=timeout) as r:
                    h = dict(r.headers)
                    if meta is None:
                        meta = h
                    return meta, str(r.url)
            except (Redirect, HTTPError) as e:
                nxt, h = self._next(url, e)
                if nxt is None:
                    raise
                if meta is None and (HDR_LINKED_ETAG in h or HDR_REPO_COMMIT in h):
                    meta = h
                url = nxt
        raise Retry(f"too many redirects: {url}")

    def get_stream(self, url, range_start=0, timeout=0):
        """GET a stream; when range_start>0 send Range to resume. Redirects handled manually."""
        hdrs = self.headers()
        if range_start:
            hdrs["Range"] = f"bytes={range_start}-"
        for _ in range(10):
            try:
                return self.open(Request(url, headers=hdrs), timeout=timeout or 15 * 60)
            except (Redirect, HTTPError) as e:
                nxt, _h = self._next(url, e)
                if nxt is None:
                    raise
                url = nxt
        raise Retry(f"too many redirects: {url}")


class Retry(Exception):
    """Retryable error (429 / 5xx / timeout / broken stream)."""


class Fatal(Exception):
    """Non-retryable error (401 / 403 / 404 / 409 ...)."""


# ------------------------- path layout -------------------------
class Layout:
    """Maps the cache root to per-repo subdirectories, strictly following the official layout."""

    def __init__(self, root):
        self.root = Path(root)

    @property
    def cas(self):
        return self.root / "cas"

    @property
    def tool(self):
        return self.root / ".tool"

    def repo_dir(self, repo_id):
        return self.root / SEP.join([TIER, *repo_id.split("/")])

    def refs(self, repo_id):
        return self.repo_dir(repo_id) / "refs"

    def blobs(self, repo_id):
        return self.repo_dir(repo_id) / "blobs"

    def snapshots(self, repo_id):
        return self.repo_dir(repo_id) / "snapshots"

    def trees(self, repo_id):
        return self.repo_dir(repo_id) / "trees"

    def no_exist(self, repo_id):
        return self.repo_dir(repo_id) / ".no_exist"

    def cas_path(self, sha):
        return self.cas / sha[:2] / sha


# ------------------------- SQLite state machine -------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(
    id TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    path_in_repo TEXT NOT NULL,
    etag TEXT,
    kind TEXT,
    content_sha256 TEXT,
    size INTEGER,
    state TEXT NOT NULL DEFAULT 'queued',
    attempt INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tasks_lookup ON tasks(repo_id, commit_sha, path_in_repo);
"""


class Journal:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(xlen(path), isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def upsert(self, t):
        with self._lock:
            self.db.execute(
                "INSERT INTO tasks(id,repo_id,commit_sha,path_in_repo,etag,kind,content_sha256,size,"
                "state,attempt,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET etag=excluded.etag, kind=excluded.kind, "
                "content_sha256=excluded.content_sha256, size=excluded.size, "
                "state=excluded.state, updated_at=excluded.updated_at",
                (t.key, t.repo_id, t.commit_sha, t.path_in_repo, t.etag, t.kind, t.content_sha256,
                 t.size, t.state, t.attempt, t.created_at, time.time()),
            )

    def set_state(self, key, state, error=None, attempt_inc=False):
        with self._lock:
            if attempt_inc:
                self.db.execute("UPDATE tasks SET state=?,error=?,attempt=attempt+1,updated_at=? WHERE id=?",
                                (state, error, time.time(), key))
            else:
                self.db.execute("UPDATE tasks SET state=?,error=?,updated_at=? WHERE id=?",
                                (state, error, time.time(), key))

    def find(self, repo_id, commit_sha):
        cur = self.db.execute("SELECT * FROM tasks WHERE repo_id=? AND commit_sha=?", (repo_id, commit_sha))
        return [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]


# ------------------------- task model -------------------------
@dataclass
class Task:
    repo_id: str
    commit_sha: str
    path_in_repo: str
    etag: str = None
    kind: str = None              # "git" | "lfs"
    content_sha256: str = None
    size: int = None
    state: str = "queued"
    attempt: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def key(self):
        return f"{self.repo_id}@{self.commit_sha}:{self.path_in_repo}"


# ------------------------- Hub API -------------------------
class Hub:
    def __init__(self, http, endpoint):
        self.http = http
        self.ep = endpoint.rstrip("/")

    def _api(self, repo_id, suffix):
        return f"{self.ep}/api/{TIER}/{quote(repo_id, safe='/')}{suffix}"

    def resolve_commit(self, repo_id, revision):
        """Resolve a branch/tag to a 40-char commit hash. A full hash costs zero requests."""
        if REGEX_COMMIT_HASH.fullmatch(revision):
            return revision
        info = self.http.get_json(self._api(repo_id, ""))
        sha = info.get("sha")
        if not sha:
            raise Fatal(f"could not resolve commit sha from the API response (revision={revision})")
        return sha

    def tree(self, repo_id, commit):
        """Return {path: entry} with fields aligned to the official trees/<sha>.json."""
        raw = self.http.get_json(self._api(repo_id, f"/tree/{quote(commit, safe='')}?recursive=true&expand=true"))
        out = {}
        for e in raw:
            if e.get("type") != "file":
                continue
            lfs = e.get("lfs") or {}
            out[e["path"]] = {
                "size": e.get("size"),
                "blob_id": e.get("oid"),
                "lfs_sha256": lfs.get("oid"),
                "lfs_size": lfs.get("size"),
                "xet_hash": e.get("xetHash"),
            }
        return out

    def resolve_url(self, repo_id, commit, path):
        """One HEAD; returns the headers of the metadata hop plus the real download URL."""
        url = f"{self.ep}/{quote(repo_id, safe='/')}/resolve/{quote(commit, safe='')}/{quote(path, safe='/')}"
        try:
            hdrs, _ = self.http.head(url)
            return url, hdrs
        except Redirect as r:
            return r.url, r.headers

    def probe(self, repo_id, commit, path):
        """Return (etag, real URL, size). etag comes from the 302 hop, never from the CDN."""
        url, hdrs = self.resolve_url(repo_id, commit, path)
        etag = norm_etag(hdrs.get(HDR_LINKED_ETAG) or hdrs.get("ETag"))
        if etag is None:
            raise Fatal(f"no usable etag header in the response: {url}")
        size = hdrs.get(HDR_LINKED_SIZE) or hdrs.get("Content-Length")
        return etag, url, (int(size) if size else None)


# ------------------------- on-disk engine -------------------------
class Engine:
    def __init__(self, layout, hub, journal, jobs, max_retry, breaker_threshold, quiet=False):
        self.L = layout
        self.hub = hub
        self.J = journal
        self.jobs = jobs
        self.max_retry = max_retry
        self.breaker = breaker_threshold
        self.quiet = quiet
        self._fail_streak = 0
        self._streak_lock = threading.Lock()
        self._downloaded = set()
        self._dl_lock = threading.Lock()
        self._caps = self._probe_caps()

    # ---------- capability probe ----------
    def _probe_caps(self):
        out = {"symlink": False, "hardlink": False}
        try:
            probe_dir = self.L.tool / "probe"
            probe_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return out
        f = probe_dir / "a.bin"
        f.write_bytes(b"x")
        try:
            os.symlink(xlen(f), xlen(probe_dir / "lnk"))
            out["symlink"] = True
        except OSError:
            pass
        try:
            os.link(xlen(f), xlen(probe_dir / "hlnk"))
            out["hardlink"] = os.stat(xlen(f)).st_ino == os.stat(xlen(probe_dir / "hlnk")).st_ino
        except OSError:
            pass
        for p in probe_dir.iterdir():
            try:
                os.remove(xlen(p))
            except OSError:
                pass
        return out

    def log(self, msg):
        if not self.quiet:
            print(msg, file=sys.stderr, flush=True)

    def ensure_root(self):
        self.L.root.mkdir(parents=True, exist_ok=True)
        tag = self.L.root / "CACHEDIR.TAG"
        if not tag.exists():
            with open(xlen(tag), "w", encoding="utf-8") as f:
                f.write(CACHEDIR_TAG)
        self.L.tool.mkdir(parents=True, exist_ok=True)
        self.L.cas.mkdir(parents=True, exist_ok=True)

    # ---------- trees / refs / no_exist ----------
    def read_tree(self, repo_id, commit):
        f = self.L.trees(repo_id) / f"{commit}.json"
        if not os.path.exists(xlen(f)):
            return None
        try:
            with open(xlen(f), encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as e:
            self.log(f"  ! trees cache is corrupt, ignoring: {f.name}: {e}")
            return None
        if data.get("format_version") != TREE_FORMAT_VERSION:
            return None
        return data["files"]

    def write_tree(self, repo_id, commit, files):
        f = self.L.trees(repo_id) / f"{commit}.json"
        atomic_write(f, json.dumps({"format_version": TREE_FORMAT_VERSION,
                                    "files": {k: files[k] for k in sorted(files)}},
                                   indent=1, ensure_ascii=False))

    def write_ref(self, repo_id, revision, commit):
        if revision == commit:
            return
        atomic_write(self.L.refs(repo_id) / revision, commit)

    def read_ref(self, repo_id, revision):
        f = self.L.refs(repo_id) / revision
        try:
            return f.read_text(encoding="utf-8").strip() if f.is_file() else None
        except OSError:
            return None

    def snapshot_has(self, repo_id, commit, path):
        p = self.L.snapshots(repo_id) / commit / path
        try:
            st = os.stat(xlen(p))
            return st.st_size > 0
        except OSError:
            return False

    # ---------- link layer: prefer hardlink, fall back to copy ----------
    def link_or_copy(self, src, dst):
        src, dst = Path(src), Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(xlen(dst)) or os.path.islink(xlen(dst)):
            try:
                os.remove(xlen(dst))
            except OSError:
                pass
        try:
            os.link(xlen(src), xlen(dst))
            return "hardlink"
        except OSError:
            shutil.copy2(xlen(src), xlen(dst))
            return "copy"

    def cas_put(self, sha, src_tmp):
        dst = self.L.cas_path(sha)
        if os.path.exists(xlen(dst)):
            os.remove(xlen(src_tmp))
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(xlen(src_tmp), xlen(dst))
        return dst

    # ---------- planning ----------
    def resolve(self, repo_id, revision):
        """Resolve a commit; on network failure fall back to the local refs/<revision>."""
        try:
            return self.hub.resolve_commit(repo_id, revision)
        except Exception as e:  # offline / 5xx / private repo without a valid token
            ref = self.read_ref(repo_id, revision)
            if ref:
                self.log(f"  ! cannot reach the Hub ({describe_exc(e)}); falling back to local refs/{revision} = {ref}")
                return ref
            raise

    def plan(self, repo_id, revision, includes, excludes):
        commit = self.resolve(repo_id, revision)
        files = self.read_tree(repo_id, commit)
        if files is None:
            files = self.hub.tree(repo_id, commit)
            self.write_tree(repo_id, commit, files)
        tasks = []
        for path, entry in sorted(files.items()):
            if includes and not any(fnmatch.fnmatch(path, g) for g in includes):
                continue
            if excludes and any(fnmatch.fnmatch(path, g) for g in excludes):
                continue
            is_lfs = bool(entry.get("lfs_sha256"))
            t = Task(repo_id, commit, path,
                     etag=entry["lfs_sha256"] if is_lfs else entry.get("blob_id"),
                     kind="lfs" if is_lfs else "git",
                     content_sha256=entry.get("lfs_sha256") if is_lfs else None,
                     size=entry.get("size"))
            t.state = "cached" if self.snapshot_has(repo_id, commit, path) else "queued"
            self.J.upsert(t)
            tasks.append(t)
        return commit, tasks, files

    # ---------- single-file sync ----------
    def sync_one(self, t, entry, dry):
        try:
            return self._sync_one(t, entry, dry)
        except Retry as e:
            self._note_fail()
            self.J.set_state(t.key, "failed", f"retryable: {e}", attempt_inc=True)
            return f"RETRY-FAIL {t.path_in_repo}: {describe_exc(e)}"
        except (Fatal, HTTPError, URLError, OSError) as e:
            self._note_fail()
            self.J.set_state(t.key, "failed", str(e), attempt_inc=True)
            return f"FAIL {t.path_in_repo}: {describe_exc(e)}"

    def _note_fail(self):
        with self._streak_lock:
            self._fail_streak += 1

    def _sync_one(self, t, entry, dry):
        if self.snapshot_has(t.repo_id, t.commit_sha, t.path_in_repo):
            self.J.set_state(t.key, "cached")
            return f"CACHED {t.path_in_repo}"

        etag = size = kind = None
        if entry:
            size = entry.get("size")
            if entry.get("lfs_sha256"):
                etag, kind = entry["lfs_sha256"], "lfs"
            else:
                etag, kind = entry.get("blob_id"), "git"
        if not etag:
            etag, _u, hsize = self.hub.probe(t.repo_id, t.commit_sha, t.path_in_repo)
            size = size or hsize
            kind = "lfs" if REGEX_SHA256.fullmatch(etag) else "git"
        t.etag, t.kind, t.size = etag, kind, size

        blob = self.L.blobs(t.repo_id) / etag
        if dry:
            return f"DRY  {t.path_in_repo} size={size} kind={kind} blob={etag[:12]}"

        need = not os.path.exists(xlen(blob))
        if need:
            with self._dl_lock:
                if blob in self._downloaded or os.path.exists(xlen(blob)):
                    need = False
                else:
                    self._download_to_cas(t, blob)
                    self._downloaded.add(blob)

        cas_file = self.L.cas_path(t.content_sha256) if t.content_sha256 else None
        m1 = self.link_or_copy(cas_file, blob) if (need and cas_file and cas_file.exists()) else "exists"
        snap = self.L.snapshots(t.repo_id) / t.commit_sha / t.path_in_repo
        m2 = self.link_or_copy(blob, snap)

        self.J.set_state(t.key, "materialized")
        with self._streak_lock:
            self._fail_streak = 0
        return f"OK   {t.path_in_repo} [{size}B {kind}] blob<-{m1} snap<-{m2}"

    def _download_to_cas(self, t, blob):
        """Download into a cas temp part file, resume via Range, then verify and commit to cas."""
        url = (f"{self.hub.ep}/{quote(t.repo_id, safe='/')}/resolve/"
               f"{quote(t.commit_sha, safe='')}/{quote(t.path_in_repo, safe='/')}")
        tmp = self.L.cas / f".{uuid.uuid4().hex}.part"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        start = 0
        last_err = None

        for attempt in range(1, self.max_retry + 1):
            self.J.set_state(t.key, "fetching" if attempt == 1 else "retrying")
            try:
                with self.hub.http.get_stream(url, start) as r:
                    status = getattr(r, "status", 200)
                    if start and status != 206:
                        start = 0
                    mode = "ab" if start else "wb"
                    with open(xlen(tmp), mode) as f:
                        while True:
                            try:
                                chunk = r.read(1 << 20)
                            except Exception as e:
                                raise Retry(f"stream interrupted at {start} bytes: {describe_exc(e)}") from None
                            if not chunk:
                                break
                            f.write(chunk)
                            start += len(chunk)
                last_err = None
                break
            except (HTTPError, URLError, TimeoutError, OSError, Retry) as e:
                if isinstance(e, HTTPError) and e.code in (401, 403, 404, 409, 422):
                    raise Fatal(f"HTTP {e.code}: {url}") from None
                last_err = e
                if attempt >= self.max_retry:
                    break
                wait = min(2 ** attempt, 30)
                have = os.stat(xlen(tmp)).st_size if os.path.exists(xlen(tmp)) else 0
                self.log(f"  ~ {t.path_in_repo} attempt {attempt} failed ({describe_exc(e)}); resuming from "
                         f"{have} bytes in {wait}s")
                self.J.set_state(t.key, "retrying", str(e), attempt_inc=True)
                start = have
                time.sleep(wait)

        if last_err is not None:
            raise Retry(f"still failing after {self.max_retry} attempts: {describe_exc(last_err)}")
        if not os.path.exists(xlen(tmp)):
            raise Retry("the download produced no data")

        got = os.stat(xlen(tmp)).st_size
        if t.size and got != t.size:
            raise Retry(f"size mismatch: expected {t.size}, got {got}")

        sha = sha256_file(tmp)
        if t.kind == "git":
            with open(xlen(tmp), "rb") as f:
                data = f.read()
            real = git_blob_oid(len(data), data)
            if real != t.etag:
                os.remove(xlen(tmp))
                raise Fatal(f"git blob oid verification failed: expected {t.etag}, got {real}")
        else:
            if sha != t.etag:
                os.remove(xlen(tmp))
                raise Fatal(f"sha256 verification failed: expected {t.etag}, got {sha}")

        self.cas_put(sha, tmp)
        t.content_sha256 = sha
        self.J.set_state(t.key, "verified")

    # ---------- batch sync ----------
    def sync(self, commit, tasks, files, dry, revision="main"):
        todo = [t for t in tasks if t.state != "cached"]
        self.log(f"planned {len(tasks)} file(s), {len(todo)} to download "
                 f"(concurrency {self.jobs}, per-file retries {self.max_retry}, "
                 f"breaker threshold {self.breaker})")
        if tasks:
            self.write_ref(tasks[0].repo_id, revision, commit)
        if not todo:
            return 0
        entry_map = {t.path_in_repo: files.get(t.path_in_repo) for t in todo}
        rc = 0
        stopped = False
        with ThreadPoolExecutor(max_workers=self.jobs) as ex:
            futures = {ex.submit(self.sync_one, t, entry_map.get(t.path_in_repo), dry): t for t in todo}
            for fut in list(futures):
                t = futures[fut]
                try:
                    msg = fut.result()
                except CancelledError:
                    msg = f"SKIP {t.path_in_repo} (cancelled by circuit breaker)"
                except Exception as e:
                    msg = f"FAIL {t.path_in_repo}: {describe_exc(e)}"
                print(msg)
                if msg.startswith(("FAIL", "RETRY-FAIL")):
                    rc += 1
                with self._streak_lock:
                    streak = self._fail_streak
                if streak >= self.breaker and not stopped:
                    print(f"!! {streak} consecutive failures -- circuit breaker tripped, "
                          f"no further tasks", file=sys.stderr)
                    for f2 in futures:
                        f2.cancel()
                    stopped = True
        return rc


# ------------------------- CLI -------------------------
def _repo_args(sp):
    sp.add_argument("repo_id")
    sp.add_argument("--rev", default="main")
    sp.add_argument("--inc", action="append", default=[], help="include glob, repeatable")
    sp.add_argument("--exc", action="append", default=[], help="exclude glob, repeatable")


def build_parser():
    p = argparse.ArgumentParser(prog="hfcache", description="Self-built HuggingFace Hub model downloader")
    p.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE")
                   or str(Path.home() / ".cache" / "huggingface" / "hub"))
    p.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    p.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"))
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--jobs", type=int, default=int(os.environ.get("HFCACHE_JOBS", 8)))
    p.add_argument("--max-retry", type=int, default=int(os.environ.get("HFCACHE_MAX_RETRY", 5)))
    p.add_argument("--breaker", type=int, default=int(os.environ.get("HFCACHE_BREAKER", 5)))
    p.add_argument("--quiet", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("resolve", help="resolve a revision to its 40-char commit hash")
    sp.add_argument("repo_id")
    sp.add_argument("--rev", default="main")

    sp = sub.add_parser("plan", help="list the files a commit needs and their state")
    _repo_args(sp)
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("sync", help="download and materialize the plan")
    _repo_args(sp)
    sp.add_argument("--dry-run", action="store_true")

    _repo_args(sub.add_parser("status", help="completeness check, offline first"))
    _repo_args(sub.add_parser("verify", help="recompute hashes for every file"))

    sp = sub.add_parser("prune", help="remove leftover part files and unreferenced cas objects")
    sp.add_argument("--dry-run", action="store_true")
    sub.add_parser("doctor", help="print environment and link capabilities")
    return p


def _plan_print(eng, args, as_json):
    commit, tasks, _ = eng.plan(args.repo_id, args.rev, args.inc, args.exc)
    if as_json:
        print(json.dumps({"repo_id": args.repo_id, "revision": args.rev, "commit": commit,
                          "files": [{"path": t.path_in_repo, "size": t.size, "kind": t.kind,
                                     "blob": t.etag, "state": t.state} for t in tasks]},
                         ensure_ascii=False, indent=2))
        return 0
    queued = [t for t in tasks if t.state == "queued"]
    print(f"repo     : {args.repo_id}")
    print(f"revision : {args.rev}")
    print(f"commit   : {commit}")
    print(f"files    : {len(tasks)} (to download {len(queued)}, "
          f"to transfer {sum(t.size or 0 for t in queued)/1e6:.1f} MB)")
    for t in tasks:
        print(f"  {'v' if t.state == 'cached' else '-'} {t.path_in_repo:60s} {(t.size or 0):>12}  {t.kind}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    layout = Layout(Path(args.cache_dir).expanduser())
    layout.root.mkdir(parents=True, exist_ok=True)
    journal = Journal(layout.tool / "journal.sqlite3")
    hub = Hub(Http(proxy=args.proxy, token=args.token), args.endpoint)
    eng = Engine(layout, hub, journal, args.jobs, args.max_retry, args.breaker, args.quiet)

    if args.cmd == "doctor":
        print(json.dumps({
            "version": VERSION, "python": sys.version.split()[0], "platform": sys.platform,
            "cache_dir": str(layout.root.resolve()), "endpoint": args.endpoint,
            "proxy": args.proxy or "(direct)", "link_capability": eng._caps,
            "note": "symlinks unavailable, hardlink scheme unaffected" if not eng._caps["symlink"] else "ok",
        }, ensure_ascii=False, indent=2))
        return 0

    eng.ensure_root()

    if args.cmd == "resolve":
        print(eng.hub.resolve_commit(args.repo_id, args.rev))
        return 0

    if args.cmd in ("plan", "status"):
        return _plan_print(eng, args, getattr(args, "json", False))

    if args.cmd == "sync":
        commit, tasks, files = eng.plan(args.repo_id, args.rev, args.inc, args.exc)
        rc = eng.sync(commit, tasks, files, args.dry_run, args.rev)
        print(f"\ndone: {rc} file(s) failed, cache at {layout.root.resolve()}")
        return 1 if rc else 0

    if args.cmd == "verify":
        commit = eng.resolve(args.repo_id, args.rev)
        files = eng.read_tree(args.repo_id, commit) or eng.hub.tree(args.repo_id, commit)
        eng.write_tree(args.repo_id, commit, files)
        ok = miss = bad = 0
        for p, e in sorted(files.items()):
            if args.inc and not any(fnmatch.fnmatch(p, g) for g in args.inc):
                continue
            if args.exc and any(fnmatch.fnmatch(p, g) for g in args.exc):
                continue
            snap = eng.L.snapshots(args.repo_id) / commit / p
            if not os.path.exists(xlen(snap)):
                miss += 1
                print(f"  missing  {p}")
                continue
            if e.get("lfs_sha256"):
                got, want = sha256_file(snap), e["lfs_sha256"]
            else:
                with open(xlen(snap), "rb") as f:
                    data = f.read()
                got, want = git_blob_oid(len(data), data), e["blob_id"]
            if got == want:
                ok += 1
            else:
                bad += 1
                print(f"  mismatch {p}: expected {want}, got {got}")
        print(f"verification finished: ok={ok} missing={miss} mismatch={bad}")
        return 1 if bad or miss else 0

    if args.cmd == "prune":
        n = freed = 0
        for part in sorted(eng.L.cas.glob(".*.part")) + sorted(eng.L.cas.glob("*/*.part")):
            try:
                if time.time() - os.stat(xlen(part)).st_mtime > 3600:
                    n += 1
                    freed += os.stat(xlen(part)).st_size
                    print(f"{'[dry] ' if args.dry_run else ''}removing stale part {part.name}")
                    if not args.dry_run:
                        os.remove(xlen(part))
            except OSError:
                pass
        for f in sorted(eng.L.cas.glob("*/*")):
            try:
                st = os.stat(xlen(f))
            except OSError:
                continue
            if not f.is_file():
                continue
            if getattr(st, "st_nlink", 1) <= 1:
                n += 1
                freed += st.st_size
                print(f"{'[dry] ' if args.dry_run else ''}removing unreferenced object "
                      f"{f.parent.name}/{f.name[:16]}")
                if not args.dry_run:
                    os.remove(xlen(f))
        print(f"processed {n} item(s), reclaiming {freed/1e6:.1f} MB")
        return 0

    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted by the user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # keep CLI output English; never surface a raw stack trace
        print(f"error: {describe_exc(exc)}", file=sys.stderr)
        sys.exit(1)
