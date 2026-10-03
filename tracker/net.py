"""HTTPS session for Yahoo Finance that also trusts the operating system's certificates.

Antivirus suites (Norton, Kaspersky, Avast...) and corporate proxies inspect HTTPS
traffic by re-signing it with their own root certificate. That root lives in the
Windows certificate store but not in certifi's bundle, which curl_cffi (the HTTP
client yfinance uses) relies on - so every request fails with
"unable to get local issuer certificate". We merge both sources into one PEM file
and hand it to the session.
"""
from __future__ import annotations

import ssl
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

_session = None
_session_lock = threading.Lock()

# yfinance keeps a process-wide singleton that holds the HTTP session, and curl handles
# crash if two threads drive them at once - so every Yahoo call must hold this lock.
YAHOO_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def ca_bundle_path() -> str | None:
    """Write certifi's CA bundle plus the OS root store to a temp file and return its path."""
    try:
        import certifi
    except ImportError:
        return None

    parts = [Path(certifi.where()).read_text(encoding="utf-8")]
    if sys.platform == "win32":
        for cert, encoding, _trust in ssl.enum_certificates("ROOT"):
            if encoding == "x509_asn":
                parts.append(ssl.DER_cert_to_PEM_cert(cert))

    path = Path(tempfile.gettempdir()) / "market_tracker_ca_bundle.pem"
    path.write_text("\n".join(parts), encoding="utf-8")
    return str(path)


def get_session():
    """The shared curl_cffi session. Only use it while holding YAHOO_LOCK."""
    global _session
    with _session_lock:
        if _session is None:
            from curl_cffi import requests as curl_requests

            _session = curl_requests.Session(impersonate="chrome", verify=ca_bundle_path() or True)
        return _session


_thread_local = threading.local()


def thread_session():
    """A private session for the calling thread, for parallel downloads that bypass yfinance
    (yfinance shares one session process-wide, so it must stay behind YAHOO_LOCK)."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        from curl_cffi import requests as curl_requests

        session = curl_requests.Session(impersonate="chrome", verify=ca_bundle_path() or True)
        _thread_local.session = session
    return session
