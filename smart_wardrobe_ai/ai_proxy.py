"""Smart Wardrobe AI proxy (docs/08 §2, docs/10 §3, ADR-031).

Runs as a Home Assistant add-on on the local network. It keeps the
provider's API key, checks the app's token, enforces a daily quota, and
turns one photo into structured clothing attributes with Gemini (free tier,
ADR-034) or Claude. Photos are never written to disk and nothing personal
is logged.
"""

import base64
import datetime
import hmac
import json
import logging
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic

import analysis

MAX_BODY_BYTES = 8_000_000
MAX_IMAGE_BYTES = 5_000_000
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
IDEMPOTENCY_SECONDS = 3600

# Server-side fallback on a refusal (opt-in beta); Haiku does not take it.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
                    "claude-fable-5-1"}

log = logging.getLogger("ai_proxy")


class ProviderError(Exception):
    def __init__(self, code, message, status, retryable, retry_after=None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


class Analyzer:
    """One Claude call per photo; every attribute at once (docs/08 §5)."""

    def __init__(self, client, model, effort):
        self.client = client
        self.model = model
        self.effort = effort

    def analyze(self, request, image, media_type):
        output_config = {
            "format": {"type": "json_schema",
                       "schema": analysis.build_schema(request)},
        }
        if not self.model.startswith("claude-haiku"):
            output_config["effort"] = self.effort
        extra = {}
        if self.model in _FALLBACK_MODELS:
            extra = {
                "extra_headers": {"anthropic-beta": "server-side-fallback-2026-07-01"},
                "extra_body": {"fallbacks": "default"},
            }
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                system=[{"type": "text", "text": analysis.SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type,
                        "data": base64.standard_b64encode(image).decode("ascii")}},
                    {"type": "text", "text": analysis.build_user_text(request)},
                ]}],
                output_config=output_config,
                **extra,
            )
        except anthropic.RateLimitError as error:
            raise ProviderError("provider_unavailable", "Provider rate limit.", 503,
                                True, _retry_after(error)) from error
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as error:
            log.error("The Anthropic API key is refused: check the add-on options")
            raise ProviderError("provider_unavailable", "Provider refused the key.",
                                502, False) from error
        except anthropic.APITimeoutError as error:
            raise ProviderError("timeout", "Provider timeout.", 504, True) from error
        except anthropic.APIConnectionError as error:
            raise ProviderError("provider_unavailable", "Provider unreachable.",
                                502, True) from error
        except anthropic.APIStatusError as error:
            retryable = error.status_code >= 500
            log.warning("Provider error %s (request %s)", error.status_code,
                        getattr(error, "request_id", None))
            raise ProviderError("provider_unavailable", "Provider error.", 502,
                                retryable) from error

        if response.stop_reason == "refusal":
            raise ProviderError("unsupported", "The photo was not analysed.", 422, False)
        if response.stop_reason == "max_tokens":
            raise ProviderError("provider_unavailable", "Incomplete analysis.", 502, True)
        text = next((block.text for block in response.content if block.type == "text"), None)
        try:
            result = analysis.normalize(analysis.parse_model_text(text or ""), request)
        except ValueError as error:
            raise ProviderError("provider_unavailable", "Unreadable analysis.", 502,
                                True) from error
        result["model"] = response.model
        result["promptVersion"] = analysis.PROMPT_VERSION
        usage = response.usage
        result["usage"] = {
            "inputTokens": usage.input_tokens,
            "outputTokens": usage.output_tokens,
            "cacheReadTokens": getattr(usage, "cache_read_input_tokens", None) or 0,
        }
        return result


def _retry_after(error):
    try:
        return int(error.response.headers.get("retry-after", "60"))
    except (AttributeError, TypeError, ValueError):
        return 60


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
# Gemini stops on these when it will not describe the photo.
_GEMINI_BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY",
                   "RECITATION", "OTHER"}


def _gemini_post(url, key, body, timeout=60.0):
    """One generateContent call; (status, headers, parsed JSON)."""
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.headers, json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read())
        except ValueError:
            payload = {}
        return error.code, error.headers, payload


class GeminiAnalyzer:
    """Same contract as [Analyzer], with the Gemini API (ADR-034): one
    generateContent call per photo, JSON constrained by the same schema.
    Plain HTTP with the standard library, so no new dependency."""

    def __init__(self, key, model, effort, post=_gemini_post):
        self.key = key
        self.model = model
        self.effort = effort
        self.post = post

    def analyze(self, request, image, media_type):
        body = {
            "systemInstruction": {"parts": [{"text": analysis.SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": media_type,
                                "data": base64.standard_b64encode(image).decode("ascii")}},
                {"text": analysis.build_user_text(request)},
            ]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": analysis.build_schema(request),
                "maxOutputTokens": 8000,
                "thinkingConfig": {"thinkingLevel": self.effort},
            },
        }
        try:
            status, headers, payload = self.post(GEMINI_URL.format(self.model), self.key, body)
        except TimeoutError as error:
            raise ProviderError("timeout", "Provider timeout.", 504, True) from error
        except (urllib.error.URLError, OSError) as error:
            raise ProviderError("provider_unavailable", "Provider unreachable.",
                                502, True) from error
        if status == 429:
            try:
                retry_after = int((headers or {}).get("retry-after", "60"))
            except (TypeError, ValueError):
                retry_after = 60
            raise ProviderError("provider_unavailable", "Provider rate limit.", 503,
                                True, retry_after)
        if status in (401, 403) or (status == 400 and "API_KEY" in json.dumps(payload)):
            log.error("The Gemini API key is refused: check the add-on options")
            raise ProviderError("provider_unavailable", "Provider refused the key.",
                                502, False)
        if status != 200:
            log.warning("Provider error %s", status)
            raise ProviderError("provider_unavailable", "Provider error.", 502,
                                status >= 500)

        if (payload.get("promptFeedback") or {}).get("blockReason"):
            raise ProviderError("unsupported", "The photo was not analysed.", 422, False)
        candidate = (payload.get("candidates") or [{}])[0]
        finish = candidate.get("finishReason")
        if finish in _GEMINI_BLOCKED:
            raise ProviderError("unsupported", "The photo was not analysed.", 422, False)
        if finish == "MAX_TOKENS":
            raise ProviderError("provider_unavailable", "Incomplete analysis.", 502, True)
        text = "".join(part.get("text", "")
                       for part in (candidate.get("content") or {}).get("parts") or []
                       if not part.get("thought"))
        try:
            result = analysis.normalize(analysis.parse_model_text(text), request)
        except ValueError as error:
            raise ProviderError("provider_unavailable", "Unreadable analysis.", 502,
                                True) from error
        result["model"] = payload.get("modelVersion") or self.model
        result["promptVersion"] = analysis.PROMPT_VERSION
        usage = payload.get("usageMetadata") or {}
        result["usage"] = {
            "inputTokens": usage.get("promptTokenCount", 0),
            "outputTokens": usage.get("candidatesTokenCount", 0)
            + usage.get("thoughtsTokenCount", 0),
            "cacheReadTokens": usage.get("cachedContentTokenCount", 0),
        }
        return result


class DailyQuota:
    """Analyses allowed per day, kept across restarts in [path]."""

    def __init__(self, limit, path=None, today=datetime.date.today):
        self.limit = limit
        self.path = path
        self.today = today
        self.lock = threading.Lock()
        self.day, self.count = str(today()), 0
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as file:
                    saved = json.load(file)
                if saved.get("day") == self.day:
                    self.count = int(saved.get("count", 0))
            except (OSError, ValueError):
                pass

    def take(self):
        """False when the quota of the day is used up."""
        with self.lock:
            day = str(self.today())
            if day != self.day:
                self.day, self.count = day, 0
            if self.count >= self.limit:
                return False
            self.count += 1
            if self.path:
                try:
                    with open(self.path, "w", encoding="utf-8") as file:
                        json.dump({"day": self.day, "count": self.count}, file)
                except OSError:
                    log.warning("Unable to save the quota")
            return True

    def seconds_to_reset(self):
        now = datetime.datetime.now()
        tomorrow = datetime.datetime.combine(now.date() + datetime.timedelta(days=1),
                                             datetime.time())
        return int((tomorrow - now).total_seconds()) + 1


class IdempotencyCache:
    """A retried request with the same key is answered without a new
    (billed) call; concurrent duplicates wait for the first one."""

    def __init__(self, ttl=IDEMPOTENCY_SECONDS, clock=time.monotonic):
        self.ttl = ttl
        self.clock = clock
        self.lock = threading.Lock()
        self.entries = {}
        self.running = {}

    def run(self, key, compute):
        if not key:
            return compute()
        while True:
            with self.lock:
                now = self.clock()
                for stale in [k for k, (at, _) in self.entries.items()
                              if now - at > self.ttl]:
                    del self.entries[stale]
                if key in self.entries:
                    return self.entries[key][1]
                event = self.running.get(key)
                if event is None:
                    self.running[key] = threading.Event()
                    break
            event.wait(timeout=120)
        try:
            result = compute()
            with self.lock:
                self.entries[key] = (self.clock(), result)
            return result
        finally:
            with self.lock:
                self.running.pop(key).set()


class AuthThrottle:
    """Blocks an address after repeated wrong tokens: the proxy may be
    reachable from the Internet, and every analysis is billed."""

    def __init__(self, limit=10, window=600, clock=time.monotonic):
        self.limit = limit
        self.window = window
        self.clock = clock
        self.lock = threading.Lock()
        self.failures = {}

    def _recent(self, address):
        now = self.clock()
        kept = [at for at in self.failures.get(address, []) if now - at < self.window]
        if kept:
            self.failures[address] = kept
        else:
            self.failures.pop(address, None)
        return kept

    def blocked(self, address):
        with self.lock:
            return len(self._recent(address)) >= self.limit

    def failed(self, address):
        with self.lock:
            self.failures.setdefault(address, []).append(self.clock())
            if len(self._recent(address)) == self.limit:
                log.warning("Too many wrong tokens: an address is blocked for %s s",
                            self.window)

    def succeeded(self, address):
        with self.lock:
            self.failures.pop(address, None)


class Proxy:
    def __init__(self, analyzer, token, quota, cache=None, throttle=None):
        self.analyzer = analyzer
        self.token = token
        self.quota = quota
        self.cache = cache or IdempotencyCache()
        self.throttle = throttle or AuthThrottle()


def _envelope(data=None, error=None, request_id=None):
    return {
        "data": data,
        "meta": {
            "requestId": request_id or str(uuid.uuid4()),
            "schemaVersion": analysis.SCHEMA_VERSION,
            "generatedAt": datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="seconds"),
        },
        "error": error,
    }


def _error(code, message, retryable=False, retry_after=None):
    return {"code": code, "message": message, "retryable": retryable,
            "retryAfterSeconds": retry_after}


def make_handler(proxy):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SmartWardrobeAI/1"

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            log.debug(format, *args)

        def _send(self, status, body, retry_after=None):
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            if retry_after:
                self.send_header("Retry-After", str(retry_after))
            self.end_headers()
            self.wfile.write(payload)

        def _fail(self, status, code, message, retryable=False, retry_after=None,
                  request_id=None):
            self._send(status, _envelope(
                error=_error(code, message, retryable, retry_after),
                request_id=request_id), retry_after)

        def _authorized(self):
            """Answers the refusal itself; true when the request may go on."""
            address = self.client_address[0]
            if proxy.throttle.blocked(address):
                self._fail(429, "unauthorized", "Too many wrong tokens.",
                           retry_after=proxy.throttle.window)
                return False
            header = self.headers.get("Authorization", "")
            given = header[7:] if header.startswith("Bearer ") else ""
            if bool(proxy.token) and hmac.compare_digest(
                    given.encode("utf-8"), proxy.token.encode("utf-8")):
                proxy.throttle.succeeded(address)
                return True
            proxy.throttle.failed(address)
            self._fail(401, "unauthorized", "Invalid token.")
            return False

        def do_GET(self):
            if self.path != "/v1/health":
                return self._fail(404, "unsupported", "Unknown endpoint.")
            if not self._authorized():
                return None
            self._send(200, _envelope({
                "status": "ok",
                "model": proxy.analyzer.model,
                "promptVersion": analysis.PROMPT_VERSION,
                "remainingToday": max(0, proxy.quota.limit - proxy.quota.count),
            }))

        def do_POST(self):
            request_id = str(uuid.uuid4())
            if self.path != "/v1/ai/clothing-analyses":
                return self._fail(404, "unsupported", "Unknown endpoint.")
            if not self._authorized():
                return None
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._fail(413, "validation", "Request too large or empty.")
            body = self.rfile.read(length)
            try:
                request, image, media_type = self._parse(body)
            except analysis.RequestError as error:
                return self._fail(error.status, error.code, str(error))
            key = self.headers.get("Idempotency-Key", "")[:128]

            def compute():
                if not proxy.quota.take():
                    wait = proxy.quota.seconds_to_reset()
                    return 429, _envelope(error=_error(
                        "quota_exceeded", "Daily analysis quota reached.", True, wait),
                        request_id=request_id), wait
                started = time.monotonic()
                try:
                    data = proxy.analyzer.analyze(request, image, media_type)
                except ProviderError as error:
                    return error.status, _envelope(error=_error(
                        error.code, str(error), error.retryable, error.retry_after),
                        request_id=request_id), error.retry_after
                log.info("Analysis %s: %s, %s in / %s out tokens, %.1f s",
                         request_id, data["model"], data["usage"]["inputTokens"],
                         data["usage"]["outputTokens"], time.monotonic() - started)
                return 200, _envelope(data, request_id=request_id), None

            status, envelope, retry_after = proxy.cache.run(key, compute)
            # Failures are not worth replaying: let a retry try again.
            if status != 200 and key:
                with proxy.cache.lock:
                    proxy.cache.entries.pop(key, None)
            self._send(status, envelope, retry_after)

        def _parse(self, body):
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise analysis.RequestError("validation", "Expected multipart data.")
            message = BytesParser(policy=HTTP).parsebytes(
                b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
            parts = {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name in ("request", "image"):
                    parts[name] = (part.get_content_type(), part.get_payload(decode=True))
            if "request" not in parts or "image" not in parts:
                raise analysis.RequestError("validation", "Missing request or image.")
            try:
                payload = json.loads(parts["request"][1].decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as error:
                raise analysis.RequestError("validation", "Invalid request JSON.") from error
            media_type, image = parts["image"]
            if media_type not in IMAGE_TYPES or not image or len(image) > MAX_IMAGE_BYTES:
                raise analysis.RequestError("validation", "Unsupported image.")
            return analysis.parse_request(payload), image, media_type

    return Handler


class TlsServer(ThreadingHTTPServer):
    """HTTPS with the certificate of the Home Assistant (`/ssl`, renewed
    by the DuckDNS add-on): reloaded when the files change, so a renewal
    needs no restart. The handshake runs in the request thread, so a slow
    client never blocks the others."""

    def __init__(self, address, handler, certfile, keyfile):
        self.certfile = certfile
        self.keyfile = keyfile
        self._stamp = None
        self._context = None
        self._context_lock = threading.Lock()
        super().__init__(address, handler)
        self.context()  # Fails now, at start, if the files are unusable.

    def context(self):
        stamp = (os.path.getmtime(self.certfile), os.path.getmtime(self.keyfile))
        with self._context_lock:
            if stamp != self._stamp:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                context.load_cert_chain(self.certfile, self.keyfile)
                self._context, self._stamp = context, stamp
                log.info("TLS certificate loaded")
            return self._context

    def handle_error(self, request, client_address):
        # Scanners and plain HTTP clients fail the handshake: one line, not
        # a stack trace per attempt.
        error = sys.exc_info()[1]
        if isinstance(error, (ssl.SSLError, ConnectionError, TimeoutError)):
            log.debug("TLS connection failed: %s", error)
        else:
            super().handle_error(request, client_address)

    def get_request(self):
        sock, address = super().get_request()
        return self.context().wrap_socket(
            sock, server_side=True, do_handshake_on_connect=False), address


MIN_TOKEN_LENGTH = 24


def load_options(path="/data/options.json"):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def build_analyzer(options):
    """The provider chosen in the add-on options (ADR-034)."""
    effort = options.get("effort", "low")
    if options.get("provider", "gemini") == "gemini":
        if not options.get("gemini_api_key"):
            log.error("Set gemini_api_key in the add-on options")
        return GeminiAnalyzer(options.get("gemini_api_key", ""),
                              options.get("gemini_model", "gemini-3.8-flash"), effort)
    if not options.get("anthropic_api_key"):
        log.error("Set anthropic_api_key in the add-on options")
    client = anthropic.Anthropic(api_key=options.get("anthropic_api_key") or None,
                                 timeout=60.0, max_retries=2)
    return Analyzer(client, options.get("model", "claude-opus-5-5"), effort)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    options = load_options()
    if not options.get("app_token"):
        log.error("Set app_token in the add-on options")
    proxy = Proxy(
        build_analyzer(options),
        options.get("app_token", ""),
        DailyQuota(int(options.get("daily_limit", 200)), "/data/usage.json"),
    )
    if len(options.get("app_token", "")) < MIN_TOKEN_LENGTH:
        # Reachable from the Internet with HTTPS: a short token is guessable.
        log.error("app_token must have at least %s characters", MIN_TOKEN_LENGTH)
        raise SystemExit(1)
    port = int(options.get("port", 8095))
    handler = make_handler(proxy)
    if options.get("ssl"):
        server = TlsServer(("0.0.0.0", port), handler,
                           os.path.join("/ssl", options.get("certfile", "fullchain.pem")),
                           os.path.join("/ssl", options.get("keyfile", "privkey.pem")))
    else:
        server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    log.info("Smart Wardrobe AI proxy on port %s (%s), model %s", port,
             "https" if options.get("ssl") else "http", proxy.analyzer.model)
    server.serve_forever()


if __name__ == "__main__":
    main()
