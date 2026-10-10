"""Tests of the proxy, with a fake Gemini API: no network, no cost."""

import datetime
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer

import ai_proxy
import analysis

TAXONOMY = {
    "categories": ["top", "shirt", "jeans", "bottom"],
    "colors": ["navy", "white", "black"],
    "styles": ["casual", "classic"],
    "materials": ["cotton", "linen", "denim"],
    "patterns": ["solid", "striped"],
    "fits": ["slim", "regular"],
    "seasons": ["all_year", "summer", "winter"],
}


def request_payload(**overrides):
    payload = {"schemaVersion": "1", "locale": "fr", "taxonomy": TAXONOMY}
    payload.update(overrides)
    return payload


def good_answer(**overrides):
    answer = {
        "is_clothing": True,
        "name": {"value": "Chemise en lin bleue", "confidence": 0.8},
        "category": {"value": "shirt", "confidence": 0.95},
        "primary_color": {"value": "navy", "confidence": 0.9},
        "secondary_colors": {"values": ["white", "navy", "purple"], "confidence": 0.5},
        "styles": {"values": ["casual"], "confidence": 0.6},
        "materials": {"values": ["linen"], "confidence": 0.4},
        "seasons": {"values": ["summer"], "confidence": 0.7},
        "pattern": {"value": "unknown", "confidence": 0.2},
        "fit": {"value": "regular", "confidence": 0.5},
        "formality": {"value": 140, "confidence": 1.7},
        "description": "Une chemise   bleue.",
        "warnings": [],
    }
    answer.update(overrides)
    return answer


class AnalysisTest(unittest.TestCase):
    def test_rejects_invalid_requests(self):
        for payload in [
            request_payload(schemaVersion="2"),
            request_payload(locale="de"),
            request_payload(taxonomy={**TAXONOMY, "colors": []}),
            request_payload(taxonomy={**TAXONOMY, "colors": ["Navy; DROP"]}),
        ]:
            with self.assertRaises(analysis.RequestError):
                analysis.parse_request(payload)

    def test_schema_only_allows_known_identifiers(self):
        request = analysis.parse_request(request_payload())
        schema = analysis.build_schema(request)
        category = schema["properties"]["category"]["properties"]["value"]["enum"]
        self.assertEqual(category, ["bottom", "jeans", "shirt", "top", "unknown"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_normalizes_and_drops_what_it_cannot_trust(self):
        request = analysis.parse_request(request_payload())
        result = analysis.normalize(good_answer(), request)
        fields = result["fields"]
        self.assertEqual(fields["category"]["value"], "shirt")
        # The primary colour and unknown ids leave the secondary colours.
        self.assertEqual(fields["secondaryColors"]["value"], ["white"])
        self.assertNotIn("pattern", fields)
        self.assertEqual(fields["formality"], {"value": 100, "confidence": 1.0})
        self.assertEqual(result["description"], "Une chemise bleue.")

    def test_a_photo_without_clothing_gives_no_field(self):
        request = analysis.parse_request(request_payload())
        result = analysis.normalize(
            good_answer(is_clothing=False, warnings=["Aucun vêtement visible."]),
            request)
        self.assertEqual(result["fields"], {})
        self.assertFalse(result["isClothing"])
        self.assertEqual(result["warnings"], ["Aucun vêtement visible."])


class FakeGemini:
    """Stands for one generateContent call: records the body, answers."""

    def __init__(self, status=200, payload=None, answer=None, finish="STOP",
                 headers=None, error=None):
        text = answer if isinstance(answer, str) else json.dumps(answer or good_answer())
        self.status = status
        self.headers = headers or {}
        self.error = error
        self.payload = payload if payload is not None else {
            "candidates": [{"finishReason": finish, "content": {"parts": [
                {"text": "Looking at the shirt.", "thought": True},
                {"text": text},
            ]}}],
            "modelVersion": "gemini-3.8-flash-001",
            "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 300,
                              "thoughtsTokenCount": 120},
        }
        self.calls = []

    def __call__(self, url, key, body):
        self.calls.append((url, key, body))
        if self.error is not None:
            raise self.error
        return self.status, self.headers, self.payload


class GeminiAnalyzerTest(unittest.TestCase):
    def setUp(self):
        self.request = analysis.parse_request(request_payload())

    def analyze(self, fake, model="gemini-3.8-flash"):
        return ai_proxy.GeminiAnalyzer("gemini-key", model, "low", post=fake).analyze(
            self.request, b"\xff\xd8jpeg", "image/jpeg")

    def test_sends_the_photo_and_schema_and_skips_thoughts(self):
        fake = FakeGemini()
        result = self.analyze(fake)
        url, key, body = fake.calls[0]
        self.assertTrue(url.endswith("/models/gemini-3.8-flash:generateContent"))
        self.assertEqual(key, "gemini-key")
        image = body["contents"][0]["parts"][0]["inlineData"]
        self.assertEqual(image, {"mimeType": "image/jpeg", "data": "/9hqcGVn"})
        config = body["generationConfig"]
        self.assertEqual(config["responseMimeType"], "application/json")
        self.assertEqual(config["responseJsonSchema"], analysis.build_schema(self.request))
        self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "low"})
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"],
                         analysis.SYSTEM_PROMPT)
        self.assertEqual(result["fields"]["category"]["value"], "shirt")
        self.assertEqual(result["model"], "gemini-3.8-flash-001")
        self.assertEqual(result["usage"], {"inputTokens": 1500, "outputTokens": 420,
                                           "cacheReadTokens": 0})

    def test_blocked_photo_is_unsupported(self):
        for fake in [FakeGemini(finish="SAFETY"),
                     FakeGemini(payload={"promptFeedback": {"blockReason": "OTHER"}})]:
            with self.assertRaises(ai_proxy.ProviderError) as caught:
                self.analyze(fake)
            self.assertEqual(caught.exception.status, 422)
            self.assertFalse(caught.exception.retryable)

    def test_truncation_and_unreadable_output_are_retryable(self):
        for fake in [FakeGemini(finish="MAX_TOKENS"), FakeGemini(answer="not json")]:
            with self.assertRaises(ai_proxy.ProviderError) as caught:
                self.analyze(fake)
            self.assertEqual(caught.exception.status, 502)
            self.assertTrue(caught.exception.retryable)

    def test_free_tier_rate_limit_is_retryable_with_its_delay(self):
        fake = FakeGemini(status=429, payload={"error": {"status": "RESOURCE_EXHAUSTED"}},
                          headers={"retry-after": "30"})
        with self.assertRaises(ai_proxy.ProviderError) as caught:
            self.analyze(fake)
        self.assertEqual((caught.exception.status, caught.exception.retry_after), (503, 30))
        self.assertTrue(caught.exception.retryable)

    def test_refused_key_is_not_retryable(self):
        payload = {"error": {"status": "INVALID_ARGUMENT", "details": [
            {"reason": "API_KEY_INVALID"}]}}
        for fake in [FakeGemini(status=400, payload=payload),
                     FakeGemini(status=403, payload={})]:
            with self.assertRaises(ai_proxy.ProviderError) as caught:
                self.analyze(fake)
            self.assertFalse(caught.exception.retryable)

    def test_network_failures(self):
        for error, status in [(urllib.error.URLError("down"), 502), (TimeoutError(), 504)]:
            with self.assertRaises(ai_proxy.ProviderError) as caught:
                self.analyze(FakeGemini(error=error))
            self.assertEqual(caught.exception.status, status)
            self.assertTrue(caught.exception.retryable)

    def test_every_gemini_model_offered_by_the_add_on_is_called_by_name(self):
        with open(os.path.join(os.path.dirname(__file__), "config.yaml"),
                  encoding="utf-8") as file:
            line = next(l for l in file if l.strip().startswith("gemini_model: list("))
        models = line.split("list(", 1)[1].rstrip().rstrip(")").split("|")
        for model in models:
            fake = FakeGemini()
            self.analyze(fake, model)
            self.assertTrue(fake.calls[0][0].endswith(f"/{model}:generateContent"))

    def test_options_build_the_analyzer(self):
        analyzer = ai_proxy.build_analyzer({"gemini_api_key": "k"})
        self.assertIsInstance(analyzer, ai_proxy.GeminiAnalyzer)
        self.assertEqual((analyzer.model, analyzer.effort), ("gemini-3.8-flash", "low"))
        chosen = ai_proxy.build_analyzer({"gemini_api_key": "k",
                                          "gemini_model": "gemini-3.5-flash-lite",
                                          "effort": "high"})
        self.assertEqual((chosen.model, chosen.effort), ("gemini-3.5-flash-lite", "high"))


class QuotaAndIdempotencyTest(unittest.TestCase):
    def test_daily_quota_resets_the_next_day(self):
        day = [datetime.date(2026, 10, 5)]
        quota = ai_proxy.DailyQuota(2, today=lambda: day[0])
        self.assertTrue(quota.take())
        self.assertTrue(quota.take())
        self.assertFalse(quota.take())
        day[0] = datetime.date(2026, 10, 6)
        self.assertTrue(quota.take())

    def test_same_key_computes_once(self):
        cache = ai_proxy.IdempotencyCache()
        calls = []

        def compute():
            calls.append(1)
            return len(calls)

        self.assertEqual(cache.run("k", compute), 1)
        self.assertEqual(cache.run("k", compute), 1)
        self.assertEqual(cache.run("other", compute), 2)


class HttpTest(unittest.TestCase):
    def setUp(self):
        self.gemini = FakeGemini()
        self.proxy = ai_proxy.Proxy(
            ai_proxy.GeminiAnalyzer("gemini-key", "gemini-3.8-flash", "low",
                                    post=self.gemini),
            "secret-token",
            ai_proxy.DailyQuota(2),
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ai_proxy.make_handler(self.proxy))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, token="secret-token", key=None, payload=None, image=b"\xff\xd8jpeg"):
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n"
            "Content-Type: application/json\r\n\r\n"
        ).encode() + json.dumps(payload or request_payload()).encode() + (
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            "filename=\"photo.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"
        ).encode() + image + f"\r\n--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}",
                   "Authorization": f"Bearer {token}"}
        if key:
            headers["Idempotency-Key"] = key
        request = urllib.request.Request(f"{self.base}/v1/ai/clothing-analyses",
                                         data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_analysis_round_trip(self):
        status, body = self.post()
        self.assertEqual(status, 200)
        self.assertIsNone(body["error"])
        self.assertEqual(body["data"]["fields"]["category"]["value"], "shirt")
        self.assertIn("requestId", body["meta"])
        sent = self.gemini.calls[0][2]["contents"][0]["parts"][0]["inlineData"]
        self.assertEqual(sent["data"], "/9hqcGVn")  # the image bytes

    def test_repeated_wrong_tokens_block_the_address(self):
        self.proxy.throttle = ai_proxy.AuthThrottle(limit=3)
        for _ in range(3):
            self.assertEqual(self.post(token="guess")[0], 401)
        status, body = self.post()  # Even the right token waits now.
        self.assertEqual(status, 429)
        self.assertEqual(body["error"]["retryAfterSeconds"], 600)
        self.assertEqual(self.gemini.calls, [])

    def test_wrong_token_is_refused_without_calling_gemini(self):
        status, body = self.post(token="nope")
        self.assertEqual(status, 401)
        self.assertEqual(body["error"]["code"], "unauthorized")
        self.assertEqual(self.gemini.calls, [])

    def test_a_retry_with_the_same_key_is_not_billed_twice(self):
        self.post(key="fingerprint-1")
        status, _ = self.post(key="fingerprint-1")
        self.assertEqual(status, 200)
        self.assertEqual(len(self.gemini.calls), 1)

    def test_quota_answers_429_with_retry_after(self):
        self.post()
        self.post()
        status, body = self.post()
        self.assertEqual(status, 429)
        self.assertEqual(body["error"]["code"], "quota_exceeded")
        self.assertTrue(body["error"]["retryable"])
        self.assertGreater(body["error"]["retryAfterSeconds"], 0)

    def test_invalid_request_is_a_validation_error(self):
        status, body = self.post(payload=request_payload(locale="xx"))
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "validation")

    def test_health_reports_the_model(self):
        request = urllib.request.Request(f"{self.base}/v1/health",
                                         headers={"Authorization": "Bearer secret-token"})
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read())
        self.assertEqual(body["data"]["model"], "gemini-3.8-flash")
        self.assertEqual(body["data"]["remainingToday"], 2)


class ThrottleTest(unittest.TestCase):
    def test_failures_expire_and_success_clears(self):
        now = [0.0]
        throttle = ai_proxy.AuthThrottle(limit=2, window=60, clock=lambda: now[0])
        throttle.failed("a")
        throttle.failed("a")
        self.assertTrue(throttle.blocked("a"))
        self.assertFalse(throttle.blocked("b"))
        now[0] = 61
        self.assertFalse(throttle.blocked("a"))
        throttle.failed("a")
        throttle.succeeded("a")
        throttle.failed("a")
        self.assertFalse(throttle.blocked("a"))


@unittest.skipUnless(shutil.which("openssl"), "openssl is needed for a test certificate")
class TlsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.cert = os.path.join(self.directory, "fullchain.pem")
        self.key = os.path.join(self.directory, "privkey.pem")
        self.make_certificate("first.test")
        proxy = ai_proxy.Proxy(
            ai_proxy.GeminiAnalyzer("gemini-key", "gemini-3.8-flash", "low",
                                    post=FakeGemini()),
            "secret-token", ai_proxy.DailyQuota(5))
        self.server = ai_proxy.TlsServer(("127.0.0.1", 0), ai_proxy.make_handler(proxy),
                                         self.cert, self.key)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.directory)

    def make_certificate(self, name):
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
             "-subj", f"/CN={name}", "-keyout", self.key, "-out", self.cert],
            check=True, capture_output=True)

    def health(self):
        context = ssl.create_default_context(cafile=self.cert)
        context.check_hostname = False
        request = urllib.request.Request(
            f"https://127.0.0.1:{self.server.server_address[1]}/v1/health",
            headers={"Authorization": "Bearer secret-token"})
        with urllib.request.urlopen(request, context=context) as response:
            return json.loads(response.read())

    def test_serves_https_and_reloads_a_renewed_certificate(self):
        self.assertEqual(self.health()["data"]["status"], "ok")
        self.make_certificate("renewed.test")
        # A different modification time, as a renewal would give.
        stamp = os.path.getmtime(self.cert) + 10
        os.utime(self.cert, (stamp, stamp))
        self.assertEqual(self.health()["data"]["status"], "ok")

    def test_plain_http_is_refused(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_address[1]}/v1/health")
        with self.assertRaises(Exception):
            urllib.request.urlopen(request, timeout=5)


if __name__ == "__main__":
    unittest.main()
