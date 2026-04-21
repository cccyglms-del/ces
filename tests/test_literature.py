import unittest
from unittest.mock import patch

from requests import exceptions as request_exceptions

from kmtool.analysis import literature


class _FakeResponse:
    def __init__(self, text="", payload=None):
        self.text = text
        self._payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, response=None, error=None):
        self.trust_env = True
        self.headers = {}
        self._response = response
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, params=None, timeout=None):
        if self._error is not None:
            raise self._error
        return self._response


class LiteratureTestCase(unittest.TestCase):
    def test_safe_request_retries_without_system_proxy(self):
        sessions = [
            _FakeSession(error=request_exceptions.ProxyError("proxy unavailable")),
            _FakeSession(response=_FakeResponse(payload={"ok": True})),
        ]

        with patch("kmtool.analysis.literature.requests.Session", side_effect=sessions):
            response = literature._safe_request("https://example.test", {"q": "demo"}, timeout=5)

        self.assertEqual(response.json(), {"ok": True})
        self.assertTrue(sessions[0].trust_env)
        self.assertFalse(sessions[1].trust_env)

    def test_extract_reported_hr_parses_confidence_interval(self):
        hr, ci_low, ci_high = literature.extract_reported_hr(
            "Median OS improved (HR 0.80; 95% CI 0.70 - 0.92)."
        )

        self.assertEqual(hr, 0.80)
        self.assertEqual(ci_low, 0.70)
        self.assertEqual(ci_high, 0.92)


if __name__ == "__main__":
    unittest.main()
