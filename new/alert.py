"""
alert.py — Email Alerts via formsubmit.co AJAX API
formsubmit.co is a free, no-backend email relay.
The FIRST submission to a new email address requires you to click a
confirmation link in the inbox — do that once and all subsequent alerts
will arrive automatically.
Install:  pip install requests  (usually already present)
"""
from __future__ import annotations
import time
import requests
# ---------------------------------------------------------------------------
def send_email_alert(
    to_email: str,
    subject: str,
    message: str,
    sender_name: str = "AI Security System",
) -> bool:
    """
    POST an alert to formsubmit.co's AJAX endpoint (no page redirect).
    Parameters
    ----------
    to_email    : recipient email address
    subject     : email subject line
    message     : plain-text body
    sender_name : display name used as "From"
    Returns
    -------
    True if the API returned HTTP 200 / {"success": "true"}, else False.
    """
    url = f"https://formsubmit.co/ajax/othiraja64@gmail.com"
    payload = {
        "name": sender_name,
        "email": to_email,         # formsubmit echoes this as reply-to
        "_subject": subject,
        "message": message,
        "_captcha": "false",       # disable CAPTCHA for automated alerts
        "_template": "table",      # nicely formatted email
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=12)
        data = resp.json()
        success = resp.status_code == 200 and str(data.get("success", "")).lower() == "true"
        if not success:
            print(f"[Alert] formsubmit.co responded: {resp.status_code} — {data}")
        return success
    except requests.exceptions.Timeout:
        print("[Alert] Request timed out — is the network available?")
        return False
    except Exception as exc:
        print(f"[Alert] Unexpected error: {exc}")
        return False
# ---------------------------------------------------------------------------
class AlertManager:
    """
    Rate-limited wrapper around send_email_alert.
    Prevents flooding the recipient when an intruder stays in frame.
    """
    def __init__(self, to_email: str, cooldown_seconds: int = 30):
        self.to_email = to_email
        self.cooldown = cooldown_seconds
        self._last_sent: float = 0.0
    @property
    def seconds_until_next(self) -> float:
        elapsed = time.time() - self._last_sent
        return max(0.0, self.cooldown - elapsed)
    @property
    def ready(self) -> bool:
        return self.seconds_until_next == 0.0
    def try_send(
        self,
        subject: str,
        message: str,
        force: bool = False,
    ) -> bool:
        """
        Send only if cooldown has elapsed (or force=True).
        Returns True if the email was dispatched.
        """
        if not self.to_email:
            return False
        if not force and not self.ready:
            return False
        ok = send_email_alert(self.to_email, subject, message)
        if ok:
            self._last_sent = time.time()
        return ok