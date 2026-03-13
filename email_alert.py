import requests

EMAIL = "othiraja64@gmail.com"

def send_sms_alert():

    url = f"https://formsubmit.co/{EMAIL}"

    data = {
        "subject": "🚨 Face Authentication Alert",
        "message": "Unauthorized person detected!",
        "_captcha": "false"
    }

    try:
        requests.post(url, data=data)
        print("📧 Email alert sent")

    except Exception as e:
        print("❌ Email alert failed:", e)