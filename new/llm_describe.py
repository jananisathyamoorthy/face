"""
llm_describe.py — Scene-to-Text via Google Gemini
Converts raw YOLO detections + face-auth results into a human-readable
action sentence using the Gemini 1.5 Flash model.

Install:  pip install google-generativeai
"""

from __future__ import annotations


class SceneDescriber:
    """
    Wrap the Google Gemini API to produce natural-language scene summaries.

    Parameters
    ----------
    api_key : str
        Your Google AI Studio API key (https://aistudio.google.com/).
    model   : str
        Gemini model name — "gemini-1.5-flash" is fast & cheap.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
    ):
        import google.generativeai as genai

        genai.configure(api_key="AIzaSyBsoeX5MXZPDV1YuhblcsgEuuZjR9DqGBM")
        self._model = genai.GenerativeModel(model)
        self._genai = genai
        print(f"[SceneDescriber] Connected to '{model}'.")

    # ------------------------------------------------------------------
    def describe(
        self,
        objects: list[dict],
        auth_names: list[str] | None = None,
        has_unknown: bool = False,
    ) -> str:
        """
        Generate a single natural-language sentence that describes the scene.

        Parameters
        ----------
        objects     : list of dicts from ObjectDetector.detect()
        auth_names  : names of authenticated people visible in frame
        has_unknown : True if at least one unauthenticated face was found
        """
        # Build object list (deduplicated with counts)
        from collections import Counter

        counts = Counter(o["name"] for o in objects)
        if counts:
            obj_str = ", ".join(
                f"{n} (×{c})" if c > 1 else n for n, c in counts.items()
            )
        else:
            obj_str = "no objects"

        # Build identity context
        person_ctx = ""
        if auth_names:
            names_joined = ", ".join(auth_names)
            person_ctx += f"Authorized persons on screen: {names_joined}. "
        if has_unknown:
            person_ctx += (
                "⚠️ WARNING: An UNAUTHORIZED / UNKNOWN person is present. "
                "Treat this as a potential security threat. "
            )

        prompt = f"""You are a concise vision assistant for a security system.

Context:
{person_ctx}
Detected objects: {obj_str}

Your task: Write ONE short, natural English sentence (max 25 words) that describes
what is happening in the scene. Be direct.

Rules:
- Use active voice ("He is cutting…", "A person is sitting…")
- If an unauthorized person is present, start with "⚠️ SECURITY ALERT:"
- If only authorized persons are present, describe the activity normally
- If no people are detected, describe the objects / environment

Examples:
  knife + vegetable + person →  "He is chopping vegetables on the counter."
  laptop + authorized_user   →  "Alice is working on her laptop."
  unauthorized + phone       →  "⚠️ SECURITY ALERT: An unidentified person is using a phone."

Now generate the sentence:"""

        try:
            response = self._model.generate_content(prompt)
            return response.text.strip()
        except Exception as exc:
            # Graceful degradation
            fallback = f"Scene contains: {obj_str}."
            if has_unknown:
                fallback = f"⚠️ SECURITY ALERT: Unknown person detected. {fallback}"
            elif auth_names:
                fallback = f"{', '.join(auth_names)} visible. {fallback}"
            print(f"[SceneDescriber] Gemini error: {exc}")
            return fallback