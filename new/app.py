"""
app.py — AI Security & Scene Intelligence System
• Background thread runs camera + YOLO + face-auth continuously
• Streamlit main thread ONLY renders — zero inference blocking
• Narration covers ALL detected objects simultaneously
"""
from __future__ import annotations

import os, time, cv2, threading, numpy as np, streamlit as st
from datetime import datetime
from collections import deque, Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Security System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');
*{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'JetBrains Mono',monospace;background:#070710;}
h1,h2,h3{font-family:'Syne',sans-serif!important;}

.banner{
  background:linear-gradient(120deg,#0d0d18 0%,#111827 60%,#0d1a0d 100%);
  border:1px solid #00ff8855;border-radius:14px;padding:18px 28px;margin-bottom:18px;
  box-shadow:0 0 40px rgba(0,255,136,.08);
}
.banner h1{color:#00ff88;font-size:1.75rem;margin:0;letter-spacing:-1px;
  text-shadow:0 0 24px rgba(0,255,136,.45);}
.banner p{color:#556;margin:4px 0 0;font-size:.7rem;letter-spacing:3px;text-transform:uppercase;}

/* Metric cards */
.mc{background:#0b0b18;border-radius:12px;padding:18px 10px;text-align:center;
    border:1px solid #1a1a2e;transition:box-shadow .3s;}
.mc.g{border-color:#00ff8866;box-shadow:0 0 18px rgba(0,255,136,.12);}
.mc.r{border-color:#ff444466;box-shadow:0 0 18px rgba(255,68,68,.12);}
.mc.b{border-color:#4488ff66;box-shadow:0 0 18px rgba(68,136,255,.12);}
.mn{font-size:2.4rem;font-weight:800;line-height:1;}
.ml{font-size:.6rem;letter-spacing:3px;text-transform:uppercase;color:#445;margin-top:6px;}

/* Scene box */
.sb{
  background:#080814;border:1px solid #00ff8822;border-left:3px solid #00ff88;
  border-radius:10px;padding:16px 20px;color:#d0fde8;font-size:.92rem;
  line-height:1.7;min-height:70px;transition:border-left-color .4s,background .4s;
}
.sb.al{border-left-color:#ff4444;background:#140808;color:#fdd0d0;}

/* Object tags */
.tags{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 4px;}
.tag{
  background:#0f1a0f;border:1px solid #00ff8833;color:#7fff99;
  padding:3px 10px;border-radius:20px;font-size:.68rem;letter-spacing:1px;
}
.tag.al{background:#1a0f0f;border-color:#ff444433;color:#ff9999;}

/* Log */
.lb{
  background:#050510;border:1px solid #111128;border-radius:10px;
  padding:12px;font-size:.7rem;max-height:260px;overflow-y:auto;color:#556;
  scrollbar-width:thin;scrollbar-color:#1a1a2e transparent;
}
.ls{color:#00ff88;} .ld{color:#ff4444;} .lw{color:#ffaa00;} .li{color:#4488ff;}

/* FPS badge */
.fps{
  display:inline-block;background:#0b1a0b;border:1px solid #00ff8844;
  color:#00ff88;padding:2px 10px;border-radius:20px;font-size:.65rem;
  letter-spacing:2px;margin-left:8px;vertical-align:middle;
}

/* Idle placeholder */
.idle{
  background:#050510;border:2px dashed #1a1a2e;border-radius:14px;
  height:380px;display:flex;align-items:center;justify-content:center;
  flex-direction:column;gap:12px;
}
.idle-icon{font-size:3.5rem;}
.idle-txt{color:#2a2a3e;font-size:.8rem;letter-spacing:3px;text-transform:uppercase;}

#MainMenu{visibility:hidden;} footer{visibility:hidden;}
[data-testid="stSidebar"]{background:#050510;border-right:1px solid #0f0f1e;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  ACTION MAP — every COCO object → natural action phrase
# ═══════════════════════════════════════════════════════════════
ACTION_MAP: dict[str, str] = {
    # Kitchen / food / drink
    "spoon"         : "holding a spoon",
    "fork"          : "holding a fork",
    "knife"         : "holding a knife",
    "cup"           : "holding a cup",
    "wine glass"    : "holding a glass",
    "bottle"        : "holding a bottle",
    "bowl"          : "eating from a bowl",
    "pizza"         : "eating pizza",
    "sandwich"      : "eating a sandwich",
    "hot dog"       : "eating a hot dog",
    "cake"          : "eating cake",
    "donut"         : "eating a donut",
    "banana"        : "eating a banana",
    "apple"         : "eating an apple",
    "orange"        : "eating an orange",
    "carrot"        : "eating a carrot",
    "broccoli"      : "eating broccoli",
    "food"          : "eating food",
    # Tech / work
    "laptop"        : "working on a laptop",
    "cell phone"    : "using a phone",
    "keyboard"      : "typing on a keyboard",
    "mouse"         : "using a mouse",
    "remote"        : "holding a remote",
    "tv"            : "watching TV",
    "monitor"       : "looking at a monitor",
    "book"          : "reading a book",
    "scissors"      : "using scissors",
    "pen"           : "writing",
    "pencil"        : "drawing or writing",
    # Personal items
    "toothbrush"    : "brushing teeth",
    "hair drier"    : "using a hair dryer",
    "handbag"       : "carrying a handbag",
    "backpack"      : "wearing a backpack",
    "suitcase"      : "carrying a suitcase",
    "umbrella"      : "holding an umbrella",
    "tie"           : "wearing a tie",
    # Furniture / environment
    "chair"         : "sitting on a chair",
    "couch"         : "sitting on a couch",
    "bed"           : "lying on a bed",
    "toilet"        : "near a toilet",
    "sink"          : "using the sink",
    "refrigerator"  : "opening the refrigerator",
    "microwave"     : "using the microwave",
    "oven"          : "using the oven",
    # Sport / outdoor
    "bicycle"       : "riding a bicycle",
    "motorcycle"    : "riding a motorcycle",
    "skateboard"    : "skateboarding",
    "surfboard"     : "surfboarding",
    "tennis racket" : "playing tennis",
    "baseball bat"  : "holding a baseball bat",
    "basketball"    : "playing basketball",
    "sports ball"   : "playing with a ball",
    "frisbee"       : "throwing a frisbee",
    "kite"          : "flying a kite",
    "skis"          : "skiing",
    "snowboard"     : "snowboarding",
    # Vehicles
    "car"           : "near a car",
    "truck"         : "near a truck",
    "bus"           : "on a bus",
    "train"         : "near a train",
    "airplane"      : "near an airplane",
    "boat"          : "on a boat",
    # Animals
    "cat"           : "with a cat",
    "dog"           : "with a dog",
    "bird"          : "near a bird",
    "horse"         : "near a horse",
    "cow"           : "near a cow",
}


def _narrate(objects: list[dict], auth_names: list[str], has_unknown: bool) -> str:
    """
    Produce a rich natural-language sentence mentioning ALL detected objects.
    """
    obj_names   = [o["name"].lower() for o in objects]
    non_person  = [n for n in obj_names if n != "person"]
    obj_counter = Counter(non_person)

    # Build action phrases for EVERY recognised object
    actions: list[str] = []
    env_items: list[str] = []
    for name, count in obj_counter.items():
        suffix = f" (×{count})" if count > 1 else ""
        if name in ACTION_MAP:
            actions.append(ACTION_MAP[name] + suffix)
        else:
            env_items.append(name + suffix)

    # ── Decide subject ──────────────────────────────────────────────────
    if has_unknown and not auth_names:
        subject  = "An UNIDENTIFIED person"
        is_alert = True
    elif has_unknown and auth_names:
        known    = " & ".join(auth_names)
        subject  = f"{known} (+ UNKNOWN intruder)"
        is_alert = True
    elif auth_names:
        subject  = auth_names[0] if len(auth_names) == 1 else " & ".join(auth_names)
        is_alert = False
    else:
        subject  = None
        is_alert = False

    prefix = "⚠️ SECURITY ALERT: " if is_alert else ""

    # ── Compose sentence ────────────────────────────────────────────────
    if subject:
        if actions and env_items:
            act_str = _join_actions(actions)
            env_str = ", ".join(env_items[:2])
            return f"{prefix}{subject} is {act_str}, in a space with {env_str}."
        elif actions:
            act_str = _join_actions(actions)
            return f"{prefix}{subject} is {act_str}."
        elif env_items:
            env_str = ", ".join(env_items[:3])
            return f"{prefix}{subject} is present near {env_str}."
        else:
            return f"{prefix}{subject} is present in the frame."
    else:
        # No person
        all_items = actions + env_items
        if all_items:
            return "The scene contains: " + ", ".join(all_items[:4]) + "."
        return "No activity detected — scene appears empty."


def _join_actions(actions: list[str]) -> str:
    """Grammatically join multiple action phrases."""
    if len(actions) == 1:
        return actions[0]
    if len(actions) == 2:
        return f"{actions[0]} and {actions[1]}"
    return ", ".join(actions[:-1]) + f", and {actions[-1]}"


# ═══════════════════════════════════════════════════════════════
#  SHARED STATE between inference thread and Streamlit thread
# ═══════════════════════════════════════════════════════════════
class _SharedState:
    """Thread-safe container for the latest detection results."""
    def __init__(self):
        self._lock       = threading.Lock()
        self.frame_rgb   = None          # latest annotated RGB frame
        self.objects     = []
        self.auth_names  = []
        self.has_unknown = False
        self.scene_text  = "Analysing…"
        self.logs        = deque(maxlen=120)
        self.auth_count  = 0
        self.unauth_count= 0
        self.frame_count = 0
        self.fps         = 0.0
        self.running     = False

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self):
        with self._lock:
            return {
                "frame_rgb"   : self.frame_rgb,
                "objects"     : list(self.objects),
                "auth_names"  : list(self.auth_names),
                "has_unknown" : self.has_unknown,
                "scene_text"  : self.scene_text,
                "logs"        : list(self.logs),
                "auth_count"  : self.auth_count,
                "unauth_count": self.unauth_count,
                "frame_count" : self.frame_count,
                "fps"         : self.fps,
                "running"     : self.running,
            }


# ── Session state bootstrap ───────────────────────────────────────────────────
if "shared" not in st.session_state:
    st.session_state.shared = _SharedState()
if "thread" not in st.session_state:
    st.session_state.thread = None

shared: _SharedState = st.session_state.shared


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADER (cached — runs once per browser session)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading AI models…")
def _load_models(yolo: str, thr: float, gkey: str):
    from face_auth     import FaceAuthenticator
    from object_detect import ObjectDetector
    from llm_describe  import SceneDescriber
    fa = FaceAuthenticator(threshold=thr)
    od = ObjectDetector(model_name=yolo, conf_threshold=0.35)   # lower = catch more
    sd = SceneDescriber(api_key=gkey) if gkey.strip() else None
    return fa, od, sd


# ═══════════════════════════════════════════════════════════════
#  INFERENCE THREAD  — runs continuously in background
# ═══════════════════════════════════════════════════════════════
def _inference_loop(
    cam_id       : int,
    face_model,
    obj_model,
    scene_model,
    llm_interval : int,
    alert_email  : str,
    alert_cooldown: int,
    shared       : _SharedState,
):
    cap = cv2.VideoCapture(int(cam_id), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(int(cam_id))
    if not cap.isOpened():
        shared.update(scene_text="❌ Cannot open camera.", running=False)
        return

    # Camera tuning for faster capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # ← key: always get the LATEST frame

    def log(msg, level="INFO"):
        p = {"INFO":"ℹ ","SUCCESS":"✓ ","WARNING":"⚠ ","DANGER":"✗ "}.get(level, "  ")
        with shared._lock:
            shared.logs.appendleft(f"[{datetime.now():%H:%M:%S}] {p}{msg}")

    log("Camera opened", "INFO")
    os.makedirs("unknown_faces", exist_ok=True)

    last_llm_ts   = 0.0
    last_alert_ts = 0.0
    fps_timer     = time.time()
    fps_frames    = 0
    auth_count    = 0
    unauth_count  = 0
    frame_count   = 0
    last_scene    = "Analysing…"

    while shared.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        fps_frames  += 1

        # ── FPS calculation (rolling 1-second window) ─────────────────
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = fps_frames / (now - fps_timer)
            fps_timer  = now
            fps_frames = 0
            shared.update(fps=round(fps, 1))

        # ── Object detection ──────────────────────────────────────────
        objects, annotated = obj_model.detect(frame)

        # ── Face authentication ───────────────────────────────────────
        auth_results = face_model.authenticate(frame)
        auth_names, has_unknown = [], False

        for r in auth_results:
            x1, y1, x2, y2 = r["bbox"]
            color = (0, 220, 0) if r["is_auth"] else (0, 0, 220)
            label = f"AUTH: {r['name']}" if r["is_auth"] else "UNAUTH"

            if r["is_auth"]:
                auth_names.append(r["name"])
                auth_count += 1
                log(f"Authorized — {r['name']}", "SUCCESS")
            else:
                has_unknown = True
                unauth_count += 1
                log("UNAUTHORIZED person detected!", "DANGER")
                cv2.imwrite(
                    f"unknown_faces/unknown_{int(time.time()*1000)}.jpg",
                    cv2.cvtColor(r["face_rgb"], cv2.COLOR_RGB2BGR),
                )
                # Email alert
                if alert_email and (now - last_alert_ts) >= alert_cooldown:
                    try:
                        from alert import send_email_alert
                        ok = send_email_alert(
                            alert_email,
                            "🚨 SECURITY ALERT: Unauthorized Access Detected",
                            f"Unauthorized person detected.\n"
                            f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                            f"Scene: {last_scene}\n"
                            f"Objects: {', '.join(o['name'] for o in objects) or 'none'}",
                        )
                        if ok:
                            last_alert_ts = now
                            log(f"Alert email sent → {alert_email}", "WARNING")
                    except Exception as e:
                        log(f"Alert error: {e}", "WARNING")

            # Draw face box on top of YOLO-annotated frame
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated, (x1, y1-lh-14), (x1+lw+6, y1), color, -1)
            cv2.putText(annotated, label, (x1+3, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            dist_lbl = f"d={r['distance']:.3f}"
            cv2.putText(annotated, dist_lbl, (x1, y2+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # ── LLM / rule-based narration (throttled) ────────────────────
        if now - last_llm_ts >= llm_interval:
            if scene_model:
                try:
                    last_scene = scene_model.describe(objects, auth_names, has_unknown)
                    log(f"Scene: {last_scene[:72]}", "INFO")
                except Exception as exc:
                    log(f"Gemini: {exc}", "WARNING")
                    last_scene = _narrate(objects, auth_names, has_unknown)
            else:
                last_scene = _narrate(objects, auth_names, has_unknown)
                log(f"Scene: {last_scene[:72]}", "INFO")
            last_llm_ts = now

        # ── HUD ───────────────────────────────────────────────────────
        ts_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(annotated, ts_str,
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1)
        cv2.putText(annotated,
                    f"Objs:{len(objects)}  FPS:{shared.fps:.1f}",
                    (10, annotated.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1)

        # ── Push results to shared state ──────────────────────────────
        shared.update(
            frame_rgb    = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            objects      = objects,
            auth_names   = auth_names,
            has_unknown  = has_unknown,
            scene_text   = last_scene,
            auth_count   = auth_count,
            unauth_count = unauth_count,
            frame_count  = frame_count,
        )

    cap.release()
    log("Camera released", "INFO")


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()
    gemini_key    = st.text_input("Gemini API Key", type="password",
                                   help="aistudio.google.com — leave blank for rule-based narration")
    alert_email   = st.text_input("Alert Email", placeholder="you@example.com")
    st.divider()
    yolo_model    = st.selectbox("YOLO Model",
                        ["yolo11n.pt","yolo11s.pt","yolo11m.pt","yolo12n.pt","yolo12s.pt"],
                        help="n=nano fastest, m=best accuracy")
    auth_thr      = st.slider("Face Auth Threshold", 0.10, 0.80, 0.40, 0.05)
    obj_conf      = st.slider("Object Confidence", 0.20, 0.80, 0.35, 0.05,
                               help="Lower = detects more objects")
    llm_interval  = st.slider("Scene Refresh (s)", 1, 10, 3)
    alert_cool    = st.slider("Alert Cooldown (s)", 10, 300, 30)
    cam_id        = st.number_input("Camera Index", 0, 4, 0)
    st.divider()
    st.caption("MTCNN · FaceNet · YOLOv11/12 · Gemini · formsubmit.co")


# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="banner">
  <h1>🔐 AI Security &amp; Scene Intelligence</h1>
  <p>Face Auth · Object Detection · Real-time Narration</p>
</div>""", unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
start_btn = c1.button("▶  Start Detection", use_container_width=True, type="primary")
stop_btn  = c2.button("⏹  Stop",            use_container_width=True)

# ── Build layout ONCE ─────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="medium")

with left:
    st.markdown("### 📹 Live Feed")
    video_ph = st.empty()

with right:
    m1, m2, m3 = st.columns(3)
    auth_ph   = m1.empty()
    unauth_ph = m2.empty()
    fps_ph    = m3.empty()

    st.markdown("### 🧠 Scene Intelligence")
    tags_ph   = st.empty()
    scene_ph  = st.empty()

    st.markdown("### 📋 Activity Log")
    log_ph    = st.empty()


# ── Render helpers ────────────────────────────────────────────────────────────
def _render(snap: dict):
    # Video
    if snap["frame_rgb"] is not None:
        video_ph.image(snap["frame_rgb"], channels="RGB", use_container_width=True)

    # Metrics
    auth_ph.markdown(
        f'<div class="mc g"><div class="mn" style="color:#00ff88">{snap["auth_count"]}</div>'
        f'<div class="ml">Authorized</div></div>', unsafe_allow_html=True)
    unauth_ph.markdown(
        f'<div class="mc r"><div class="mn" style="color:#ff4444">{snap["unauth_count"]}</div>'
        f'<div class="ml">Intruders</div></div>', unsafe_allow_html=True)
    fps_ph.markdown(
        f'<div class="mc b"><div class="mn" style="color:#4488ff">{snap["fps"]:.1f}</div>'
        f'<div class="ml">FPS</div></div>', unsafe_allow_html=True)

    # Object tags — show ALL detected objects
    obj_counter = Counter(o["name"] for o in snap["objects"] if o["name"].lower() != "person")
    tag_html = '<div class="tags">'
    alert_cls = "tag al" if snap["has_unknown"] else "tag"
    for name, cnt in obj_counter.most_common():
        label = f"{name} ×{cnt}" if cnt > 1 else name
        tag_html += f'<span class="{alert_cls}">{label}</span>'
    if not obj_counter:
        tag_html += '<span style="color:#2a2a3e;font-size:.7rem">No objects detected</span>'
    tag_html += "</div>"
    tags_ph.markdown(tag_html, unsafe_allow_html=True)

    # Scene
    txt = snap["scene_text"]
    is_alert = txt.startswith("⚠️") or "ALERT" in txt.upper()
    cls = "sb al" if is_alert else "sb"
    scene_ph.markdown(f'<div class="{cls}">{txt}</div>', unsafe_allow_html=True)

    # Logs
    out = []
    for ln in snap["logs"][:35]:
        if "✓" in ln:   out.append(f'<span class="ls">{ln}</span>')
        elif "✗" in ln: out.append(f'<span class="ld">{ln}</span>')
        elif "⚠" in ln: out.append(f'<span class="lw">{ln}</span>')
        elif "ℹ" in ln: out.append(f'<span class="li">{ln}</span>')
        else:            out.append(ln)
    log_ph.markdown(f'<div class="lb">{"<br>".join(out)}</div>', unsafe_allow_html=True)


def _idle():
    video_ph.markdown("""
    <div class="idle">
      <div class="idle-icon">📷</div>
      <div class="idle-txt">Press ▶ Start Detection</div>
    </div>""", unsafe_allow_html=True)
    for ph in (auth_ph, unauth_ph, fps_ph, tags_ph, scene_ph, log_ph):
        ph.empty()


# ── Start / Stop logic ────────────────────────────────────────────────────────
if start_btn and not shared.running:
    if not os.path.exists("embeddings/face_embeddings.npz"):
        st.error("❌ embeddings/face_embeddings.npz not found. Run your enrolment script first.")
        st.stop()
    try:
        face_m, obj_m, scene_m = _load_models(yolo_model, auth_thr, gemini_key)
    except Exception as exc:
        st.error(f"Model error: {exc}")
        st.stop()

    shared.update(running=True)
    t = threading.Thread(
        target=_inference_loop,
        args=(cam_id, face_m, obj_m, scene_m,
              llm_interval, alert_email, alert_cool, shared),
        daemon=True,
    )
    t.start()
    st.session_state.thread = t

if stop_btn:
    shared.update(running=False)

# ═══════════════════════════════════════════════════════════════
#  RENDER LOOP  — just pulls snapshots from shared state
#  No inference here — zero blocking — pure display
# ═══════════════════════════════════════════════════════════════
if shared.running:
    while shared.running:
        snap = shared.snapshot()
        _render(snap)
        time.sleep(0.033)          # ~30 fps display refresh
    # Thread finished
    _render(shared.snapshot())     # one final frame
    _idle()
else:
    _idle()