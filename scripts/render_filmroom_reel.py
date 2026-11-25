"""Render the 74s 1080p NFL Big Data Bowl 2027 Scouting Film-Room Reel with
studio ZONOS3-Mini (Wyatt) narration, restrained documentary pulse soundbed (-14 LUFS),
live Playwright Chromium 10 Hz Canvas 2D trajectory animations across all 4 scenarios,
all 5 Economist/NYT pure-white figures, and burned-in synced English SRT subtitles."""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
MATERIALS = ROOT / "materials"
FIGURES = MATERIALS / "figures"
HTML_REPORT = MATERIALS / "NFL_BDB2027_Interactive_Report.html"
AUDIO_DIR = Path("/Volumes/Extra/datasets/wave2_studio_audio/A2_NFL")

OUT_MP4 = MATERIALS / "NFL_BDB2027_FilmRoom_Reel_1080p.mp4"
OUT_SRT = MATERIALS / "NFL_BDB2027_FilmRoom_Reel_1080p.srt"
FFMPEG_BIN = "/Users/eligo/.local/bin/ffmpeg"
FFPROBE_BIN = "/Users/eligo/.local/bin/ffprobe"

W, H = 1920, 1080
FPS = 30
SR = 44100

# Strict Economist / NYT Pure-White Palette (zero beige, zero blue/cyan/teal/purple)
C_WHITE = (255, 255, 255)
C_INK = (17, 17, 17)
C_MUTED = (92, 92, 92)
C_RULE = (224, 224, 224)
C_GREEN = (24, 78, 50)
C_RED = (204, 41, 41)
C_GOLD = (200, 146, 46)
C_SOFT = (247, 247, 247)


def _load_font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    candidates: list[str] = []
    if serif:
        candidates = [
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/NewYork.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                idx = 1 if (bold and path.endswith("HelveticaNeue.ttc")) else 0
                return ImageFont.truetype(str(p), size=size, index=idx)
            except Exception:
                try:
                    return ImageFont.truetype(str(p), size=size)
                except Exception:
                    continue
    return ImageFont.load_default()


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"Expected 16-bit PCM WAV for {path}, got sample width {sw}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return data, sr


def _write_wav_stereo(path: Path, left: np.ndarray, right: np.ndarray, sr: int = SR) -> None:
    stereo = np.stack([left, right], axis=1)
    stereo = np.clip(stereo, -0.99, 0.99)
    pcm = (stereo * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def build_audio_and_srt(work_dir: Path, srt_path: Path) -> tuple[Path, float, dict[str, float]]:
    """Assemble the 3 studio ZONOS3-Mini WAVs with restrained documentary pulse bed and exact SRT."""
    a1_raw, sr1 = _read_wav_mono(AUDIO_DIR / "act1_thesis.wav")
    a2_raw, sr2 = _read_wav_mono(AUDIO_DIR / "act2_filmroom.wav")
    a3_raw, sr3 = _read_wav_mono(AUDIO_DIR / "act3_results.wav")
    assert sr1 == sr2 == sr3 == SR, f"Unexpected sample rates: {sr1}, {sr2}, {sr3}"

    # Trim only verified leading silence (measured via silencedetect: 0.56s, 0.51s, 0.59s)
    trim1, trim2, trim3 = 0.30, 0.30, 0.35
    a1 = a1_raw[int(trim1 * SR):]
    a2 = a2_raw[int(trim2 * SR):]
    a3 = a3_raw[int(trim3 * SR):]

    # Micro fade edges on voice clips to guarantee zero clicks
    fade_n = int(0.015 * SR)
    for clip in (a1, a2, a3):
        clip[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        clip[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

    gap12 = 0.30
    gap23 = 0.30
    tail_pad = 0.25

    t_act1_start = 0.0
    t_act2_start = len(a1) / SR + gap12
    t_act3_start = t_act2_start + len(a2) / SR + gap23
    total_sec = t_act3_start + len(a3) / SR + tail_pad
    # Snap total duration cleanly to exact video frame count (<= 74.8s, strictly inside 60-75s)
    total_frames = int(round(total_sec * FPS))
    total_sec = total_frames / FPS
    n_samples = int(round(total_sec * SR))

    voice = np.zeros(n_samples, dtype=np.float32)
    i1 = int(round(t_act1_start * SR))
    i2 = int(round(t_act2_start * SR))
    i3 = int(round(t_act3_start * SR))
    voice[i1 : i1 + len(a1)] += a1
    voice[i2 : i2 + len(a2)] += a2
    end3 = min(n_samples, i3 + len(a3))
    voice[i3:end3] += a3[: end3 - i3]

    # Synthesize restrained documentary pulse ambient bed (D dorian warm pulse @ 96 BPM)
    t = np.arange(n_samples, dtype=np.float32) / SR
    beat_hz = 96.0 / 60.0  # 1.6 Hz quarter-note pulse
    # Soft warm low-pass-like harmonic drone (D2=73.42, A2=110.0, F3=174.61, A3=220.0)
    # Chord progression shifts gently by Act: Dm -> Bbmaj7 -> C -> Dm
    bar_idx = ((t * beat_hz) / 4.0).astype(np.int32) % 4
    bass_freqs = np.where(bar_idx == 0, 73.42, np.where(bar_idx == 1, 58.27, np.where(bar_idx == 2, 65.41, 73.42)))
    fifth_freqs = bass_freqs * 1.5
    pad_freqs = np.where(bar_idx == 0, 220.0, np.where(bar_idx == 1, 174.61, np.where(bar_idx == 2, 196.0, 220.0)))

    phase_bass = 2.0 * math.pi * np.cumsum(bass_freqs) / SR
    phase_fifth = 2.0 * math.pi * np.cumsum(fifth_freqs) / SR
    phase_pad = 2.0 * math.pi * np.cumsum(pad_freqs) / SR

    # Gentle documentary pulse envelope (smooth 8th-note breath, zero harsh click)
    beat_phase = (t * beat_hz) % 1.0
    pulse_env = 0.55 + 0.45 * np.exp(-beat_phase * 5.2) * (1.0 - np.exp(-beat_phase * 55.0))
    # Subtle soft muted tick on downbeats (filtered sine burst at 180 Hz -> 90 Hz)
    tick_env = np.exp(-beat_phase * 28.0) * (1.0 - np.exp(-beat_phase * 180.0))
    tick = np.sin(2.0 * math.pi * 135.0 * beat_phase) * tick_env * 0.20

    bed_l = (
        0.48 * np.sin(phase_bass) * pulse_env
        + 0.24 * np.sin(phase_fifth + 0.2) * (0.7 + 0.3 * np.sin(2.0 * math.pi * 0.12 * t))
        + 0.14 * np.sin(phase_pad) * (0.65 + 0.35 * np.sin(2.0 * math.pi * 0.08 * t + 0.8))
        + tick
    ).astype(np.float32)
    bed_r = (
        0.48 * np.sin(phase_bass + 0.06) * pulse_env
        + 0.24 * np.sin(phase_fifth - 0.18) * (0.7 + 0.3 * np.cos(2.0 * math.pi * 0.12 * t))
        + 0.14 * np.sin(phase_pad + 0.45) * (0.65 + 0.35 * np.cos(2.0 * math.pi * 0.08 * t))
        + tick
    ).astype(np.float32)

    # Smooth fade-in (0.8s) and fade-out (1.6s) on bed
    fi = int(0.8 * SR)
    fo = int(1.6 * SR)
    bed_l[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    bed_r[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    bed_l[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    bed_r[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)

    # Smooth RMS speech envelope for gentle sidechain ducking (bed sits ~25 dB below voice)
    win = int(0.12 * SR)
    abs_v = np.abs(voice)
    cs = np.cumsum(np.insert(abs_v, 0, 0.0))
    env_v = (cs[win:] - cs[:-win]) / win
    env_v = np.pad(env_v, (win // 2, n_samples - len(env_v) - win // 2), mode="edge")
    duck = np.clip(1.0 - 4.5 * env_v, 0.38, 1.0).astype(np.float32)

    bed_gain = 0.055  # Restrained broadcast documentary background level
    mix_l = voice + bed_l * bed_gain * duck
    mix_r = voice + bed_r * bed_gain * duck

    raw_mix_wav = work_dir / "mix_raw.wav"
    norm_mix_wav = work_dir / "mix_lufs14.wav"
    _write_wav_stereo(raw_mix_wav, mix_l, mix_r, SR)

    # Two-pass or clean single-pass EBU R128 -14 LUFS loudness normalization to 48kHz stereo
    log_path = Path("/tmp/nfl_reel_audio_ffmpeg.log")
    with log_path.open("wb") as logf:
        subprocess.run(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                str(raw_mix_wav),
                "-af",
                "highpass=f=35,loudnorm=I=-14:TP=-1.5:LRA=11",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(norm_mix_wav),
            ],
            check=True,
            stdout=logf,
            stderr=logf,
        )

    # Exact timestamped English SRT matched to ZONOS3-Mini Wyatt audio silence boundaries
    s1 = t_act1_start - trim1
    s2 = t_act2_start - trim2
    s3 = t_act3_start - trim3
    cues = [
        # Act 1 (act1_thesis.wav)
        (max(0.15, 0.55 + s1), 4.85 + s1, "NFL scouting has trusted 40-yard stopwatches for decades,"),
        (5.25 + s1, 8.60 + s1, "yet 40 percent of top prospects now opt out."),
        (9.00 + s1, 13.35 + s1, "Beyond the Stopwatch unlocks hidden draft alpha"),
        (13.40 + s1, 18.85 + s1, "from 23 million real 10 Hz optical tracking frames across 510 rookies."),
        # Act 2 (act2_filmroom.wav)
        (0.50 + s2, 5.85 + s2, "Watch our 10 Hz optical film room compare fifth-round gem"),
        (6.00 + s2, 9.75 + s2, "Puka Nacua against second-round pick Ja'Lynn Polk."),
        (10.55 + s2, 16.60 + s2, "Even without a 40-yard time, Nacua explodes through gauntlet deceleration"),
        (16.75 + s2, 19.85 + s2, "at 9.59 yards per second,"),
        (20.30 + s2, 27.25 + s2, "the exact braking windows highlighted in gold\nby our multi-scale 1D-CNN attention encoder."),
        # Act 3 (act3_results.wav)
        (0.55 + s3, 8.55 + s3, "Under strict, zero-leakage cross-validation\nacross 22.6 million regular-season game frames,"),
        (8.95 + s3, 14.15 + s3, "our GPU deep fusion lifts Sunday game translation R-squared"),
        (14.30 + s3, 19.65 + s3, "from 0.156 to 0.240,"),
        (19.90 + s3, 22.95 + s3, "more than doubles unseen-year generalization,"),
        (23.30 + s3, min(total_sec - 0.10, 28.08 + s3), "and hits 0.543 on defensive line get-off."),
    ]

    def _fmt_ts(sec: float) -> str:
        ms = int(round(max(0.0, sec) * 1000))
        hh = ms // 3600000
        mm = (ms % 3600000) // 60000
        ss = (ms % 60000) // 1000
        rem = ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{rem:03d}"

    srt_blocks = []
    for idx, (st, en, text) in enumerate(cues, start=1):
        srt_blocks.append(f"{idx}\n{_fmt_ts(st)} --> {_fmt_ts(en)}\n{text}\n")
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

    # Also write an explicit 1920x1080 ASS file from the exact same cues so libass renders crisp lower-third subtitles
    def _fmt_ass(sec: float) -> str:
        cs = int(round(max(0.0, sec) * 100))
        hh = cs // 360000
        mm = (cs % 360000) // 6000
        ss = (cs % 6000) // 100
        rem = cs % 100
        return f"{hh}:{mm:02d}:{ss:02d}.{rem:02d}"

    ass_header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {W}",
            f"PlayResY: {H}",
            "ScaledBorderAndShadow: yes",
            "WrapStyle: 2",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Editorial,Helvetica Neue,34,&H00FFFFFF,&H00FFFFFF,&H1A111111,&H1A111111,-1,0,0,0,100,100,0.4,0,3,12,0,2,100,100,26,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )
    ass_lines = [ass_header]
    for st, en, text in cues:
        clean_t = text.replace("\n", "\\N")
        ass_lines.append(f"Dialogue: 0,{_fmt_ass(st)},{_fmt_ass(en)},Editorial,,0,0,0,,{clean_t}")
    ass_path = work_dir / "editorial_captions.ass"
    ass_path.write_text("\n".join(ass_lines) + "\n", encoding="utf-8")

    timing = {
        "act1_start": t_act1_start,
        "act2_start": t_act2_start,
        "act3_start": t_act3_start,
        "total_sec": total_sec,
        "total_frames": float(total_frames),
        "ass_path": str(ass_path),
    }
    return norm_mix_wav, total_sec, timing


def _compose_figure_slide(
    fig_png: Path,
    kicker: str,
    stat_pill: str,
    accent: tuple[int, int, int] = C_GREEN,
) -> Image.Image:
    """Place a 2x Retina pure-white figure onto a 1920x1080 studio frame with clean top bar & bottom subtitle zone."""
    canvas = Image.new("RGB", (W, H), C_WHITE)
    draw = ImageDraw.Draw(canvas)

    # Top Economist editorial red flag + 1px hairline
    draw.rectangle([0, 0, W, 6], fill=C_INK)
    draw.rectangle([72, 0, 152, 10], fill=C_RED)

    f_kicker = _load_font(20, bold=True)
    f_pill = _load_font(21, bold=True)
    draw.text((72, 26), kicker.upper(), fill=C_RED, font=f_kicker)

    # Right-aligned stat badge
    bbox = draw.textbbox((0, 0), stat_pill, font=f_pill)
    pw = bbox[2] - bbox[0] + 32
    ph = 36
    px0 = W - 72 - pw
    py0 = 18
    draw.rectangle([px0, py0, px0 + pw, py0 + ph], fill=accent)
    draw.text((px0 + 16, py0 + 6), stat_pill, fill=C_WHITE, font=f_pill)
    draw.line([(72, 66), (W - 72, 66)], fill=C_RULE, width=1)

    # Fit figure cleanly between y=76 and y=942 (leaving crisp 138px bottom subtitle safe area!)
    with Image.open(fig_png) as im:
        im_rgb = im.convert("RGB")
        max_w = W - 144
        max_h = 856
        scale = min(max_w / im_rgb.width, max_h / im_rgb.height)
        nw = int(round(im_rgb.width * scale))
        nh = int(round(im_rgb.height * scale))
        resized = im_rgb.resize((nw, nh), Image.Resampling.LANCZOS)
        ox = (W - nw) // 2
        oy = 76 + (max_h - nh) // 2
        canvas.paste(resized, (ox, oy))

    return canvas


def _compose_closing_card() -> Image.Image:
    """Crisp Economist/NYT closing card with verified 4-decimal headline metrics."""
    canvas = Image.new("RGB", (W, H), C_WHITE)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, W, 8], fill=C_INK)
    draw.rectangle([96, 0, 210, 16], fill=C_RED)

    f_eyebrow = _load_font(24, bold=True)
    f_title = _load_font(74, bold=True, serif=True)
    f_sub = _load_font(30, bold=False)
    f_num = _load_font(54, bold=True, serif=True)
    f_lbl = _load_font(20, bold=True)
    f_meta = _load_font(22, bold=False)

    draw.text((96, 130), "NFL BIG DATA BOWL 2027  ·  OPEN TRACK  ·  100% OFFICIAL DATA", fill=C_RED, font=f_eyebrow)
    draw.text((96, 185), "Beyond the Stopwatch", fill=C_INK, font=f_title)
    draw.text(
        (96, 286),
        "End-to-End 10 Hz Temporal 1D-CNN + Attention & Physics Fusion Across 23.4M Real Frames",
        fill=C_MUTED,
        font=f_sub,
    )
    draw.line([(96, 360), (W - 96, 360)], fill=C_INK, width=2)

    cards = [
        ("+0.2398", "STRATIFIED 5-FOLD R²", "Sunday Game Power (vs +0.1560)", C_GREEN),
        ("+0.2049", "TEMPORAL LOYO R²", "2.09x Unseen-Year Lift (vs +0.0982)", C_GREEN),
        ("+0.5432", "DL FIRST-STEP GET-OFF R²", "1D-CNN Deep Fusion (p = 0.0025)", C_RED),
        ("+0.3439", "NGS + DEEP OPTICAL R²", "Career Value Lift across 510 Rookies", C_INK),
    ]
    cw = (W - 192 - 3 * 28) // 4
    for i, (num, title, desc, color) in enumerate(cards):
        x0 = 96 + i * (cw + 28)
        y0 = 420
        draw.rectangle([x0, y0, x0 + cw, y0 + 245], fill=C_SOFT, outline=C_RULE, width=1)
        draw.rectangle([x0, y0, x0 + cw, y0 + 6], fill=color)
        draw.text((x0 + 26, y0 + 34), num, fill=color, font=f_num)
        draw.text((x0 + 26, y0 + 118), title, fill=C_INK, font=f_lbl)
        draw.text((x0 + 26, y0 + 156), desc, fill=C_MUTED, font=_load_font(19))

    draw.line([(96, 735), (W - 96, 735)], fill=C_RULE, width=1)
    draw.text(
        (96, 770),
        "Reproducible Python Package · Colab NVIDIA L4 GPU Temporal Encoder · Interactive 10 Hz Film-Room Report",
        fill=C_INK,
        font=f_meta,
    )
    draw.text(
        (96, 812),
        "510 Prospects (2023–2025) · 431,094 Combine 10 Hz Frames · 22,654,290 NFL Game Tracking Frames · Zero Leakage",
        fill=C_MUTED,
        font=f_meta,
    )
    return canvas


def _add_progress_bar(img: Image.Image, frac: float) -> Image.Image:
    """Stamp a crisp 4px Economist crimson playback progress indicator along the top edge."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    bar_w = int(round(W * max(0.0, min(1.0, frac))))
    if bar_w > 0:
        draw.rectangle([0, 0, bar_w, 5], fill=C_RED)
    return out


def _ken_burns(img: Image.Image, frac: float, zoom_start: float = 1.0, zoom_end: float = 1.032) -> Image.Image:
    """Gentle broadcast zoom (1.00 -> 1.032) keeping crisp typography and pure-white edges."""
    z = zoom_start + (zoom_end - zoom_start) * frac
    if abs(z - 1.0) < 1e-4:
        return img
    cw = int(round(W / z))
    ch = int(round(H / z))
    x0 = (W - cw) // 2
    y0 = max(0, int(round((H - ch) * 0.35)))
    cropped = img.crop((x0, y0, x0 + cw, y0 + ch))
    return cropped.resize((W, H), Image.Resampling.BILINEAR)


def _blend(a: Image.Image, b: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(a, b, max(0.0, min(1.0, alpha)))


def _capture_playwright_assets(work_dir: Path) -> dict[str, object]:
    """Capture genuine Chromium renders of the Hero, Architecture section, and LIVE animated 10 Hz Film-Room frames."""
    assets: dict[str, object] = {}
    url = HTML_REPORT.resolve().as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-mock-keychain", "--password-store=basic"],
        )
        context = browser.new_context(
            viewport={"width": W, "height": H},
            device_scale_factor=1,
        )
        page = context.new_page()
        page.goto(url, wait_until="load")
        page.evaluate("() => document.fonts ? document.fonts.ready : Promise.resolve()")

        # Inject non-destructive broadcast framing tweaks (keep 100% pure white & real DOM; reserve bottom safe margin)
        page.add_style_tag(
            content="""
            html, body { background: #FFFFFF !important; overflow: hidden !important; }
            .wrap { max-width: 1460px !important; }
            """
        )

        # 1. Hero Masthead capture (top of report)
        page.evaluate("() => window.scrollTo(0, 0)")
        hero_path = work_dir / "hero.png"
        page.screenshot(path=str(hero_path), type="png")
        assets["hero"] = Image.open(hero_path).convert("RGB")
        print("[PW] Captured Hero Masthead (1920x1080)")

        # 2. PyTorch 1D-CNN Architecture section capture
        page.evaluate("""() => {
            const el = document.getElementById('sec-cnn');
            if (el) el.scrollIntoView({ block: 'start' });
            window.scrollBy(0, -18);
        }""")
        cnn_path = work_dir / "cnn_arch.png"
        page.screenshot(path=str(cnn_path), type="png")
        assets["cnn_arch"] = Image.open(cnn_path).convert("RGB")
        print("[PW] Captured PyTorch 1D-CNN Architecture section")

        # 3. LIVE 10 Hz Film-Room Trajectory Replay across ALL 4 real scenarios!
        # Compact vertical rhythm so tabs + fieldCanvas + telemetryCanvas + BOTH live metric cards fit cleanly above y=936px
        page.add_style_tag(
            content="""
            #sec-filmroom { padding-top: 10px !important; padding-bottom: 8px !important; }
            #sec-filmroom .sec-head { margin-bottom: 6px !important; display: flex !important; align-items: baseline !important; justify-content: space-between !important; gap: 20px !important; }
            #sec-filmroom .sec-head h2 { font-size: 24px !important; margin: 0 !important; }
            #sec-filmroom .sec-lead { font-size: 13px !important; margin: 0 !important; }
            .film-room { padding: 12px 18px !important; }
            .tab-row { margin-bottom: 6px !important; }
            .tab-btn { padding: 6px 14px !important; font-size: 13px !important; }
            .film-headline { font-size: 15px !important; margin: 4px 0 2px !important; }
            .film-source { font-size: 11.5px !important; margin: 0 0 6px !important; }
            .canvas-stage { gap: 6px !important; }
            #fieldCanvas { max-height: 256px !important; width: 100% !important; height: auto !important; }
            #telemetryCanvas { max-height: 158px !important; width: 100% !important; height: auto !important; }
            .player-controls { margin: 6px 0 !important; padding: 5px 10px !important; }
            .ctrl-btn, .ctrl-select { padding: 4px 10px !important; font-size: 12px !important; }
            .telemetry-compare { gap: 12px !important; margin-top: 6px !important; }
            .telem-card { padding: 8px 14px !important; }
            .telem-meta { margin: 2px 0 6px !important; font-size: 11.5px !important; }
            .telem-metrics strong { font-size: 15px !important; }
            """
        )
        page.evaluate("""() => {
            const el = document.getElementById('sec-filmroom');
            if (el) el.scrollIntoView({ block: 'start' });
        }""")

        scenario_frames: list[list[Image.Image]] = []
        # Target number of real 10 Hz steps to sample per scenario
        # Scenario 0 (WR Gauntlet Puka Nacua vs Ja'Lynn Polk): full trajectory sweep (~76 steps)
        # Scenarios 1, 2, 3: crisp live sweeps (~36 steps each)
        targets = [76, 36, 36, 36]
        for s_idx, n_steps in enumerate(targets):
            page.click(f'.tab-btn[data-scenario-idx="{s_idx}"]')
            max_idx = int(page.evaluate("() => parseInt(document.getElementById('frameScrubber').max, 10)"))
            indices = [int(round(i * max_idx / max(1, n_steps - 1))) for i in range(n_steps)]
            frames_for_s: list[Image.Image] = []
            for k, f_idx in enumerate(indices):
                page.evaluate(
                    """(idx) => {
                        const s = document.getElementById('frameScrubber');
                        s.value = String(idx);
                        s.dispatchEvent(new Event('input', { bubbles: true }));
                    }""",
                    f_idx,
                )
                buf = page.screenshot(type="jpeg", quality=94)
                from io import BytesIO
                im = Image.open(BytesIO(buf)).convert("RGB")
                # Clean Economist header bar + bottom subtitle safe zone polish
                draw = ImageDraw.Draw(im)
                draw.rectangle([0, 946, W, H], fill=C_WHITE)
                draw.line([(72, 946), (W - 72, 946)], fill=C_RULE, width=1)
                frames_for_s.append(im)
            scenario_frames.append(frames_for_s)
            print(f"[PW] Captured Scenario {s_idx}: {len(frames_for_s)} live 10 Hz Canvas2D frames (max scrubber={max_idx})")

        assets["scenarios"] = scenario_frames
        browser.close()

    return assets


def render_reel(out_mp4: Path = OUT_MP4, out_srt: Path = OUT_SRT) -> None:
    with tempfile.TemporaryDirectory(prefix="nfl_bdb2027_reel_") as td:
        work_dir = Path(td)
        print("[1/4] Building -14 LUFS studio audio mix (ZONOS3-Mini Wyatt + documentary pulse bed) & SRT ...")
        audio_wav, total_sec, timing = build_audio_and_srt(work_dir, out_srt)
        total_frames = int(round(total_sec * FPS))
        print(
            f"      Duration: {total_sec:.2f}s ({total_frames} frames @ {FPS}fps) | "
            f"Act1={timing['act1_start']:.2f}s, Act2={timing['act2_start']:.2f}s, Act3={timing['act3_start']:.2f}s"
        )

        print("[2/4] Capturing live Playwright Chromium renders (Hero, 1D-CNN, 4x animated 10 Hz Film-Room scenarios) ...")
        assets = _capture_playwright_assets(work_dir)

        print("[3/4] Composing Economist/NYT pure-white figure frames & streaming 1080p raw frames to ffmpeg ...")
        fig1_slide = _compose_figure_slide(
            FIGURES / "fig1_kinematic_transfer_matrix.png",
            "Act I · Combine-to-NFL Kinematic Transfer Matrix",
            "100% Skill Coverage (510/510) · 22.65M NFL Frames",
            C_GREEN,
        )
        fig2_slide = _compose_figure_slide(
            FIGURES / "fig2_optical_velocity_decel_profiles.png",
            "Act I · Real 10 Hz Optical Braking & Launch Signatures",
            "Puka Nacua Opt-Out Peak Brake -5.8 yd/s²",
            C_GREEN,
        )
        fig4_slide = _compose_figure_slide(
            FIGURES / "fig4_cv_model_ablation.png",
            "Act III · Strict Zero-Leakage Nested CV & Colab L4 GPU Fusion",
            "R² +0.1560 to +0.2398 · LOYO +0.0982 to +0.2049",
            C_GREEN,
        )
        fig3_slide = _compose_figure_slide(
            FIGURES / "fig3_draft_capital_residual_alpha.png",
            "Act III · Draft-Capital Residual Alpha Across 510 Rookies",
            "33 Day-3 Optical Gems · Puka Nacua +2.79σ",
            C_RED,
        )
        fig5_slide = _compose_figure_slide(
            FIGURES / "fig5_positional_biomechanical_archetypes.png",
            "Act III · Position-Specific Biomechanical Playbook",
            "DL First-Step Get-Off R² = +0.5432 (p = 0.0025)",
            C_GREEN,
        )
        closing_slide = _compose_closing_card()

        hero_img: Image.Image = assets["hero"]  # type: ignore[assignment]
        cnn_img: Image.Image = assets["cnn_arch"]  # type: ignore[assignment]
        # Clean bottom safe zone on hero & cnn_arch so burned subtitles never collide with text
        for page_im in (hero_img, cnn_img):
            d = ImageDraw.Draw(page_im)
            d.rectangle([0, 946, W, H], fill=C_WHITE)
            d.line([(72, 946), (W - 72, 946)], fill=C_RULE, width=1)

        scenarios: list[list[Image.Image]] = assets["scenarios"]  # type: ignore[assignment]

        # Build exact time-aligned scene table synchronized to Wyatt's 3 studio narration acts
        a2 = timing["act2_start"]
        a3 = timing["act3_start"]
        t_end = total_sec

        # Scene definitions: (start_sec, end_sec, producer_fn(local_frac) -> Image)
        def _seq_frame(frames_list: list[Image.Image], local_f: float) -> Image.Image:
            idx = min(len(frames_list) - 1, max(0, int(local_f * len(frames_list))))
            return frames_list[idx]

        # For Scenario 0 (WR Gauntlet Puka Nacua vs Ja'Lynn Polk, 15.8s):
        # Smooth sweep through full rep (0%->100%) with deliberate emphasis through peak braking & attention window!
        def _sc0_frame(local_f: float) -> Image.Image:
            # Ease slightly slower through the middle braking/attention zone (0.35..0.75)
            fl = scenarios[0]
            # Slight S-curve highlight around golden braking zone
            if local_f < 0.72:
                prog = local_f / 0.72
            else:
                # Re-sweep final peak acceleration & gold attention band cleanly for emphasis
                prog = 0.42 + 0.58 * ((local_f - 0.72) / 0.28)
            idx = min(len(fl) - 1, max(0, int(prog * len(fl))))
            return fl[idx]

        scenes: list[tuple[float, float, object]] = [
            # ACT 1: Thesis & Blind Spot (0.0 -> a2)
            (0.0, 6.2, lambda lf: _ken_burns(hero_img, lf, 1.00, 1.028)),
            (6.2, 12.8, lambda lf: _ken_burns(fig1_slide, lf, 1.00, 1.025)),
            (12.8, a2, lambda lf: _ken_burns(fig2_slide, lf, 1.00, 1.025)),
            # ACT 2: Live 10 Hz Interactive Canvas 2D Film-Room across ALL 4 scenarios (a2 -> a3)
            # Keep Scenario 0 (WR Gauntlet Puka Nacua 9.59 yd/s vs Ja'Lynn Polk) through 38.6s to match narration!
            (a2, a2 + 19.6, _sc0_frame),                                  # WR Gauntlet: Puka Nacua vs Ja'Lynn Polk
            (a2 + 19.6, a2 + 22.2, lambda lf: _seq_frame(scenarios[1], lf)),  # WR Comeback Route
            (a2 + 22.2, a2 + 24.8, lambda lf: _seq_frame(scenarios[2], lf)),  # DB Backpedal & 90-Deg Break
            (a2 + 24.8, a3, lambda lf: _seq_frame(scenarios[3], lf)),         # Sunday NFL 80-Yd Game Translation
            # ACT 3: Colab L4 GPU Deep Fusion & Exact Results (a3 -> t_end)
            (a3, a3 + 5.6, lambda lf: _ken_burns(cnn_img, lf, 1.00, 1.022)),
            (a3 + 5.6, a3 + 15.2, lambda lf: _ken_burns(fig4_slide, lf, 1.00, 1.026)),
            (a3 + 15.2, a3 + 21.0, lambda lf: _ken_burns(fig3_slide, lf, 1.00, 1.026)),
            (a3 + 21.0, a3 + 25.4, lambda lf: _ken_burns(fig5_slide, lf, 1.00, 1.024)),
            (a3 + 25.4, t_end, lambda lf: closing_slide),
        ]

        fade_sec = 0.35
        raw_mp4 = work_dir / "video_nosub.mp4"
        ff_log = Path("/tmp/nfl_reel_encode_ffmpeg.log")

        # Stream raw RGB24 frames directly to ffmpeg stdin; redirect stdout/stderr to log file (NEVER unread PIPE!)
        with ff_log.open("wb") as logf:
            proc = subprocess.Popen(
                [
                    FFMPEG_BIN,
                    "-y",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{W}x{H}",
                    "-pix_fmt",
                    "rgb24",
                    "-r",
                    str(FPS),
                    "-i",
                    "-",
                    "-i",
                    str(audio_wav),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    "48000",
                    "-shortest",
                    str(raw_mp4),
                ],
                stdin=subprocess.PIPE,
                stdout=logf,
                stderr=logf,
            )
            assert proc.stdin is not None
            try:
                for fi in range(total_frames):
                    t_sec = fi / FPS
                    # Find active scene
                    sc_i = len(scenes) - 1
                    for idx, (st, en, _) in enumerate(scenes):
                        if st <= t_sec < en:
                            sc_i = idx
                            break
                    st, en, fn = scenes[sc_i]
                    lf = (t_sec - st) / max(1e-6, en - st)
                    frame_im: Image.Image = fn(lf)  # type: ignore[operator]

                    # Crossfade with previous scene over first fade_sec seconds
                    if sc_i > 0 and (t_sec - st) < fade_sec:
                        p_st, p_en, p_fn = scenes[sc_i - 1]
                        prev_im: Image.Image = p_fn(1.0)  # type: ignore[operator]
                        alpha = (t_sec - st) / fade_sec
                        frame_im = _blend(prev_im, frame_im, alpha)

                    frame_im = _add_progress_bar(frame_im, fi / max(1, total_frames - 1))
                    proc.stdin.write(frame_im.tobytes())
            finally:
                proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg raw video encode failed with rc={rc}; see {ff_log}")

        # 4. Burn crisp 1920x1080 ASS Economist/NYT English subtitles onto final 1080p MP4 with faststart
        print("[4/4] Burning crisp 1080p Economist/NYT English subtitles into final MP4 ...")
        ass_escaped = str(Path(str(timing["ass_path"])).resolve()).replace("\\", "\\\\").replace(":", "\\:")
        sub_log = Path("/tmp/nfl_reel_sub_ffmpeg.log")
        with sub_log.open("wb") as logf:
            subprocess.run(
                [
                    FFMPEG_BIN,
                    "-y",
                    "-i",
                    str(raw_mp4),
                    "-vf",
                    f"ass='{ass_escaped}'",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(out_mp4),
                ],
                check=True,
                stdout=logf,
                stderr=logf,
            )

        print(f"[DONE] Rendered {out_mp4} ({out_mp4.stat().st_size:,} bytes) and {out_srt} ({out_srt.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 1080p NFL BDB 2027 Film-Room Reel with subtitles.")
    parser.add_argument("--out-mp4", type=Path, default=OUT_MP4)
    parser.add_argument("--out-srt", type=Path, default=OUT_SRT)
    args = parser.parse_args()
    render_reel(out_mp4=args.out_mp4, out_srt=args.out_srt)


if __name__ == "__main__":
    main()
