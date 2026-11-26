"""Render the V6 Championship 74s 1080p NFL Big Data Bowl 2027 Scouting Film-Room Reel.

Features:
- Studio ZONOS3-Mini (Wyatt) 3-act narration + restrained documentary pulse soundbed (-14 LUFS)
- V6 Opening Metrics Banner (0.2398 -> 0.2562 OOF R², LOYO 0.2286, DL/EDGE 0.7045, Partial R² +0.1121)
- LIVE 1920x1080 Playwright Chromium recording of `materials/NFL_BDB2027_Interactive_FilmRoom.html`:
    * Synchronized 10 Hz Ghost Comparison + 0.1s Dual-Head Attention Glow across ALL 4 position matchups
      (WR/TE Puka Nacua vs Ja'Lynn Polk, DB Jaylon Jones vs Cam Smith, DL/EDGE Tuli Tuipulotu vs
      Lukas Van Ness, OL Mason McCormick vs Will Campbell)
    * LIVE interactive GM Draft-Room Surplus EV Calculator slider drag & preset switches
- All 6 refreshed Economist/NYT pure-white 2x Retina figures (fig1-fig6, spotlighting new fig6 & fig4)
- V6 Closing Card + burned-in synced English SRT/ASS captions in bottom safe zone (zero occlusion)
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
MATERIALS = ROOT / "materials"
FIGURES = MATERIALS / "figures"
HTML_FILMROOM = MATERIALS / "NFL_BDB2027_Interactive_FilmRoom.html"
AUDIO_DIR = Path("/Volumes/Extra/datasets/wave2_studio_audio/A2_NFL")

OUT_MP4 = MATERIALS / "NFL_BDB2027_FilmRoom_Reel_1080p.mp4"
OUT_SRT = MATERIALS / "NFL_BDB2027_FilmRoom_Reel_1080p.srt"
FFMPEG_BIN = "/Users/eligo/.local/bin/ffmpeg"

W, H = 1920, 1080
FPS = 30
SR = 44100

# Strict Economist / NYT Pure-White Palette (zero beige, zero blue/cyan/teal/purple)
C_WHITE = (255, 255, 255)
C_INK = (17, 17, 17)
C_MUTED = (82, 82, 82)
C_RULE = (224, 224, 224)
C_GREEN = (24, 78, 50)
C_RED = (196, 60, 41)
C_GOLD = (198, 139, 44)
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


def build_audio_and_srt(work_dir: Path, srt_path: Path) -> tuple[Path, float, dict[str, float | str]]:
    """Assemble the 3 studio ZONOS3-Mini Wyatt WAVs with restrained documentary pulse bed (-14 LUFS) & exact SRT."""
    a1_raw, sr1 = _read_wav_mono(AUDIO_DIR / "act1_thesis.wav")
    a2_raw, sr2 = _read_wav_mono(AUDIO_DIR / "act2_filmroom.wav")
    a3_raw, sr3 = _read_wav_mono(AUDIO_DIR / "act3_results.wav")
    assert sr1 == sr2 == sr3 == SR, f"Unexpected sample rates: {sr1}, {sr2}, {sr3}"

    trim1, trim2, trim3 = 0.30, 0.30, 0.35
    a1 = a1_raw[int(trim1 * SR):]
    a2 = a2_raw[int(trim2 * SR):]
    a3 = a3_raw[int(trim3 * SR):]

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

    # Restrained documentary pulse ambient bed (D dorian warm pulse @ 96 BPM, ~25 dB below voice)
    t = np.arange(n_samples, dtype=np.float32) / SR
    beat_hz = 96.0 / 60.0
    bar_idx = ((t * beat_hz) / 4.0).astype(np.int32) % 4
    bass_freqs = np.where(bar_idx == 0, 73.42, np.where(bar_idx == 1, 58.27, np.where(bar_idx == 2, 65.41, 73.42)))
    fifth_freqs = bass_freqs * 1.5
    pad_freqs = np.where(bar_idx == 0, 220.0, np.where(bar_idx == 1, 174.61, np.where(bar_idx == 2, 196.0, 220.0)))

    phase_bass = 2.0 * math.pi * np.cumsum(bass_freqs) / SR
    phase_fifth = 2.0 * math.pi * np.cumsum(fifth_freqs) / SR
    phase_pad = 2.0 * math.pi * np.cumsum(pad_freqs) / SR

    beat_phase = (t * beat_hz) % 1.0
    pulse_env = 0.55 + 0.45 * np.exp(-beat_phase * 5.2) * (1.0 - np.exp(-beat_phase * 55.0))
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

    fi, fo = int(0.8 * SR), int(1.6 * SR)
    bed_l[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    bed_r[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    bed_l[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    bed_r[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)

    win = int(0.12 * SR)
    abs_v = np.abs(voice)
    cs = np.cumsum(np.insert(abs_v, 0, 0.0))
    env_v = (cs[win:] - cs[:-win]) / win
    env_v = np.pad(env_v, (win // 2, n_samples - len(env_v) - win // 2), mode="edge")
    duck = np.clip(1.0 - 4.5 * env_v, 0.38, 1.0).astype(np.float32)

    bed_gain = 0.055
    mix_l = voice + bed_l * bed_gain * duck
    mix_r = voice + bed_r * bed_gain * duck

    raw_mix_wav = work_dir / "mix_raw.wav"
    norm_mix_wav = work_dir / "mix_lufs14.wav"
    _write_wav_stereo(raw_mix_wav, mix_l, mix_r, SR)

    log_path = Path("/tmp/nfl_reel_v6_audio_ffmpeg.log")
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

    # Exact timestamped English SRT matched to Wyatt's 3 studio WAVs + crisp V6 updated metric callouts
    s1 = t_act1_start - trim1
    s2 = t_act2_start - trim2
    s3 = t_act3_start - trim3
    cues = [
        (max(0.15, 0.55 + s1), 4.85 + s1, "NFL scouting has trusted 40-yard stopwatches for decades,"),
        (5.25 + s1, 8.60 + s1, "yet 40 percent of top prospects now opt out."),
        (9.00 + s1, 13.35 + s1, "Beyond the Stopwatch unlocks hidden draft alpha"),
        (13.40 + s1, 18.85 + s1, "from 23 million real 10 Hz optical tracking frames across 510 rookies."),
        (0.50 + s2, 5.85 + s2, "Watch our 10 Hz optical film room compare fifth-round gem"),
        (6.00 + s2, 9.75 + s2, "Puka Nacua against second-round pick Ja'Lynn Polk."),
        (10.55 + s2, 16.60 + s2, "Even without a 40-yard time, Nacua explodes through gauntlet deceleration"),
        (16.75 + s2, 19.85 + s2, "at 9.59 yards per second (realized +$49.0M rookie EV),"),
        (20.30 + s2, 27.25 + s2, "the exact 0.3s burst and braking windows illuminated\nby our 4-position dual-head temporal attention encoders."),
        (0.55 + s3, 8.55 + s3, "Under strict zero-leakage CV across 22.6M game frames,\nGM Draft Surplus EV turns optical alpha into cap dollars,"),
        (8.95 + s3, 14.15 + s3, "lifting Sunday game translation R-squared from 0.1560"),
        (14.30 + s3, 19.65 + s3, "to 0.2398 (V2) and 0.2562 in V6 Positional Attention (Partial R² +0.1121),"),
        (19.90 + s3, 22.95 + s3, "reaching 0.2286 unseen-year LOYO (2.33x stopwatch),"),
        (23.30 + s3, min(total_sec - 0.10, 28.08 + s3), "0.5432 DL get-off, and 0.7045 DL/EDGE LOYO (q_FDR = 0.0012)."),
    ]

    def _fmt_ts(sec: float) -> str:
        ms = int(round(max(0.0, sec) * 1000))
        hh = ms // 3600000
        mm = (ms % 3600000) // 60000
        ss = (ms % 60000) // 1000
        rem = ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{rem:03d}"

    srt_blocks = [f"{idx}\n{_fmt_ts(st)} --> {_fmt_ts(en)}\n{text}\n" for idx, (st, en, text) in enumerate(cues, start=1)]
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

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
            "Style: Editorial,Helvetica Neue,33,&H00FFFFFF,&H00FFFFFF,&H1A111111,&H1A111111,-1,0,0,0,100,100,0.4,0,3,12,0,2,96,96,24,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )
    ass_lines = [ass_header]
    for st, en, text in cues:
        clean_t = text.replace("\n", "\\N")
        ass_lines.append(f"Dialogue: 0,{_fmt_ass(st)},{_fmt_ass(en)},Editorial,,0,0,0,,{clean_t}")
    ass_path = work_dir / "editorial_captions_v6.ass"
    ass_path.write_text("\n".join(ass_lines) + "\n", encoding="utf-8")

    timing: dict[str, float | str] = {
        "act1_start": t_act1_start,
        "act2_start": t_act2_start,
        "act3_start": t_act3_start,
        "total_sec": total_sec,
        "total_frames": float(total_frames),
        "ass_path": str(ass_path),
    }
    return norm_mix_wav, total_sec, timing


def _stamp_studio_chrome(img: Image.Image, kicker: str, stat_pill: str, accent: tuple[int, int, int] = C_GREEN) -> Image.Image:
    """Wrap any 1920x1080 Chromium screenshot with Economist top flag bar + clean 134px white bottom safe zone."""
    out = img.copy()
    draw = ImageDraw.Draw(out)

    # Crisp top broadcast ribbon (y=0..64)
    draw.rectangle([0, 0, W, 64], fill=C_WHITE)
    draw.rectangle([0, 0, W, 6], fill=C_INK)
    draw.rectangle([72, 0, 152, 10], fill=C_RED)

    f_kicker = _load_font(20, bold=True)
    f_pill = _load_font(20, bold=True)
    draw.text((72, 24), kicker.upper(), fill=C_RED, font=f_kicker)

    bbox = draw.textbbox((0, 0), stat_pill, font=f_pill)
    pw = bbox[2] - bbox[0] + 30
    ph = 34
    px0 = W - 72 - pw
    py0 = 17
    draw.rectangle([px0, py0, px0 + pw, py0 + ph], fill=accent)
    draw.text((px0 + 15, py0 + 5), stat_pill, fill=C_WHITE, font=f_pill)
    draw.line([(72, 63), (W - 72, 63)], fill=C_RULE, width=1)

    # Pure-white bottom caption safe zone (y=946..1080)
    draw.rectangle([0, 946, W, H], fill=C_WHITE)
    draw.line([(72, 946), (W - 72, 946)], fill=C_RULE, width=1)
    return out


def _compose_figure_slide(
    fig_png: Path,
    kicker: str,
    stat_pill: str,
    accent: tuple[int, int, int] = C_GREEN,
) -> Image.Image:
    """Place a 2x Retina pure-white figure onto a 1920x1080 studio frame with clean top bar & bottom subtitle zone."""
    canvas = Image.new("RGB", (W, H), C_WHITE)
    with Image.open(fig_png) as im:
        im_rgb = im.convert("RGB")
        max_w = W - 144
        max_h = 858
        scale = min(max_w / im_rgb.width, max_h / im_rgb.height)
        nw = int(round(im_rgb.width * scale))
        nh = int(round(im_rgb.height * scale))
        resized = im_rgb.resize((nw, nh), Image.Resampling.LANCZOS)
        ox = (W - nw) // 2
        oy = 74 + (max_h - nh) // 2
        canvas.paste(resized, (ox, oy))
    return _stamp_studio_chrome(canvas, kicker, stat_pill, accent=accent)


def _compose_title_card(is_closing: bool = False) -> Image.Image:
    """Crisp Economist/NYT pure-white V6 Opening / Closing card with exact 4-decimal headline metrics."""
    canvas = Image.new("RGB", (W, H), C_WHITE)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, W, 8], fill=C_INK)
    draw.rectangle([96, 0, 212, 16], fill=C_RED)

    f_eyebrow = _load_font(23, bold=True)
    f_title = _load_font(72, bold=True, serif=True)
    f_sub = _load_font(28, bold=False)
    f_num = _load_font(52, bold=True, serif=True)
    f_lbl = _load_font(19, bold=True)
    f_desc = _load_font(19, bold=False)
    f_meta = _load_font(21, bold=False)

    kicker = (
        "NFL BIG DATA BOWL 2027  ·  V6 CHAMPIONSHIP SUMMARY  ·  ZERO LEAKAGE"
        if is_closing
        else "NFL BIG DATA BOWL 2027  ·  OPEN TRACK  ·  100% OFFICIAL DATA & COLAB L4 GPU"
    )
    draw.text((96, 118), kicker, fill=C_RED, font=f_eyebrow)
    draw.text((96, 170), "Beyond the Stopwatch (V6)", fill=C_INK, font=f_title)
    draw.text(
        (96, 268),
        "4-Position 10 Hz Dual-Head Temporal Attention & GM Draft-Room Surplus EV Across 22.65M NFL Frames",
        fill=C_MUTED,
        font=f_sub,
    )
    draw.line([(96, 336), (W - 96, 336)], fill=C_INK, width=2)

    cards = [
        ("+0.2562", "OVERALL V6 OOF R²", "Up from 0.2398 (V2) & 0.1560 (SW)", "95% CI [+0.1954, +0.3150]", C_GREEN),
        ("+0.2286", "TEMPORAL LOYO R²", "2.33x Unseen-Year Lift vs SW", "ΔR² +0.0237 vs V2 (q=0.0264)", C_GREEN),
        ("+0.7045", "DL / EDGE LOYO R²", "Stratified OOF R² +0.6892", "Partial R² +0.3566 (q=0.0012)", C_RED),
        ("+0.1121", "PARTIAL R² VS DRAFT+SW", "Benjamini-Hochberg q = 0.0012", "Day-3 EV +$6.2M · Bust -20.4%", C_GOLD),
    ]
    cw = (W - 192 - 3 * 26) // 4
    for i, (num, title, d1, d2, color) in enumerate(cards):
        x0 = 96 + i * (cw + 26)
        y0 = 386
        draw.rectangle([x0, y0, x0 + cw, y0 + 270], fill=C_SOFT, outline=C_RULE, width=1)
        draw.rectangle([x0, y0, x0 + cw, y0 + 6], fill=color)
        draw.text((x0 + 24, y0 + 30), num, fill=color, font=f_num)
        draw.text((x0 + 24, y0 + 114), title, fill=C_INK, font=f_lbl)
        draw.text((x0 + 24, y0 + 154), d1, fill=C_INK, font=f_desc)
        draw.text((x0 + 24, y0 + 186), d2, fill=C_MUTED, font=f_desc)

    draw.line([(96, 710), (W - 96, 710)], fill=C_RULE, width=1)
    draw.text(
        (96, 744),
        "328 Colab NVIDIA L4 GPU Models · 4 Position Encoders (WR/TE, DB, DL/EDGE, OL) · 510-Prospect GM Surplus EV Engine",
        fill=C_INK,
        font=f_meta,
    )
    draw.text(
        (96, 786),
        "431,094 Combine 10 Hz Optical Frames · 22,654,290 Regular-Season NFL Game Frames · 6 Pure-White 2x Retina Figures",
        fill=C_MUTED,
        font=f_meta,
    )
    draw.line([(72, 946), (W - 72, 946)], fill=C_RULE, width=1)
    return canvas


def _add_progress_bar(img: Image.Image, frac: float) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    bar_w = int(round(W * max(0.0, min(1.0, frac))))
    if bar_w > 0:
        draw.rectangle([0, 0, bar_w, 5], fill=C_RED)
    return out


def _ken_burns(img: Image.Image, frac: float, zoom_start: float = 1.0, zoom_end: float = 1.026) -> Image.Image:
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


def _capture_v6_filmroom_live_assets() -> dict[str, list[Image.Image]]:
    """Capture LIVE 1920x1080 Chromium frames of the V6 Interactive Film-Room Studio:
    1) All 4 position-group 10 Hz Ghost Comparison + 0.1s Dual-Head Attention Glow matchups
    2) Live interactive GM Draft-Room Surplus EV Calculator slider drag & preset switches
    """
    assert HTML_FILMROOM.exists(), f"Missing {HTML_FILMROOM}"
    url = HTML_FILMROOM.resolve().as_uri()
    assets: dict[str, list[Image.Image]] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-mock-keychain", "--password-store=basic"],
        )
        try:
            context = browser.new_context(
                viewport={"width": W, "height": H},
                device_scale_factor=1,
            )
            page = context.new_page()
            page.goto(url, wait_until="load")
            page.evaluate("() => document.fonts ? document.fonts.ready : Promise.resolve()")

            # Broadcast-compact framing override so entire interactive studio fits inside y=66..938 above subtitle safe zone
            page.add_style_tag(
                content="""
                html, body { background: #FFFFFF !important; overflow: hidden !important; }
                .masthead { display: none !important; }
                .page-shell { max-width: 1760px !important; padding-inline: 48px !important; }
                #saliency-studio { padding-top: 72px !important; padding-bottom: 240px !important; border-bottom: none !important; }
                #saliency-studio .section-head { margin-bottom: 8px !important; display: flex !important; align-items: baseline !important; gap: 20px !important; }
                #saliency-studio h2 { font-size: 26px !important; }
                #saliency-studio .section-lead { margin: 0 !important; font-size: 15px !important; }
                .tablist-row { margin-bottom: 8px !important; }
                .matchup-tab { min-block-size: 36px !important; padding: 6px 14px !important; }
                .matchup-brief { margin-bottom: 8px !important; gap: 12px !important; }
                .player-strip { padding: 7px 12px !important; }
                .filmroom-grid { gap: 16px !important; }
                .canvas-panel { padding: 10px 12px !important; }
                .studio-canvas { max-height: 410px !important; }
                .control-bar { margin-top: 8px !important; padding: 7px 12px !important; }
                .telemetry-wrap { margin-top: 8px !important; }
                .data-table caption { padding: 5px 10px !important; }
                .data-table th, .data-table td { padding: 5px 10px !important; }

                #draft-ev { padding-top: 76px !important; padding-bottom: 240px !important; border-bottom: none !important; }
                #draft-ev .section-head { margin-bottom: 12px !important; display: flex !important; align-items: baseline !important; gap: 20px !important; }
                #draft-ev h2 { font-size: 28px !important; }
                #draft-ev .section-lead { margin: 0 !important; }
                .ev-workspace { gap: 28px !important; }
                .ev-controls-col { gap: 14px !important; }
                .ev-output-col { gap: 12px !important; }
                #draftEvCanvas { max-height: 320px !important; }
                .comp-card { padding: 10px 12px !important; }
                """
            )

            # --- PART A: Live 10 Hz Saliency Matchups (all 4 position groups) ---
            page.evaluate("() => { const el = document.getElementById('saliency-studio'); window.scrollTo({ top: el.offsetTop, behavior: 'instant' }); }")
            page.wait_for_timeout(120)

            matchup_specs = [
                (0, 64, "Act II · Live 10 Hz Ghost Replay & Dual-Head Attention (WR/TE)", "Puka Nacua Opt-Out · Peak 7.7-8.0s · +$49.0M EV", C_GREEN),
                (1, 24, "Act II · Live 10 Hz Dual-Head Attention (DB 90-Deg Break)", "Jaylon Jones vs Cam Smith · DB Partial R² +0.1236", C_GREEN),
                (2, 24, "Act II · Live 10 Hz Dual-Head Attention (DL/EDGE Pass Rush)", "Tuli Tuipulotu · DL/EDGE LOYO R² +0.7045", C_RED),
                (3, 22, "Act II · Live 10 Hz Dual-Head Attention (OL Pass-Pro Mirror)", "Mason McCormick (+$29.8M EV) vs Will Campbell", C_GOLD),
            ]

            tabs = page.locator('[role="tab"]')
            for m_idx, n_steps, kicker, pill, acc in matchup_specs:
                tabs.nth(m_idx).click()
                page.locator('input[name="attn-head"][value="aw"]').check()
                page.wait_for_timeout(60)
                max_idx = int(page.evaluate("() => parseInt(document.getElementById('frameScrubber').max, 10)"))
                indices = [int(round(i * max_idx / max(1, n_steps - 1))) for i in range(n_steps)]
                frames_m: list[Image.Image] = []
                for step_i, f_idx in enumerate(indices):
                    # On WR/TE Puka Nacua matchup, switch to Head 2 (Brake) during the final third to showcase dual-head glow
                    if m_idx == 0 and step_i == int(n_steps * 0.65):
                        page.locator('input[name="attn-head"][value="h2"]').check()
                    page.evaluate(
                        """(idx) => {
                            const s = document.getElementById('frameScrubber');
                            s.value = String(idx);
                            s.dispatchEvent(new Event('input', { bubbles: true }));
                        }""",
                        f_idx,
                    )
                    buf = page.screenshot(type="jpeg", quality=94)
                    im = Image.open(BytesIO(buf)).convert("RGB")
                    frames_m.append(_stamp_studio_chrome(im, kicker, pill, accent=acc))
                assets[f"matchup_{m_idx}"] = frames_m
                print(f"[PW V6] Matchup {m_idx}: captured {len(frames_m)} live 10 Hz Dual-Head Saliency frames (max_f={max_idx})")

            # --- PART B: Live GM Draft-Room Surplus EV Calculator Slider Drag & Preset Sweep ---
            page.evaluate("() => { const el = document.getElementById('draft-ev'); window.scrollTo({ top: el.offsetTop, behavior: 'instant' }); }")
            page.wait_for_timeout(120)

            ev_frames: list[Image.Image] = []
            # Sequence: (pos_group, pick, optical_z, badge_text, accent)
            ev_keyframes: list[tuple[str, int, float, str, tuple[int, int, int]]] = []
            # 1) Start at Puka Nacua exact comp (#177, +0.6σ -> 98% match +$49.0M) and sweep optical alpha +0.6 -> +2.0σ
            for z_val in np.linspace(0.6, 2.0, 14):
                ev_keyframes.append(("WR_TE", 177, round(float(z_val), 1), "WR/TE Pick #177 · Puka Nacua 98% Comp (+$49.0M)", C_GREEN))
            # 2) Switch to DL/EDGE & drag pick #120 -> #54 (Tuli Tuipulotu) at +1.4σ
            for pk_val in np.linspace(120, 54, 12):
                ev_keyframes.append(("DL_EDGE", int(round(pk_val)), 1.4, "DL/EDGE LOYO +0.7045 · Live Surplus EV Arbitrage", C_GREEN))
            # 3) Switch to DB Round 1 Trap warning (#24, sweeping +0.2σ -> -1.5σ -> crimson trap risk!)
            for z_val in np.linspace(0.2, -1.5, 14):
                ev_keyframes.append(("DB", 24, round(float(z_val), 1), "DB Pick #24 · Stopwatch Trap Risk (+45.5% Bust)", C_RED))

            for grp, pk, oz, pill, acc in ev_keyframes:
                page.locator(f'input[name="ev-pos-group"][value="{grp}"]').check()
                page.evaluate(
                    """([pk, oz]) => {
                        const p = document.getElementById('draftPickSlider');
                        const o = document.getElementById('opticalAlphaSlider');
                        p.value = String(pk);
                        p.dispatchEvent(new Event('input', { bubbles: true }));
                        o.value = String(oz);
                        o.dispatchEvent(new Event('input', { bubbles: true }));
                    }""",
                    [pk, oz],
                )
                buf = page.screenshot(type="jpeg", quality=94)
                im = Image.open(BytesIO(buf)).convert("RGB")
                ev_frames.append(
                    _stamp_studio_chrome(
                        im,
                        "Act III · Live GM Draft-Room 4-Yr Rookie Surplus EV Simulator",
                        pill,
                        accent=acc,
                    )
                )
            assets["draft_ev"] = ev_frames
            print(f"[PW V6] GM Draft EV Calculator: captured {len(ev_frames)} live interactive slider frames")
        finally:
            browser.close()

    return assets


def render_reel(out_mp4: Path = OUT_MP4, out_srt: Path = OUT_SRT) -> None:
    with tempfile.TemporaryDirectory(prefix="nfl_bdb2027_v6_reel_") as td:
        work_dir = Path(td)
        print("[1/4] Building -14 LUFS studio audio mix (ZONOS3-Mini Wyatt + documentary pulse bed) & V6 SRT ...")
        audio_wav, total_sec, timing = build_audio_and_srt(work_dir, out_srt)
        total_frames = int(round(total_sec * FPS))
        print(
            f"      Duration: {total_sec:.2f}s ({total_frames} frames @ {FPS}fps) | "
            f"Act1={float(timing['act1_start']):.2f}s, Act2={float(timing['act2_start']):.2f}s, Act3={float(timing['act3_start']):.2f}s"
        )

        print("[2/4] Capturing LIVE 1920x1080 Playwright Chromium V6 Film-Room Saliency & GM EV Simulator ...")
        live_assets = _capture_v6_filmroom_live_assets()

        print("[3/4] Composing V6 title/closing cards & all 6 pure-white 2x Retina figures -> ffmpeg ...")
        opening_slide = _compose_title_card(is_closing=False)
        closing_slide = _compose_title_card(is_closing=True)

        fig1_slide = _compose_figure_slide(
            FIGURES / "fig1_kinematic_transfer_matrix.png",
            "Act I · Combine-to-NFL Kinematic Transfer Matrix (V4 Verified)",
            "100% Coverage (510/510) · Partial R² +0.1188 (q=0.0012)",
            C_GREEN,
        )
        fig2_slide = _compose_figure_slide(
            FIGURES / "fig2_optical_velocity_decel_profiles.png",
            "Act I · Real 10 Hz Velocity, Braking & V4 Peak 0.3s Windows",
            "WR Peak 2.76-3.06s · Puka Nacua Brake -5.8 yd/s²",
            C_GREEN,
        )
        fig4_slide = _compose_figure_slide(
            FIGURES / "fig4_cv_model_ablation.png",
            "Act III · V4 Zero-Leakage Positional Ablation & BH-FDR Rigor",
            "OOF +0.2562 · LOYO +0.2286 · DL/EDGE +0.7045",
            C_GREEN,
        )
        fig6_slide = _compose_figure_slide(
            FIGURES / "fig6_temporal_attention_saliency_and_draft_ev.png",
            "Act III · New Figure 6: 0.1s Dual-Head Saliency & GM Surplus EV",
            "Day-3 EV Swing +$6.2M · Bust Risk -20.4% (n=510)",
            C_RED,
        )
        fig3_slide = _compose_figure_slide(
            FIGURES / "fig3_draft_capital_residual_alpha.png",
            "Act III · Draft-Capital Residual Alpha Across 510 Rookies",
            "33 Day-3 Gems (Pick #100+) · Avg +$24.9M 4-Yr EV",
            C_GOLD,
        )
        fig5_slide = _compose_figure_slide(
            FIGURES / "fig5_positional_biomechanical_archetypes.png",
            "Act III · Position-Specific Biomechanical Playbook (WR/DB/DL/OL/TE)",
            "DL/EDGE +0.6892 · DB Partial R² +0.1236 (q=0.0075)",
            C_GREEN,
        )

        a2 = float(timing["act2_start"])
        a3 = float(timing["act3_start"])
        t_end = total_sec

        def _seq(frames_list: list[Image.Image], lf: float) -> Image.Image:
            idx = min(len(frames_list) - 1, max(0, int(lf * len(frames_list))))
            return frames_list[idx]

        m0 = live_assets["matchup_0"]
        m1 = live_assets["matchup_1"]
        m2 = live_assets["matchup_2"]
        m3 = live_assets["matchup_3"]
        ev_seq = live_assets["draft_ev"]

        scenes: list[tuple[float, float, object]] = [
            # ACT 1: V6 Headline Breakthrough & Transfer/Velocity Figures (0.0 -> a2)
            (0.0, 6.4, lambda lf: _ken_burns(opening_slide, lf, 1.00, 1.022)),
            (6.4, 12.8, lambda lf: _ken_burns(fig1_slide, lf, 1.00, 1.024)),
            (12.8, a2, lambda lf: _ken_burns(fig2_slide, lf, 1.00, 1.024)),
            # ACT 2: Live 10 Hz Dual-Head Attention Ghost Replay across ALL 4 Position Matchups (a2 -> a3)
            (a2, a2 + 15.8, lambda lf: _seq(m0, lf)),          # WR/TE: Puka Nacua vs Ja'Lynn Polk (Combined -> Brake Head)
            (a2 + 15.8, a2 + 19.6, lambda lf: _seq(m1, lf)),   # DB: Jaylon Jones vs Cam Smith
            (a2 + 19.6, a2 + 23.4, lambda lf: _seq(m2, lf)),   # DL/EDGE: Tuli Tuipulotu vs Lukas Van Ness
            (a2 + 23.4, a3, lambda lf: _seq(m3, lf)),          # OL: Mason McCormick vs Will Campbell
            # ACT 3: Live GM Draft Surplus EV Slider Drag + Refreshed Fig4/Fig6/Fig3/Fig5 + V6 Closing Card (a3 -> t_end)
            (a3, a3 + 8.2, lambda lf: _seq(ev_seq, lf)),
            (a3 + 8.2, a3 + 13.8, lambda lf: _ken_burns(fig4_slide, lf, 1.00, 1.024)),
            (a3 + 13.8, a3 + 19.8, lambda lf: _ken_burns(fig6_slide, lf, 1.00, 1.024)),
            (a3 + 19.8, a3 + 22.3, lambda lf: _ken_burns(fig3_slide, lf, 1.00, 1.022)),
            (a3 + 22.3, a3 + 24.7, lambda lf: _ken_burns(fig5_slide, lf, 1.00, 1.022)),
            (a3 + 24.7, t_end, lambda lf: closing_slide),
        ]

        fade_sec = 0.35
        raw_mp4 = work_dir / "video_v6_nosub.mp4"
        ff_log = Path("/tmp/nfl_reel_v6_encode_ffmpeg.log")

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
                    sc_i = len(scenes) - 1
                    for idx, (st, en, _) in enumerate(scenes):
                        if st <= t_sec < en:
                            sc_i = idx
                            break
                    st, en, fn = scenes[sc_i]
                    lf = (t_sec - st) / max(1e-6, en - st)
                    frame_im: Image.Image = fn(lf)  # type: ignore[operator]

                    if sc_i > 0 and (t_sec - st) < fade_sec:
                        _, _, p_fn = scenes[sc_i - 1]
                        prev_im: Image.Image = p_fn(1.0)  # type: ignore[operator]
                        frame_im = _blend(prev_im, frame_im, (t_sec - st) / fade_sec)

                    frame_im = _add_progress_bar(frame_im, fi / max(1, total_frames - 1))
                    proc.stdin.write(frame_im.tobytes())
            finally:
                proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg raw video encode failed with rc={rc}; see {ff_log}")

        print("[4/4] Burning crisp 1080p Economist/NYT English subtitles into V6 final MP4 ...")
        ass_escaped = str(Path(str(timing["ass_path"])).resolve()).replace("\\", "\\\\").replace(":", "\\:")
        sub_log = Path("/tmp/nfl_reel_v6_sub_ffmpeg.log")
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

        print(f"[DONE V6] Rendered {out_mp4} ({out_mp4.stat().st_size:,} bytes) and {out_srt} ({out_srt.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render V6 1080p NFL BDB 2027 Film-Room Reel with subtitles.")
    parser.add_argument("--out-mp4", type=Path, default=OUT_MP4)
    parser.add_argument("--out-srt", type=Path, default=OUT_SRT)
    args = parser.parse_args()
    render_reel(out_mp4=args.out_mp4, out_srt=args.out_srt)


if __name__ == "__main__":
    main()
