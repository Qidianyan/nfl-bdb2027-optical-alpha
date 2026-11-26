"""Export all 6 Economist/NYT pure-white SVG figures to crisp 2x Retina PNGs via Playwright Chromium."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

FIGURES_DEFAULT = Path(__file__).resolve().parents[3] / "materials" / "figures"

SVG_FILES = [
    "fig1_kinematic_transfer_matrix.svg",
    "fig2_optical_velocity_decel_profiles.svg",
    "fig3_draft_capital_residual_alpha.svg",
    "fig4_model_ablation_lift.svg",
    "fig5_positional_biomechanical_archetypes.svg",
    "fig6_temporal_attention_saliency_and_draft_ev.svg",
]

PNG_ALIASES = {
    "fig4_model_ablation_lift.png": ["fig4_cv_model_ablation.png"],
}


def parse_svg_size(svg_text: str) -> tuple[int, int]:
    head = svg_text[:600]
    w_m = re.search(r'\bwidth="(\d+(?:\.\d+)?)"', head)
    h_m = re.search(r'\bheight="(\d+(?:\.\d+)?)"', head)
    if w_m and h_m:
        return int(round(float(w_m.group(1)))), int(round(float(h_m.group(1))))
    vb_m = re.search(r'\bviewBox="[^"]*?\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"', head)
    if vb_m:
        return int(round(float(vb_m.group(1)))), int(round(float(vb_m.group(2))))
    return 1180, 700


def export_all_svgs_to_png(figures_dir: Path = FIGURES_DEFAULT, scale: int = 2) -> list[Path]:
    exported: list[Path] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--use-mock-keychain", "--password-store=basic"],
        )
        context = browser.new_context(
            viewport={"width": 1400, "height": 1000},
            device_scale_factor=scale,
        )
        page = context.new_page()

        for svg_name in SVG_FILES:
            svg_path = figures_dir / svg_name
            if not svg_path.exists():
                raise FileNotFoundError(f"Missing SVG figure: {svg_path}")
            svg_text = svg_path.read_text(encoding="utf-8")
            w, h = parse_svg_size(svg_text)
            page.set_viewport_size({"width": w, "height": h})

            html_doc = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<style>"
                "html, body { margin: 0; padding: 0; background: #FFFFFF; overflow: hidden; }"
                "svg { display: block; width: " + str(w) + "px; height: " + str(h) + "px; }"
                "</style></head><body>"
                + svg_text
                + "</body></html>"
            )
            page.set_content(html_doc, wait_until="load")
            page.evaluate("() => document.fonts ? document.fonts.ready : Promise.resolve()")

            png_name = svg_path.with_suffix(".png").name
            png_path = figures_dir / png_name
            page.screenshot(path=str(png_path), full_page=False, type="png")

            with Image.open(png_path) as im:
                pw, ph = im.size
                assert pw == w * scale and ph == h * scale, (
                    f"{png_name}: expected {w * scale}x{h * scale}, got {pw}x{ph}"
                )
            exported.append(png_path)
            print(f"[PNG {scale}x] {png_name}: {w * scale}x{h * scale} ({png_path.stat().st_size:,} bytes)")

            for alias in PNG_ALIASES.get(png_name, []):
                alias_path = figures_dir / alias
                shutil.copy2(png_path, alias_path)
                exported.append(alias_path)
                print(f"[PNG {scale}x alias] {alias} -> copied from {png_name}")

        browser.close()
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all 5 BDB 2027 SVGs to 2x Retina PNGs.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DEFAULT)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    export_all_svgs_to_png(args.figures_dir, scale=args.scale)


if __name__ == "__main__":
    main()
