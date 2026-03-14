from __future__ import annotations

import os
from pathlib import Path

import fitz
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT / "tmp" / "pdfs" / "camoe_technical_doc"
OUTPUT_DIR = ROOT / "output" / "pdf"
SOURCE_PATH = TMP_DIR / "source.md"
PDF_PATH = OUTPUT_DIR / "CaMoE_v23_Technical_Documentation_zh.pdf"
ARCH_PATH = TMP_DIR / "architecture_overview.png"
FLOW_PATH = TMP_DIR / "training_and_settlement_flow.png"
RENDER_DIR = TMP_DIR / "rendered_pages"


def ensure_dirs() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)


def register_fonts() -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def find_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str, radius: int = 16) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=3)


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=6, align="center")


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleZH",
            parent=base["Title"],
            fontName="STSong-Light",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1C2840"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleZH",
            parent=base["Normal"],
            fontName="STSong-Light",
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4C5A76"),
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "Heading1ZH",
            parent=base["Heading1"],
            fontName="STSong-Light",
            fontSize=18,
            leading=24,
            textColor=colors.HexColor("#1C2840"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Heading2ZH",
            parent=base["Heading2"],
            fontName="STSong-Light",
            fontSize=14,
            leading=19,
            textColor=colors.HexColor("#243B63"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyZH",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=10.5,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            textColor=colors.HexColor("#1F2635"),
        ),
        "bullet": ParagraphStyle(
            "BulletZH",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=10.5,
            leading=15,
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "CaptionZH",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#5A6478"),
            spaceBefore=2,
            spaceAfter=10,
        ),
        "code": ParagraphStyle(
            "CodeStyle",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.2,
            leading=10.4,
            backColor=colors.HexColor("#F5F7FB"),
            borderPadding=6,
            borderColor=colors.HexColor("#D4DCE8"),
            borderWidth=0.5,
            borderRadius=3,
            spaceAfter=8,
        ),
    }


def build_architecture_diagram(path: Path) -> None:
    image = PILImage.new("RGB", (1600, 980), "#F4F7FB")
    draw = ImageDraw.Draw(image)
    title_font = find_font(42)
    box_font = find_font(24)
    small_font = find_font(20)
    draw.text((60, 40), "CaMoE v23 Runtime Architecture", font=title_font, fill="#1C2840")
    draw.text((60, 95), "Current code path only. No NOTE future vision.", font=small_font, fill="#4C5A76")

    def arrow(start: tuple[int, int], end: tuple[int, int]) -> None:
        draw.line([start, end], fill="#3A4A6B", width=5)

    input_box = (80, 180, 350, 280)
    emb_box = (80, 340, 350, 450)
    block_box = (420, 150, 1180, 760)
    head_box = (1250, 330, 1510, 450)
    seq_box = (470, 250, 1130, 400)
    ffn_box = (470, 430, 1130, 610)
    critic_box = (470, 640, 1130, 730)

    for box, fill, outline, text in [
        (input_box, "#FFFFFF", "#5A78A0", "Input IDs\n[B, T]"),
        (emb_box, "#FFFFFF", "#5A78A0", "Embedding\n[B, T, D]"),
        (block_box, "#FFFFFF", "#395886", ""),
        (seq_box, "#EAF2FF", "#5A78A0", "Sequence Market\nTimeMix vs Neuro-Symbolic ROSA\nRewardCritic + Top-1 route"),
        (ffn_box, "#EDF9F0", "#5B8A6A", "FFN Market\nRWKV / DeepEmbed / SlimDeepEmbed\nSparse dispatch when possible"),
        (critic_box, "#FFF6E8", "#B98532", "State per layer\nwallets (capital), q, prices, shared loss_ema"),
        (head_box, "#FFFFFF", "#5A78A0", "LN + LM Head\nLogits / Loss"),
    ]:
        draw_box(draw, box, fill, outline)
        if text:
            draw_centered(draw, box, text, box_font, "#182338")

    draw.text((460, 180), "CaMoE Block x n_layers", font=find_font(30), fill="#1C2840")
    arrow((350, 230), (420, 230))
    arrow((215, 280), (215, 340))
    arrow((1180, 390), (1250, 390))
    arrow((800, 400), (800, 430))
    arrow((800, 610), (800, 640))
    image.save(path)


def build_flow_diagram(path: Path) -> None:
    image = PILImage.new("RGB", (1600, 1180), "#F7F8FC")
    draw = ImageDraw.Draw(image)
    title_font = find_font(40)
    box_font = find_font(23)
    draw.text((60, 40), "Training And Settlement Flow", font=title_font, fill="#1C2840")

    boxes = [
        ((110, 170, 470, 280), "#FFFFFF", "#5A78A0", "1. Batch\ninput_ids, targets"),
        ((110, 340, 470, 490), "#EAF2FF", "#5A78A0", "2. Forward\nEmbedding -> Blocks -> logits -> loss"),
        ((560, 170, 950, 320), "#EDF9F0", "#5B8A6A", "3. Router\nprices, shares,\nexpected_profit,\nwinner"),
        ((560, 390, 950, 560), "#FFF6E8", "#B98532", "4. Settlement\nreward = sigmoid(improvement * scale)\nimprovement = (loss_ema - loss) / max(loss_ema, eps)\nprofit = shares * (reward - price)"),
        ((1040, 170, 1470, 320), "#F4EEFF", "#7A62AA", "5. RewardCritic\nwinner-position MSE supervision"),
        ((1040, 410, 1470, 560), "#FFF0F2", "#C16A78", "6. Logging\nwallet, price, entropy,\nexpected_profit, reward"),
        ((350, 650, 1230, 1010), "#FFFFFF", "#395886", "Current train.py schedule\n- phase label keeps uniform_warmup/full_market\n- actual forward uses uniform=False from step 0\n- exploration epsilon = 1.0 during warmup\n- epsilon decays to configured floor over market_ramp_steps\n- settlement and critic update run every step"),
    ]
    for box, fill, outline, text in boxes:
        draw_box(draw, box, fill, outline)
        draw_centered(draw, box, text, box_font, "#182338")
    draw.line([(290, 280), (290, 340)], fill="#3A4A6B", width=5)
    draw.line([(470, 240), (560, 240)], fill="#3A4A6B", width=5)
    draw.line([(755, 320), (755, 390)], fill="#3A4A6B", width=5)
    draw.line([(950, 240), (1040, 240)], fill="#3A4A6B", width=5)
    draw.line([(950, 475), (1040, 475)], fill="#3A4A6B", width=5)
    draw.line([(755, 560), (755, 650)], fill="#3A4A6B", width=5)
    image.save(path)


def parse_source(path: Path) -> list:
    st = styles()
    story: list = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_code = False
    code_lines: list[str] = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                story.append(Preformatted("\n".join(code_lines), st["code"]))
                in_code = False
                code_lines = []
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            story.append(Spacer(1, 2 * mm))
            continue
        if line == "[PAGEBREAK]":
            story.append(PageBreak())
            continue
        if line.startswith("[IMAGE:"):
            payload = line[len("[IMAGE:") : -1]
            image_name, caption = payload.split("|", 1)
            image_path = TMP_DIR / image_name
            height = 108 * mm if "architecture" in image_name else 130 * mm
            story.append(Image(str(image_path), width=177 * mm, height=height))
            story.append(Paragraph(caption, st["caption"]))
            continue
        if line.startswith("# "):
            story.append(Paragraph(line[2:].strip(), st["title"]))
            continue
        if line.startswith("## "):
            story.append(Paragraph(line[3:].strip(), st["h1"]))
            continue
        if line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), st["h2"]))
            continue
        if line.startswith("- "):
            story.append(Paragraph(f"• {line[2:].strip()}", st["bullet"]))
            continue
        if line.startswith("> "):
            story.append(Paragraph(line[2:].strip(), st["subtitle"]))
            continue
        story.append(Paragraph(line, st["body"]))
    return story


def draw_header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5A6478"))
    canvas.drawString(doc.leftMargin, A4[1] - 12 * mm, "CaMoE v23 Technical Documentation")
    canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="CaMoE v23 技术实现文档",
        author="OpenAI Codex",
    )
    doc.build(parse_source(SOURCE_PATH), onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)


def render_pdf_previews(page_limit: int = 4) -> None:
    for existing in RENDER_DIR.glob("page_*.png"):
        existing.unlink()
    doc = fitz.open(PDF_PATH)
    for page_index in range(min(len(doc), page_limit)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        pix.save(RENDER_DIR / f"page_{page_index + 1}.png")
    doc.close()


def main() -> None:
    ensure_dirs()
    register_fonts()
    build_architecture_diagram(ARCH_PATH)
    build_flow_diagram(FLOW_PATH)
    build_pdf()
    render_pdf_previews()
    print(PDF_PATH)


if __name__ == "__main__":
    main()
