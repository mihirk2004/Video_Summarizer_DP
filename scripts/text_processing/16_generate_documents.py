#!/usr/bin/env python3
"""
Phase 4 — Step 6: Document Generation
Creates structured PDF/Markdown/HTML documents from multimodal inference results.

Input:
    - results/multimodal_inference.json (or path to inference results)

Output:
    - results/documents/{lecture_id}.md
    - results/documents/{lecture_id}.html
    - results/documents/{lecture_id}.pdf

Usage:
    python scripts/text_processing/16_generate_documents.py                          # All lectures
    python scripts/text_processing/16_generate_documents.py --lecture lecture_107     # Single lecture
    python scripts/text_processing/16_generate_documents.py --format markdown        # MD only
    python scripts/text_processing/16_generate_documents.py --format pdf             # PDF only
    python scripts/text_processing/16_generate_documents.py --format html            # HTML only

Requirements:
    pip install jinja2 markdown
    Optional: pandoc (for PDF generation)
"""
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config_loader import config_loader
from scripts.utils.logger import setup_logger


class DocumentGenerator:
    """Generate PDF / Markdown / HTML documents from multimodal inference results"""

    def __init__(self, config: Dict, lecture_filter: str = None,
                 format_filter: str = None):
        self.config = config
        self.lecture_filter = lecture_filter
        self.logger = setup_logger("document_gen")

        # Config
        doc_cfg = config.get('document', {})
        self.output_dir = Path(doc_cfg.get('output_dir', 'results/documents'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir = Path(doc_cfg.get('template_dir', 'templates'))
        self.max_img_width = doc_cfg.get('max_image_width', 600)
        self.include_timestamps = doc_cfg.get('include_timestamps', True)
        self.mathjax_cdn = doc_cfg.get(
            'mathjax_cdn',
            'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js')

        formats = doc_cfg.get('formats', ['pdf', 'markdown', 'html'])
        if format_filter:
            self.formats = [format_filter]
        else:
            self.formats = formats

        # Inference results path
        self.results_path = Path(config['paths']['outputs'].get(
            'results', 'results')) / "multimodal_inference.json"

    def _load_results(self) -> Dict[str, List[Dict]]:
        """Load inference results and group by lecture"""
        self.logger.info(f"Loading results from {self.results_path} ...")

        with open(self.results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data['segments']

        # Group by lecture
        lectures = defaultdict(list)
        for seg in segments:
            lectures[seg['lecture_id']].append(seg)

        # Sort segments within each lecture by start time
        for lec_id in lectures:
            lectures[lec_id].sort(key=lambda s: s.get('start', 0))

        if self.lecture_filter:
            if self.lecture_filter in lectures:
                lectures = {self.lecture_filter: lectures[self.lecture_filter]}
            else:
                self.logger.warning(
                    f"Lecture {self.lecture_filter} not found in results!")
                return {}

        self.logger.info(f"  {len(lectures)} lectures to process")
        return dict(lectures)

    # ------------------------------------------------------------------
    # Markdown Generation
    # ------------------------------------------------------------------
    def _generate_markdown(self, lecture_id: str, segments: List[Dict]) -> str:
        """Generate Markdown document for a lecture"""
        lines = []

        # Header
        lines.append(f"# Lecture: {lecture_id.replace('_', ' ').title()}\n")

        # Metadata
        total_duration = max((s.get('end', 0) for s in segments), default=0)
        n_visual = sum(1 for s in segments if s.get('has_visuals'))
        duration_str = f"{int(total_duration // 60)}:{int(total_duration % 60):02d}"

        lines.append(f"> **Duration**: {duration_str} | "
                      f"**Segments**: {len(segments)} | "
                      f"**Visual Elements**: {n_visual}\n")

        # Table of Contents
        lines.append("## Table of Contents\n")
        for i, seg in enumerate(segments, 1):
            start = self._format_time(seg.get('start', 0))
            end = self._format_time(seg.get('end', 0))
            time_str = f" ({start}–{end})" if self.include_timestamps else ""
            lines.append(f"{i}. [Segment {i}{time_str}](#segment-{i})")
        lines.append("")

        # Segments
        for i, seg in enumerate(segments, 1):
            start = self._format_time(seg.get('start', 0))
            end = self._format_time(seg.get('end', 0))
            time_str = f" ({start}–{end})" if self.include_timestamps else ""

            lines.append(f"---\n")
            lines.append(f"## Segment {i}{time_str} {{#segment-{i}}}\n")

            # Summary
            summary = seg.get('final_summary', seg.get('fusion_summary', ''))
            if summary:
                lines.append(f"{summary}\n")

            # Visual elements
            if seg.get('has_visuals'):
                captions = seg.get('frame_captions', [])
                image_paths = seg.get('image_paths', [])
                categories = seg.get('image_categories', [])
                timestamps = seg.get('image_timestamps', [])

                for j, img_path in enumerate(image_paths):
                    cat = categories[j] if j < len(categories) else "Visual"
                    ts = timestamps[j] if j < len(timestamps) else 0
                    ts_str = self._format_time(ts)

                    lines.append(f"\n### {cat.replace('_', ' ')} (at {ts_str})\n")

                    # Image embed (relative path)
                    rel_path = self._copy_frame(img_path, lecture_id)
                    if rel_path:
                        lines.append(f"![{cat}]({rel_path})\n")

                    # Caption
                    if j < len(captions) and captions[j].get('caption'):
                        lines.append(f"> {captions[j]['caption']}\n")

                    # LaTeX for equations
                    if cat == "Equation" and j < len(captions):
                        latex = captions[j].get('latex', '')
                        if latex:
                            lines.append(f"\n$${latex}$$\n")

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML Generation
    # ------------------------------------------------------------------
    def _generate_html(self, lecture_id: str, segments: List[Dict],
                        markdown_content: str) -> str:
        """Generate HTML document with MathJax and responsive layout"""

        # Try Jinja2 template first
        template_path = self.template_dir / "lecture_document.html"
        if template_path.exists():
            return self._render_jinja_template(lecture_id, segments, template_path)

        # Fallback: convert markdown to HTML
        try:
            import markdown
            body = markdown.markdown(
                markdown_content,
                extensions=['tables', 'fenced_code', 'toc'])
        except ImportError:
            body = f"<pre>{markdown_content}</pre>"

        total_duration = max((s.get('end', 0) for s in segments), default=0)
        duration_str = f"{int(total_duration // 60)}:{int(total_duration % 60):02d}"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{lecture_id.replace('_', ' ').title()} — Lecture Notes</title>
    <script src="{self.mathjax_cdn}" async></script>
    <style>
        :root {{
            --bg: #0f172a; --surface: #1e293b; --text: #e2e8f0;
            --accent: #38bdf8; --accent2: #818cf8; --border: #334155;
            --code-bg: #0d1117;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            background: var(--bg); color: var(--text);
            line-height: 1.7; max-width: 900px; margin: 0 auto;
            padding: 2rem 1.5rem;
        }}
        h1 {{
            font-size: 2rem; font-weight: 700;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        h2 {{ font-size: 1.4rem; color: var(--accent); margin: 2rem 0 0.5rem; }}
        h3 {{ font-size: 1.1rem; color: var(--accent2); margin: 1rem 0 0.3rem; }}
        .meta {{
            background: var(--surface); border-radius: 8px;
            padding: 1rem; margin: 1rem 0; border-left: 3px solid var(--accent);
            font-size: 0.9rem;
        }}
        .segment {{
            background: var(--surface); border-radius: 12px;
            padding: 1.5rem; margin: 1.5rem 0;
            border: 1px solid var(--border);
            transition: border-color 0.3s;
        }}
        .segment:hover {{ border-color: var(--accent); }}
        .segment-header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.5rem;
        }}
        .time-badge {{
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            color: white; padding: 0.2rem 0.6rem;
            border-radius: 20px; font-size: 0.8rem; font-weight: 600;
        }}
        .visual-tag {{
            display: inline-block; background: var(--code-bg);
            color: var(--accent2); padding: 0.15rem 0.5rem;
            border-radius: 4px; font-size: 0.75rem; margin: 0.1rem;
        }}
        .visual-block {{
            background: var(--code-bg); border-radius: 8px;
            padding: 1rem; margin: 0.8rem 0;
            border: 1px solid var(--border);
        }}
        .visual-block img {{
            max-width: 100%; border-radius: 6px;
            margin: 0.5rem 0;
        }}
        blockquote {{
            border-left: 3px solid var(--accent2);
            padding: 0.5rem 1rem; margin: 0.5rem 0;
            background: rgba(129, 140, 248, 0.1);
            border-radius: 0 6px 6px 0; font-style: italic;
        }}
        .toc {{
            background: var(--surface); border-radius: 8px;
            padding: 1rem 1.5rem; margin: 1rem 0;
        }}
        .toc ol {{ padding-left: 1.5rem; }}
        .toc li {{ margin: 0.3rem 0; }}
        .toc a {{ color: var(--accent); text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
        a {{ color: var(--accent); }}
        hr {{ border: none; border-top: 1px solid var(--border); margin: 2rem 0; }}
        .footer {{
            text-align: center; padding: 2rem 0; color: #64748b;
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
    <h1>{lecture_id.replace('_', ' ').title()}</h1>
    <div class="meta">
        <strong>Duration:</strong> {duration_str} |
        <strong>Segments:</strong> {len(segments)} |
        <strong>Visual Elements:</strong> {sum(1 for s in segments if s.get('has_visuals'))}
    </div>

    <div class="toc">
        <h2>Table of Contents</h2>
        <ol>
"""
        # TOC
        for i, seg in enumerate(segments, 1):
            start = self._format_time(seg.get('start', 0))
            end = self._format_time(seg.get('end', 0))
            html += f'            <li><a href="#seg{i}">Segment {i} ({start}–{end})</a></li>\n'

        html += """        </ol>
    </div>
    <hr>
"""
        # Segments
        for i, seg in enumerate(segments, 1):
            start = self._format_time(seg.get('start', 0))
            end = self._format_time(seg.get('end', 0))
            summary = seg.get('final_summary', '')
            tags = seg.get('visual_tags', [])

            html += f"""
    <div class="segment" id="seg{i}">
        <div class="segment-header">
            <h2>Segment {i}</h2>
            <span class="time-badge">{start} – {end}</span>
        </div>
"""
            if tags:
                html += '        <div style="margin-bottom: 0.5rem;">'
                for tag in tags:
                    html += f'<span class="visual-tag">{tag.replace("_", " ")}</span> '
                html += '</div>\n'

            html += f'        <p>{summary}</p>\n'

            # Visual elements
            if seg.get('has_visuals'):
                for j, img_path in enumerate(seg.get('image_paths', [])):
                    cat = seg['image_categories'][j] if j < len(seg.get('image_categories', [])) else ''
                    ts = seg['image_timestamps'][j] if j < len(seg.get('image_timestamps', [])) else 0

                    rel_path = self._copy_frame(img_path, lecture_id)
                    captions = seg.get('frame_captions', [])
                    caption = captions[j].get('caption', '') if j < len(captions) else ''

                    html += f"""
        <div class="visual-block">
            <h3>{cat.replace('_', ' ')} (at {self._format_time(ts)})</h3>
"""
                    if rel_path:
                        html += f'            <img src="{rel_path}" alt="{cat}" loading="lazy">\n'
                    if caption:
                        html += f'            <blockquote>{caption}</blockquote>\n'
                    html += '        </div>\n'

            html += '    </div>\n'

        html += """
    <hr>
    <div class="footer">
        Generated by Multimodal Lecture Summarizer — Phase 4 Pipeline
    </div>
</body>
</html>"""
        return html

    def _render_jinja_template(self, lecture_id: str, segments: List[Dict],
                                 template_path: Path) -> str:
        """Render using Jinja2 template"""
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(str(template_path.parent)))
        template = env.get_template(template_path.name)

        total_duration = max((s.get('end', 0) for s in segments), default=0)

        return template.render(
            lecture_id=lecture_id,
            lecture_title=lecture_id.replace('_', ' ').title(),
            segments=segments,
            total_duration=self._format_time(total_duration),
            n_segments=len(segments),
            n_visual=sum(1 for s in segments if s.get('has_visuals')),
            mathjax_cdn=self.mathjax_cdn,
            format_time=self._format_time,
        )

    # ------------------------------------------------------------------
    # PDF Generation
    # ------------------------------------------------------------------
    def _generate_pdf(self, lecture_id: str, md_path: Path) -> Optional[Path]:
        """Convert Markdown to PDF using Pandoc"""
        if not shutil.which('pandoc'):
            self.logger.warning(
                "Pandoc not found! Install it for PDF generation: "
                "https://pandoc.org/installing.html")
            return None

        pdf_path = self.output_dir / lecture_id / f"{lecture_id}.pdf"

        cmd = [
            'pandoc', str(md_path),
            '-o', str(pdf_path),
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt',
            '-V', 'colorlinks=true',
            '--highlight-style=tango',
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return pdf_path
            else:
                self.logger.warning(f"Pandoc failed: {result.stderr[:200]}")
                # Try without xelatex
                cmd[cmd.index('--pdf-engine=xelatex')] = '--pdf-engine=pdflatex'
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    return pdf_path
                self.logger.warning(f"Pandoc fallback also failed: {result.stderr[:200]}")
                return None
        except subprocess.TimeoutExpired:
            self.logger.warning("Pandoc timed out")
            return None
        except FileNotFoundError:
            self.logger.warning("Pandoc not found")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to M:SS"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}:{s:02d}"

    def _copy_frame(self, img_path: str, lecture_id: str) -> Optional[str]:
        """Copy frame to output directory and return relative path"""
        src = Path(img_path)
        if not src.exists():
            return None

        dest_dir = self.output_dir / lecture_id / "frames"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        if not dest.exists():
            shutil.copy2(str(src), str(dest))

        return f"frames/{src.name}"

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def generate(self):
        self.logger.info("=" * 60)
        self.logger.info("Document Generation Pipeline")
        self.logger.info("=" * 60)

        lectures = self._load_results()
        if not lectures:
            self.logger.error("No results to process!")
            return

        generated = {'markdown': 0, 'html': 0, 'pdf': 0}

        for lecture_id, segments in tqdm(lectures.items(), desc="Generating docs"):
            lec_dir = self.output_dir / lecture_id
            lec_dir.mkdir(parents=True, exist_ok=True)

            # Markdown
            if 'markdown' in self.formats or 'pdf' in self.formats:
                md_content = self._generate_markdown(lecture_id, segments)
                md_path = lec_dir / f"{lecture_id}.md"
                md_path.write_text(md_content, encoding='utf-8')
                generated['markdown'] += 1

            # HTML
            if 'html' in self.formats:
                html_content = self._generate_html(
                    lecture_id, segments, md_content if 'md_content' in dir() else '')
                html_path = lec_dir / f"{lecture_id}.html"
                html_path.write_text(html_content, encoding='utf-8')
                generated['html'] += 1

            # PDF
            if 'pdf' in self.formats:
                md_path = lec_dir / f"{lecture_id}.md"
                if md_path.exists():
                    pdf_path = self._generate_pdf(lecture_id, md_path)
                    if pdf_path:
                        generated['pdf'] += 1

        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"Document Generation Complete!")
        self.logger.info(f"  Markdown: {generated['markdown']}")
        self.logger.info(f"  HTML:     {generated['html']}")
        self.logger.info(f"  PDF:      {generated['pdf']}")
        self.logger.info(f"  Output:   {self.output_dir}")
        self.logger.info(f"{'=' * 50}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4: Document Generation")
    parser.add_argument("--lecture", type=str, default=None,
                        help="Generate for single lecture")
    parser.add_argument("--format", type=str, default=None,
                        choices=['markdown', 'html', 'pdf'],
                        help="Generate only this format")
    args = parser.parse_args()

    config = config_loader.load_all()
    generator = DocumentGenerator(
        config, lecture_filter=args.lecture, format_filter=args.format)
    generator.generate()


if __name__ == "__main__":
    main()
