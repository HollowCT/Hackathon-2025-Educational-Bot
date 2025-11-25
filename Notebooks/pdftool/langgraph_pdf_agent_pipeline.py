"""
File: langgraph_pdf_agent_pipeline.py

Simple 4-agent pipeline (LangGraph-style) to:
 1) receive a user's prompt,
 2) generate structured text (JSON) about the topic,
 3) generate images from the prompt,
 4) assemble a PDF from the JSON and images,
 5) return the PDF path to the caller.

This is a minimal, easy-to-read implementation suitable for local runs and testing.
It uses:
 - a tiny "agent" orchestration (LangGraph-like style, no external langgraph lib required)
 - OpenAI (optional) for better text generation if OPENAI_API_KEY is set
 - Pillow to create placeholder images (or can be swapped with a real image API)
 - PyMuPDF (fitz) to create the PDF

Install requirements:
    pip install openai Pillow pymupdf

Usage:
    python langgraph_pdf_agent_pipeline.py
"""

import os
import io
import json
import uuid
import tempfile
from typing import Dict, List, Any, Optional

# Third-party libs
try:
        import openai
except Exception:
        openai = None

from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF

# -------------------------------
# Configuration / Utilities
# -------------------------------

# If you want better LLM outputs, export OPENAI_API_KEY in your environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if openai and OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY

# Simple helper to ensure output dir
def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

# -------------------------------
# Minimal "LangGraph-like" agent base
# -------------------------------

class Agent:
        """
        Minimal agent base class. Each agent exposes a .run() method.
        This mimics the idea of LangGraph agents passing structured data.
        """
        def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
                raise NotImplementedError()

# -------------------------------
# Agent 1: Frontend / Receiver
# -------------------------------

class FrontendAgent(Agent):
        """
        Receives a user prompt and orchestrates the pipeline:
            - passes prompt to TextAgent
            - passes prompt to ImageAgent
            - passes JSON + images to PDFAgent
            - returns PDF path
        """
        def __init__(self, text_agent: Agent, image_agent: Agent, pdf_agent: Agent):
                self.text_agent = text_agent
                self.image_agent = image_agent
                self.pdf_agent = pdf_agent

        def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
                prompt = payload.get("prompt", "").strip()
                if not prompt:
                        raise ValueError("Empty prompt")

                # 1) Generate structured text JSON
                text_out = self.text_agent.run({"prompt": prompt})
                # text_out expected: {"json": {...}}

                # 2) Generate images (list of image file paths)
                image_out = self.image_agent.run({"prompt": prompt, "count": payload.get("image_count", 2)})
                # image_out expected: {"images": [paths]}

                # 3) Create PDF
                pdf_out = self.pdf_agent.run({
                        "json": text_out["json"],
                        "images": image_out["images"],
                        "title": payload.get("title", None)
                })
                # pdf_out expected: {"pdf_path": path}

                return {"pdf_path": pdf_out["pdf_path"]}

# -------------------------------
# Agent 2: Text generator -> JSON
# -------------------------------

class TextAgent(Agent):
        """
        Generates a JSON structure about the topic.
        Output JSON schema (example):
        {
            "title": "...",
            "summary": "...",
            "sections": [
                {"heading": "...", "content": "..."},
                ...
            ]
        }
        """

        def __init__(self, use_openai: bool = bool(OPENAI_API_KEY and openai)):
                self.use_openai = use_openai

        def _call_openai(self, prompt: str) -> str:
                """Simple OpenAI completion (fallback to a short prompt completion)."""
                if not openai:
                        raise RuntimeError("openai library not installed")
                # Use ChatCompletion if available, otherwise text completion
                try:
                        resp = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role":"user","content":prompt}],
                                max_tokens=600
                        )
                        return resp.choices[0].message.content.strip()
                except Exception:
                        # fallback to Completion
                        resp = openai.Completion.create(
                                model="text-davinci-003",
                                prompt=prompt,
                                max_tokens=600
                        )
                        return resp.choices[0].text.strip()

        def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
                user_prompt = payload.get("prompt", "")
                # Construct an instruction asking for JSON
                instruction = (
                        "Write a JSON object about the following topic. "
                        "Include keys: title (string), summary (string), sections (array of {heading, content}). "
                        "Make 3 sections. Output ONLY valid JSON.\n\nTopic: " + user_prompt
                )

                if self.use_openai:
                        try:
                                raw = self._call_openai(instruction)
                                # try to parse JSON
                                parsed = json.loads(raw)
                                return {"json": parsed}
                        except Exception:
                                # fallback to deterministic generator below
                                pass

                # Simple local deterministic generator (if OpenAI not available)
                title = user_prompt[:60].title()
                summary = f"This document covers: {user_prompt}."
                sections = []
                for i, heading in enumerate(["Overview", "Details", "Conclusion"], start=1):
                        content = f"{heading} about {user_prompt}. This is section {i}."
                        sections.append({"heading": heading, "content": content})
                result = {"title": title, "summary": summary, "sections": sections}
                return {"json": result}

# -------------------------------
# Agent 3: Image generator
# -------------------------------

class ImageAgent(Agent):
        """
        Generates images for the prompt. For simplicity, produces placeholder images
        (Pillow) with the prompt text. Replace or extend this agent to call a real
        image generation API (DALL·E, Stable Diffusion, etc.).
        """

        def __init__(self, out_dir: Optional[str] = None):
                self.out_dir = out_dir or tempfile.mkdtemp(prefix="images_")
                ensure_dir(self.out_dir)

        def _make_placeholder_image(self, text: str, path: str, size=(800, 600), bgcolor=(240, 240, 240)):
                img = Image.new("RGB", size, bgcolor)
                draw = ImageDraw.Draw(img)
                # Simple font fallback
                try:
                        font = ImageFont.truetype("DejaVuSans.ttf", 20)
                except Exception:
                        font = ImageFont.load_default()
                # Wrap text simply
                margin = 20
                lines = []
                words = text.split()
                line = ""
                for w in words:
                        if len(line + " " + w) > 40:
                                lines.append(line)
                                line = w
                        else:
                                line = (line + " " + w).strip()
                if line:
                        lines.append(line)
                y = margin
                for l in lines[:15]:
                        draw.text((margin, y), l, fill=(20, 20, 20), font=font)
                        y += 22
                img.save(path, format="PNG")

        def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
                prompt = payload.get("prompt", "")
                count = int(payload.get("count", 2))
                images = []
                for i in range(count):
                        filename = f"img_{uuid.uuid4().hex[:8]}.png"
                        path = os.path.join(self.out_dir, filename)
                        # For a real implementation, call an image API here using `prompt`
                        placeholder_text = f"{prompt} (image {i+1})"
                        self._make_placeholder_image(placeholder_text, path)
                        images.append(path)
                return {"images": images}

# -------------------------------
# Agent 4: PDF creator using PyMuPDF (fitz)
# -------------------------------

class PDFAgent(Agent):
        """
        Takes the JSON from TextAgent and images from ImageAgent and creates a PDF.
        The layout is simple:
            - cover page with title and summary
            - one page per section (heading + content)
            - images appended each on its own page
        """

        def __init__(self, out_dir: Optional[str] = None):
                self.out_dir = out_dir or tempfile.mkdtemp(prefix="pdfs_")
                ensure_dir(self.out_dir)

        def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
                data = payload.get("json", {})
                images = payload.get("images", [])
                title = payload.get("title") or data.get("title", "Document")
                summary = data.get("summary", "")

                pdf_path = os.path.join(self.out_dir, f"{uuid.uuid4().hex[:8]}.pdf")
                doc = fitz.open()

                # Cover page
                cover = doc.new_page(width=595, height=842)  # A4-ish in points
                title_block = title[:120]
                cover.insert_text((72, 100), title_block, fontsize=24, fontname="helv", rotate=0)
                cover.insert_text((72, 140), summary, fontsize=12, fontname="helv")

                # Sections
                for sec in data.get("sections", []):
                        page = doc.new_page(width=595, height=842)
                        heading = sec.get("heading", "")
                        content = sec.get("content", "")
                        page.insert_text((72, 72), heading, fontsize=18, fontname="helv")
                        # Simple text wrapping: PyMuPDF will wrap if you use insert_textbox
                        rect = fitz.Rect(72, 110, 523, 770)
                        page.insert_textbox(rect, content, fontsize=11, fontname="helv", align=0)

                # Images: add each image centered on its own page
                for img_path in images:
                        try:
                                img_doc = fitz.open(img_path)  # opens image as doc
                                r = img_doc[0].rect
                                page = doc.new_page(width=r.width, height=r.height)
                                page.insert_image(r, filename=img_path)
                        except Exception:
                                # fallback: create a blank page with small note
                                page = doc.new_page()
                                page.insert_text((72, 72), f"Image failed to load: {img_path}", fontsize=12)

                # Save PDF
                doc.save(pdf_path)
                doc.close()
                return {"pdf_path": pdf_path}

# -------------------------------
# Simple CLI demonstration
# -------------------------------

def run_demo():
        # Instantiate agents
        text_agent = TextAgent()
        image_agent = ImageAgent()
        pdf_agent = PDFAgent()
        frontend = FrontendAgent(text_agent, image_agent, pdf_agent)

        # Example user prompt (replace with any topic)
        user_prompt = "Sustainable home energy solutions and best practices"

        print("Running pipeline for prompt:", user_prompt)
        result = frontend.run({"prompt": user_prompt, "image_count": 2})
        pdf_path = result["pdf_path"]
        print("PDF generated at:", pdf_path)
        print("Open the PDF with your default viewer or inspect the file.")

if __name__ == "__main__":
        run_demo()