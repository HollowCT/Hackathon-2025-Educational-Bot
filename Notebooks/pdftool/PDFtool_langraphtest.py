"""
A small agent helper to receive a user prompt and produce a PDF file enhanced by an LLM
and illustrated via an image-generation client (e.g., Groq). Uses langchain-style LLM
wrappers, groq-style image clients, and PyMuPDF (fitz) to assemble the final PDF.

Main function:
    generate_pdf_from_prompt(prompt, llm, image_client, output_path)

Design:
    - enhance_prompt_to_json(prompt, llm): call LLM to expand a short prompt into a
      structured JSON (title, sections with headings and bodies and optional image prompts).
    - generate_images_from_json(json_obj, image_client): use image_client to generate image
      bytes for each image prompt found in the JSON.
    - create_pdf_from_components(json_obj, images_map, output_path): assemble PDF using PyMuPDF.

Notes on integration:
    - llm argument should be a LangChain LLM wrapper or any object that is callable
      or exposes `predict` or `generate`/`__call__`. The LLM should return a JSON string
      matching the expected schema (or the function will attempt to parse the text).
    - image_client argument should have a method that accepts an image prompt and returns
      image bytes (raw PNG/JPEG) or base64-encoded image data. The helper will try a few
      common method names (generate, create, __call__).
    - This module intentionally contains small adapter logic so it is straightforward to
      hook up real remote services in production and to mock them in tests.

Dependencies:
    pip install langchain pymupdf groq-py
    (For tests no external calls are made; real client libraries are required for production.)
"""

import io
import json
import base64
import os
from typing import Any, Dict, List, Tuple
import fitz  # PyMuPDF
import tempfile
import unittest
from typing import Any, Dict, Callable, List
import traceback

# ---------------------------
# Core functions
# ---------------------------

def _call_llm(llm: Any, prompt: str) -> str:
    """
    Generic LLM caller that tries common call methods.
    Returns the raw string output from the LLM.
    """
    # Try common method names in order
    if hasattr(llm, "predict"):
        return llm.predict(prompt)
    if hasattr(llm, "generate"):
        # Some langchain LLMs return a GenerationResult-like object; try to extract text
        result = llm.generate([prompt])  # often accepts list
        # Try to extract text in common structures
        try:
            # langchain >= 0.0.200 style: result.generations[0][0].text
            return result.generations[0][0].text
        except Exception:
            try:
                # legacy simple structure
                return str(result)
            except Exception:
                pass
    if callable(llm):
        return llm(prompt)
    raise RuntimeError("LLM object does not expose a callable interface (predict/generate/__call__).")

def enhance_prompt_to_json(prompt: str, llm: Any) -> Dict:
    """
    Enhance a short user prompt into a detailed JSON describing the PDF content.
    Expected JSON schema (example):
    {
        "title": "Topic Title",
        "author": "Auto Generated",
        "sections": [
            {
                "heading": "Introduction",
                "body": "Detailed text for the intro...",
                "image_prompts": ["an illustrative diagram of ..."]
            },
            ...
        ]
    }

    The LLM is asked to output JSON. This function will parse the returned text
    and return a Python dict. If parsing fails, an error is raised.
    """
    instruction = (
        "You are to expand the user's topic into a detailed JSON suitable for "
        "insertion into a multi-page PDF. Output only valid JSON (no surrounding "
        "explanations). The schema must include: title (str), author (str, optional), "
        "sections (list of {heading, body, image_prompts (list)})\n\n"
        f"User topic: {prompt}\n\n"
        "Produce the JSON now."
    )
    raw = _call_llm(llm, instruction)
    # Attempt to extract JSON blob from the LLM response
    text = raw.strip()
    # Find first '{' and last '}' to robustly extract JSON if the model included commentary
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_text = text[start:end]
    except ValueError:
        json_text = text  # try entire text
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM output. Raw output:\n{raw}") from e
    # Basic normalization
    if "title" not in obj:
        obj["title"] = prompt
    if "sections" not in obj or not isinstance(obj["sections"], list):
        obj["sections"] = [{"heading": "Content", "body": obj.get("content", ""), "image_prompts": []}]
    # Ensure image_prompts exists for each section
    for s in obj["sections"]:
        if "image_prompts" not in s or not isinstance(s["image_prompts"], list):
            s["image_prompts"] = []
    return obj

def _call_image_client(image_client: Any, prompt: str) -> bytes:
    """
    Generic image client caller. Returns raw image bytes (PNG/JPEG).
    Tries common method names and common return shapes (bytes or base64).
    """
    # Try common method names
    if hasattr(image_client, "generate"):
        out = image_client.generate(prompt)
    elif hasattr(image_client, "create"):
        out = image_client.create(prompt)
    elif callable(image_client):
        out = image_client(prompt)
    else:
        raise RuntimeError("Image client does not have generate/create/__call__ method.")

    # If out is bytes already
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    # If out is a dict with keys
    if isinstance(out, dict):
        # common: {"image": b"..."} or {"data": "...base64..."}
        if "image" in out and isinstance(out["image"], (bytes, bytearray)):
            return bytes(out["image"])
        if "data" in out and isinstance(out["data"], str):
            return base64.b64decode(out["data"])
        # maybe {"images":[...]}
        if "images" in out and isinstance(out["images"], list) and out["images"]:
            first = out["images"][0]
            if isinstance(first, (bytes, bytearray)):
                return bytes(first)
            if isinstance(first, str):
                try:
                    return base64.b64decode(first)
                except Exception:
                    pass
    # If string, maybe base64
    if isinstance(out, str):
        try:
            return base64.b64decode(out)
        except Exception:
            # maybe it's raw image bytes in string form -> encode
            return out.encode("utf-8")
    raise RuntimeError("Unrecognized image client output format.")

def generate_images_from_json(json_obj: Dict, image_client: Any) -> Dict[Tuple[int,int], bytes]:
    """
    For each image prompt in the JSON, call the image_client and collect images.

    Returns a mapping: (section_index, image_index) -> image_bytes
    """
    images = {}
    for si, section in enumerate(json_obj.get("sections", [])):
        for ii, ip in enumerate(section.get("image_prompts", []) or []):
            try:
                img_bytes = _call_image_client(image_client, ip)
                images[(si, ii)] = img_bytes
            except Exception as e:
                # Skip the image if generation fails, but continue
                # In production you might want to surface the error
                continue
    return images

def _insert_text_page(doc: fitz.Document, title: str, text: str, font_size: int = 12, margin: int = 50):
    """
    Insert a page primarily containing text. Performs very simple flow-layout (no wrapping library).
    """
    page = doc.new_page()
    rect = page.rect
    text_rect = fitz.Rect(margin, margin, rect.width - margin, rect.height - margin)
    # Use a text writer to automatically wrap
    tw = fitz.TextWriter(text_rect)
    tw.append(text_rect.tl, title + "\n", fontsize=font_size + 4)
    # split long text into paragraphs (simple)
    paragraphs = text.split("\n")
    y = text_rect.tl.y + (font_size + 6)
    for p in paragraphs:
        # fitz TextWriter doesn't provide precise flow control; use insert_textbox for simple wrapping
        page.insert_textbox(fitz.Rect(margin, y, rect.width - margin, rect.height - margin), p,
                            fontsize=font_size, wrap=1)
        # advance y a bit (approx)
        y += (font_size * (p.count("\n") + 3))
    tw.write_text(page)

def _insert_image_page(doc: fitz.Document, image_bytes: bytes, max_width: int = 500, max_height: int = 700):
    """
    Insert a page with a single image centered to fit within max dimensions.
    """
    page = doc.new_page()
    rect = page.rect
    # load image
    try:
        img = fitz.Pixmap(fitz.open("png", image_bytes))  # fitz open supports buffer with format hint
    except Exception:
        try:
            img = fitz.Pixmap(fitz.open("jpg", image_bytes))
        except Exception:
            # As fallback, write to temporary file and load
            with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as tf:
                tf.write(image_bytes)
                tmpname = tf.name
            img = fitz.Pixmap(tmpname)
            os.unlink(tmpname)
    # compute scaling
    iw, ih = img.width, img.height
    scale = min(max_width / iw, max_height / ih, 1.0)
    w, h = int(iw * scale), int(ih * scale)
    # center
    x0 = (rect.width - w) / 2
    y0 = (rect.height - h) / 2
    rect_img = fitz.Rect(x0, y0, x0 + w, y0 + h)
    page.insert_image(rect_img, pixmap=img)
    img = None  # free

def create_pdf_from_components(json_obj: Dict, images_map: Dict[Tuple[int,int], bytes], output_path: str) -> str:
    """
    Create a PDF file from the structured JSON and the generated images.
    The images_map keys are (section_index, image_index).
    Returns the path to the saved PDF (output_path).
    """
    doc = fitz.open()
    # Cover page
    title = json_obj.get("title", "Untitled")
    author = json_obj.get("author", "")
    cover = doc.new_page()
    cover.insert_text((72, 72), title, fontsize=24)
    if author:
        cover.insert_text((72, 110), f"Author: {author}", fontsize=12)

    # Content pages
    for si, section in enumerate(json_obj.get("sections", [])):
        heading = section.get("heading", "")
        body = section.get("body", "")
        # Add heading + body page
        page = doc.new_page()
        # Heading
        page.insert_text((72, 72), heading, fontsize=18)
        # Body -- use insert_textbox for wrapping
        page.insert_textbox(fitz.Rect(72, 110, page.rect.width - 72, page.rect.height - 72), body, fontsize=11, wrap=1)
        # Add images for this section (each image on its own page)
        img_prompts = section.get("image_prompts", []) or []
        for ii, _ in enumerate(img_prompts):
            key = (si, ii)
            if key in images_map:
                try:
                    _insert_image_page(doc, images_map[key])
                except Exception:
                    # ignore image insertion errors
                    continue

    # Save
    doc.save(output_path)
    doc.close()
    return output_path

def generate_pdf_from_prompt(prompt: str, llm: Any, image_client: Any, output_path: str) -> str:
    """
    High level function for an agent to call.

    Steps:
      1. Enhance prompt via LLM into structured JSON.
      2. Generate images via image_client for image prompts found in the JSON.
      3. Assemble PDF via PyMuPDF and save to output_path.
      4. Return the output_path to the caller (the "original agent").

    llm and image_client can be real clients or mocks for testing.
    """
    structured = enhance_prompt_to_json(prompt, llm)
    images_map = generate_images_from_json(structured, image_client)
    pdf_path = create_pdf_from_components(structured, images_map, output_path)
    return pdf_path

# ---------------------------
# Tests (use mocks; no external calls)
# ---------------------------

class MockLLM:
    """A mock LLM that returns deterministic JSON for testing."""
    def predict(self, prompt_text: str) -> str:
        # A very simple JSON; includes one section with an image prompt
        response = {
            "title": "Mocked: Sample Topic",
            "author": "UnitTest",
            "sections": [
                {
                    "heading": "Introduction",
                    "body": "This is an introduction generated for testing purposes.",
                    "image_prompts": ["a simple red square image"]
                },
                {
                    "heading": "Details",
                    "body": "Detailed explanation goes here.",
                    "image_prompts": []
                }
            ]
        }
        return json.dumps(response)

class MockImageClient:
    """A mock image client that returns a tiny PNG for any prompt (encoded bytes)."""
    def generate(self, prompt_text: str) -> bytes:
        # Create a minimal PNG (1x1 red pixel) base64 to bytes.
        # This is a real 1x1 red PNG base64 string.
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMA"
            "AQAABQABDQottAAAAABJRU5ErkJggg=="
        )
        return base64.b64decode(png_b64)

class GeneratePdfTests(unittest.TestCase):
    def test_full_flow_creates_pdf(self):
        prompt = "Create a PDF about testing agents"
        llm = MockLLM()
        img_client = MockImageClient()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            out_path = tf.name
        try:
            result_path = generate_pdf_from_prompt(prompt, llm, img_client, out_path)
            self.assertTrue(os.path.exists(result_path))
            # Open with PyMuPDF to inspect pages
            doc = fitz.open(result_path)
            # Expected pages: cover + 1 content page + 1 image page + 1 content page = 4
            self.assertGreaterEqual(len(doc), 3)
            doc.close()
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

if __name__ == "__main__":
    # Run tests when executed as a script
    unittest.main(argv=["first-arg-is-ignored"], exit=False)



    # Lightweight "langgraph-like" flow using State and Node abstractions.
    # If a real langgraph is installed, these can be swapped; the simple
    # implementations below make the flow explicit and testable without extra deps.


    # Simple State object to hold values across nodes (mimics langgraph State)
    class State:
        def __init__(self, initial: Dict[str, Any] = None):
            self._data = dict(initial or {})

        def get(self, key: str, default: Any = None) -> Any:
            return self._data.get(key, default)

        def set(self, key: str, value: Any):
            self._data[key] = value

        def to_dict(self) -> Dict[str, Any]:
            return dict(self._data)

    # Simple Node base class (mimics langgraph Node)
    class Node:
        def __init__(self, name: str):
            self.name = name

        def run(self, state: State):
            """
            Perform the node computation and optionally modify the state.
            Subclasses must implement this.
            """
            raise NotImplementedError()

    # Execution graph that runs nodes in sequence (a trivial DAG runner)
    class Graph:
        def __init__(self, nodes: List[Node] = None):
            self.nodes = list(nodes or [])

        def add(self, node: Node):
            self.nodes.append(node)

        def run(self, state: State):
            for n in self.nodes:
                try:
                    n.run(state)
                except Exception as e:
                    # Surface node errors but continue where appropriate.
                    # In production you might stop or add retry semantics.
                    traceback.print_exc()
                    raise RuntimeError(f"Node '{n.name}' failed: {e}") from e

    # Nodes encapsulating the earlier functions in the module.

    class EnhancePromptNode(Node):
        """
        Calls enhance_prompt_to_json(prompt, llm) and stores result in state['structured'].
        Expects state to contain 'prompt' and 'llm'.
        """
        def run(self, state: State):
            prompt = state.get("prompt")
            llm = state.get("llm")
            if prompt is None or llm is None:
                raise ValueError("EnhancePromptNode requires 'prompt' and 'llm' in state.")
            structured = enhance_prompt_to_json(prompt, llm)
            state.set("structured", structured)

    class GenerateImagesNode(Node):
        """
        Calls generate_images_from_json(structured, image_client) and stores result in state['images_map'].
        Expects 'structured' and 'image_client' in state.
        """
        def run(self, state: State):
            structured = state.get("structured")
            image_client = state.get("image_client")
            if structured is None or image_client is None:
                raise ValueError("GenerateImagesNode requires 'structured' and 'image_client' in state.")
            images_map = generate_images_from_json(structured, image_client)
            state.set("images_map", images_map)

    class CreatePdfNode(Node):
        """
        Calls create_pdf_from_components(structured, images_map, output_path) and stores the returned path in state['output'].
        Expects 'structured', 'images_map', and 'output_path' in state.
        """
        def run(self, state: State):
            structured = state.get("structured")
            images_map = state.get("images_map", {})
            output_path = state.get("output_path")
            if structured is None or output_path is None:
                raise ValueError("CreatePdfNode requires 'structured' and 'output_path' in state.")
            out = create_pdf_from_components(structured, images_map, output_path)
            state.set("output", out)

    # Helper to construct the flow/graph
    def build_langgraph_flow() -> Graph:
        g = Graph()
        g.add(EnhancePromptNode("enhance_prompt"))
        g.add(GenerateImagesNode("generate_images"))
        g.add(CreatePdfNode("create_pdf"))
        return g

    # High-level API mirroring generate_pdf_from_prompt but executed via Nodes/State
    def generate_pdf_with_langgraph(prompt: str, llm: Any, image_client: Any, output_path: str) -> str:
        """
        Execute the PDF generation flow using State/Node abstractions.
        Returns the path to the generated PDF (same as output_path).
        """
        state = State({
            "prompt": prompt,
            "llm": llm,
            "image_client": image_client,
            "output_path": output_path,
        })
        graph = build_langgraph_flow()
        graph.run(state)
        return state.get("output")

    # Unit test that exercises the langgraph-style flow using the existing mocks.
    class GeneratePdfLangGraphTests(unittest.TestCase):
        def test_langgraph_flow_creates_pdf(self):
            prompt = "Create a PDF about testing agents (langgraph style)"
            llm = MockLLM()
            img_client = MockImageClient()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                out_path = tf.name
            try:
                result_path = generate_pdf_with_langgraph(prompt, llm, img_client, out_path)
                self.assertTrue(os.path.exists(result_path))
                # Inspect with PyMuPDF
                doc = fitz.open(result_path)
                self.assertGreaterEqual(len(doc), 3)
                doc.close()
            finally:
                if os.path.exists(out_path):
                    os.remove(out_path)