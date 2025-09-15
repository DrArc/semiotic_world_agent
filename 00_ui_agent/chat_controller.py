# chat_controller.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Backends the user already has
from lm_client import ask_agent
from lm_vision import analyze_with_lmstudio

try:
    from rag_semio import retrieve_context
except Exception:
    # Soft fallback: no RAG available
    def retrieve_context(query: str, top_k: int = 5) -> str:
        return ""

class ChatController:
    """
    Thin controller that:
      1) builds a context-augmented prompt (RAG) for the text agent,
      2) parses the JSON tool command,
      3) optionally runs the vision tool,
      4) returns a compact dict the UI can consume.
    """
    def __init__(self, rag_dir: Optional[str] = None, model: Optional[str] = None):
        self.rag_dir = rag_dir or os.getenv("SEMIO_RAG_DIR", "rag")
        self.model = model or os.getenv("SEMIO_LM_MODEL", "google/gemma-3-27b")

    # ---- public API ----
    def ask(self, user_text: str, image_path: Optional[str] = None, depth_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns a dict:
        {
           "action": "...",
           "args": {...},         # normalized args for UI to dispatch
           "analysis": {...},     # optional vision analysis JSON
           "notes": "...",        # human-readable summary for chat box
        }
        """
        # 1) Retrieve context for semiotics/architecture and append to query
        context = ""
        try:
            context = retrieve_context(user_text, top_k=5, root=self.rag_dir).strip()
        except Exception:
            context = ""
        rag_block = f"\n\n[CONTEXT]\n{context}\n\n" if context else ""

        # 2) Ask the local LM Studio agent for a JSON command
        agent_input = user_text + rag_block + "\nReturn only the JSON per schema."
        cmd = ask_agent(agent_input)

        # Normalize action + args
        action = (cmd.get("action") or "noop").strip()
        out: Dict[str, Any] = {"action": action, "args": {}, "notes": ""}

        # Map known fields
        if "prompt" in cmd: out["args"]["prompt"] = cmd["prompt"]
        if "style"  in cmd: out["args"]["style"]  = cmd["style"]
        if "image"  in cmd: out["args"]["image"]  = cmd["image"]
        if "depth"  in cmd: out["args"]["depth"]  = cmd["depth"]
        if "goals"  in cmd: out["args"]["goals"]  = cmd["goals"]

        # 3) Optional: run vision analysis immediately if user asked for it,
        #    or if the action is analyze_image and we have a current image.
        if action == "analyze_image":
            img = out["args"].get("image") or image_path
            dep = out["args"].get("depth") or depth_path
            goals = out["args"].get("goals") or user_text
            if not img:
                out["notes"] = "No image available to analyze."
            else:
                try:
                    analysis = analyze_with_lmstudio(img, dep, goals=goals, model=self.model)
                    out["analysis"] = analysis
                    out["notes"] = f"Analysis ready (denotation/connotation/myth)."
                except Exception as e:
                    out["notes"] = f"Vision analysis failed: {e}"

        # Friendly summary
        if not out.get("notes"):
            if action == "generate_image":
                out["notes"] = "Routing to Step‑1: image generation."
            elif action == "monochrome_clean":
                out["notes"] = "Routing to Step‑2: Kontext cleanup."
            elif action == "generate_3d":
                out["notes"] = "Routing to Step‑3: 2D→3D."
            elif action == "comfy360":
                out["notes"] = "Comfy360 not wired here."
            elif action == "noop":
                out["notes"] = cmd.get("error", "No actionable JSON returned.")
            else:
                out["notes"] = f"Unhandled action: {action}."

        # Always echo raw for debugging if present
        if "raw" in cmd and "error" in cmd:
            out["raw"] = cmd["raw"]
        return out