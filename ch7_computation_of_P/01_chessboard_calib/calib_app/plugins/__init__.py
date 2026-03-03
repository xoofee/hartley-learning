"""Plugins: demo registry for extensible demos (realtime pose, etc.)."""
from .registry import Demo, register_demo, get_demos, get_demo_by_id

__all__ = ["Demo", "register_demo", "get_demos", "get_demo_by_id"]
