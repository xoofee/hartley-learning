"""
Feature/plugin registry for camera model app (Open-Closed principle).

Demos: exclusive modes (only one active). Register with register_demo(Demo()).
Features: optional checkboxes (legacy). Register with register_feature(name, Feature()).
"""

from .registry import (
    register_feature,
    get_features,
    Feature,
    register_demo,
    get_demos,
    get_demo_by_id,
    Demo,
)

__all__ = [
    "register_feature",
    "get_features",
    "Feature",
    "register_demo",
    "get_demos",
    "get_demo_by_id",
    "Demo",
]
