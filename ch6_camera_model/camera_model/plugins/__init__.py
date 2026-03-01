"""
Feature/plugin registry for camera model app (Open-Closed principle).

New features (binocular, multi-camera, epipolar demo, angle-by-K, etc.) can be
registered without modifying the core app. Each plugin provides:
- Optional: checkbox widget and callback when toggled
- Optional: extra draw logic (3D / image)
- Optional: mouse/event hooks on the image plot

Usage:
  from camera_model.plugins import register_feature, get_features
  register_feature("backproject", BackprojectFeature())
"""

from .registry import register_feature, get_features, Feature

__all__ = ["register_feature", "get_features", "Feature"]
