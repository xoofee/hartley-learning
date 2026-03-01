"""
Plugin/feature registry: add new demos without changing core GUI.
"""
from __future__ import annotations

from typing import Callable, Any

_FEATURES: dict[str, "Feature"] = {}


class Feature:
    """
    Optional UI/behavior for the imaging app.
    Override what you need; default no-ops.
    """

    def checkbox_widget(self, parent) -> Any:
        """Return a QWidget (e.g. QCheckBox) to add to the right pane, or None."""
        return None

    def on_toggled(self, checked: bool) -> None:
        """Called when the feature's checkbox is toggled."""
        pass

    def on_draw_3d(self, ax3d, context: dict) -> None:
        """Called during _draw_all after 3D scene is drawn. context has state, scene pts, etc."""
        pass

    def on_draw_image(self, ax_img, context: dict) -> None:
        """Called during _draw_all after image is drawn."""
        pass

    def connect_canvas_events(self, canvas, handlers: dict) -> list:
        """Connect canvas events; return list of connection ids for disconnect. handlers: {event_name: callback}."""
        return []

    def disconnect_canvas_events(self, canvas, cids: list) -> None:
        """Disconnect events by id."""
        for cid in cids:
            canvas.mpl_disconnect(cid)


def register_feature(name: str, feature: Feature) -> None:
    _FEATURES[name] = feature


def get_features() -> dict[str, Feature]:
    return dict(_FEATURES)
