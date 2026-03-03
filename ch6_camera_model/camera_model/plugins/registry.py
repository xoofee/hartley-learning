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


# ---------------------------------------------------------------------------
# Exclusive demos (only one active at a time; each independent logic)
# ---------------------------------------------------------------------------

_DEMOS: list["Demo"] = []
_DEMOS_BY_ID: dict[str, "Demo"] = {}


class Demo:
    """
    One demo/mode in the Demos area. Buttons are exclusive: only one demo active.
    When demo is off, on_deactivated() is called so the demo can release all related state.
    """

    def id(self) -> str:
        """Unique id (e.g. 'none', 'p_planes', 'backproject', 'angulometer')."""
        raise NotImplementedError

    def label(self) -> str:
        """Short label for the demo button."""
        raise NotImplementedError

    def on_activated(self, context: dict) -> None:
        """Called when this demo becomes the active demo."""
        pass

    def on_deactivated(self) -> None:
        """Called when this demo is turned off; release all related objects/state."""
        pass

    def needs_image_events(self) -> bool:
        """If True, app will connect image-plot mouse events and dispatch to this demo."""
        return False

    def hide_scene_shapes(self) -> bool:
        """If True, app will not draw the default scene shapes (square, triangle, rectangle)."""
        return False

    def on_draw_3d(self, ax3d, context: dict) -> None:
        """Draw on 3D axes (only called when this demo is active)."""
        pass

    def on_draw_image(self, ax_img, context: dict) -> None:
        """Draw on image axes (only called when this demo is active)."""
        pass

    def on_image_button_press(self, event, context: dict) -> None:
        """Mouse press on image plot (only when needs_image_events and this demo is active)."""
        pass

    def on_image_motion(self, event, context: dict) -> None:
        """Mouse motion (only when needs_image_events and this demo is active). context has state, etc."""
        pass

    def on_image_button_release(self, event, context: dict) -> None:
        """Mouse release on image plot."""
        pass


def register_demo(demo: Demo) -> None:
    """Register a demo; order of registration is button order. Idempotent per id."""
    if demo.id() in _DEMOS_BY_ID:
        return
    _DEMOS.append(demo)
    _DEMOS_BY_ID[demo.id()] = demo


def get_demos() -> list[Demo]:
    return list(_DEMOS)


def get_demo_by_id(demo_id: str) -> Demo | None:
    return _DEMOS_BY_ID.get(demo_id)
