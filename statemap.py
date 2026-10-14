"""US state outlines for the ibu map, drawn by hand (no map tiles, no CDN, works offline).

Albers equal-area conic projection in numpy; Alaska and Hawaii are re-projected and placed in
an inset under the Southwest, the way most U.S. maps do.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _albers(lon, lat, lat1, lat2, lon0, lat0):
    r = np.radians
    n = (np.sin(r(lat1)) + np.sin(r(lat2))) / 2
    c = np.cos(r(lat1)) ** 2 + 2 * n * np.sin(r(lat1))
    rho0 = np.sqrt(c - 2 * n * np.sin(r(lat0))) / n
    rho = np.sqrt(c - 2 * n * np.sin(r(lat))) / n
    th = n * r(np.asarray(lon) - lon0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


PARAMS = {
    "lower48": (29.5, 45.5, -96.0, 37.5),
    "Alaska": (55.0, 65.0, -154.0, 50.0),
    "Hawaii": (8.0, 18.0, -157.0, 3.0),
}


def load_state_shapes():
    """-> {state name: [(x_array, y_array), ...]} in map units, plus {state: (cx, cy)} label points."""
    g = json.load(open(os.path.join(HERE, "data", "us_states.geojson")))
    raw = {}
    for f in g["features"]:
        name = f["properties"]["NAME"]
        geo = f["geometry"]
        polys = [geo["coordinates"]] if geo["type"] == "Polygon" else geo["coordinates"]
        rings = [np.array(p[0], dtype=float) for p in polys]            # outer rings only
        key = name if name in ("Alaska", "Hawaii") else "lower48"
        lon0 = PARAMS[key][2]
        out = []
        for r in rings:
            lon = r[:, 0].copy()
            if name == "Alaska":                                        # Aleutians cross the date line
                lon = np.where(lon > 0, lon - 360, lon)
            if len(r) < 8 and name != "Hawaii":
                continue
            x, y = _albers(lon, r[:, 1], *PARAMS[key])
            out.append((x, y))
        raw[name] = out
    # place insets relative to the lower-48 bounding box
    allx = np.concatenate([x for k, v in raw.items() if k not in ("Alaska", "Hawaii") for x, _ in v])
    ally = np.concatenate([y for k, v in raw.items() if k not in ("Alaska", "Hawaii") for _, y in v])
    x0, y0, W, H = allx.min(), ally.min(), np.ptp(allx), np.ptp(ally)

    def place(name, scale, ax, ay):
        xs = np.concatenate([x for x, _ in raw[name]])
        ys = np.concatenate([y for _, y in raw[name]])
        bx, by = xs.min(), ys.min()
        tx, ty = x0 + ax * W, y0 + ay * H
        raw[name] = [((x - bx) * scale + tx, (y - by) * scale + ty) for x, y in raw[name]]

    place("Alaska", 0.34, 0.00, -0.17)
    place("Hawaii", 1.0, 0.31, -0.12)
    shapes, labels = {}, {}
    for name, rings in raw.items():
        shapes[name] = rings
        big = max(rings, key=lambda r: np.ptp(r[0]) * np.ptp(r[1]))
        labels[name] = (float(big[0].mean()), float(big[1].mean()))
    return shapes, labels
