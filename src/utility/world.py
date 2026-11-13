import math

class World:
    CULL = 4.0

    def __init__(self, meta):
        W = meta["gate_opening_m"]
        T = meta["bar_m"]
        D = meta["depth_m"]
        H = W
        self.W, self.H, self.T = W, H, T
        self.boundary = meta["boundary_m"]
        self.cones = [tuple(c) for c in meta["cones"]]
        self.frames = []
        for g in meta["gates_in_order"]:
            yaw = math.radians(g["yaw_deg"])
            c, s = math.cos(yaw), math.sin(yaw)
            if g.get("double_stack"):
                specs = [(f'{g["gate"]}L', g["z_lower"], True), (f'{g["gate"]}U', g["z_upper"], False)]
            else:
                specs = [(str(g["gate"]), g["z_center"], True)]
            for fid, zc, has_stand in specs:
                boxes = [
                    (0, -(W / 2 + T / 2), zc, D / 2, T / 2, (H + 2 * T) / 2, "left post"),
                    (0, (W / 2 + T / 2), zc, D / 2, T / 2, (H + 2 * T) / 2, "right post"),
                    (0, 0, zc + H / 2 + T / 2, D / 2, W / 2, T / 2, "top bar"),
                    (0, 0, zc - (H / 2 + T / 2), D / 2, W / 2, T / 2, "bottom bar"),
                ]
                sh = zc - H / 2 - T
                if has_stand and sh > 0.05:
                    boxes.append((0, 0, sh / 2, 0.06, 0.06, sh / 2, "stand"))
                self.frames.append(dict(id=fid, gx=g["x"], gy=g["y"], c=c, s=s, zc=zc, boxes=boxes))
        self.by_id = {f["id"]: f for f in self.frames}

    def min_clear(self, p):
        """(distance, object id, part) to the nearest solid thing from point p (ENU)."""
        x, y, z = p
        best = (1e9, None, None)
        for f in self.frames:
            dx, dy = x - f["gx"], y - f["gy"]
            if abs(dx) > self.CULL or abs(dy) > self.CULL:
                continue
            u = dx * f["c"] + dy * f["s"]
            v = -dx * f["s"] + dy * f["c"]
            for cu, cv, cw, hu, hv, hw, name in f["boxes"]:
                a = abs(u - cu) - hu
                b = abs(v - cv) - hv
                cc = abs(z - cw) - hw
                a = a if a > 0 else 0.0
                b = b if b > 0 else 0.0
                cc = cc if cc > 0 else 0.0
                d = math.sqrt(a * a + b * b + cc * cc)
                if d < best[0]:
                    best = (d, f["id"], name)
        for i, (cx, cy) in enumerate(self.cones):
            if abs(x - cx) > self.CULL or abs(y - cy) > self.CULL:
                continue
            hh = max(math.hypot(x - cx, y - cy) - 0.2, 0.0)
            vv = max(z - 0.7, 0.0)
            d = math.hypot(hh, vv)
            if d < best[0]:
                best = (d, f"cone{i}", "cone")
        return best

    def frame_local(self, fid, p):
        f = self.by_id[fid]
        dx, dy = p[0] - f["gx"], p[1] - f["gy"]
        return dx * f["c"] + dy * f["s"], -dx * f["s"] + dy * f["c"], p[2] - f["zc"]
