#!/usr/bin/env python3
"""
Fly the drone-racing track with MAVSDK-Python + PX4 offboard mode (no QGroundControl).

    pip install "mavsdk<4"

    python3 fly_route.py --dry-run --speed 5 --plot plan.png   # plan only, no sim needed
    python3 fly_route.py --speed 5                              # fly (PX4 SITL running)

How it flies
  1. PLAN (before arming).  A smooth cubic-Hermite spline runs through every gate center,
     leaving each gate along its flight axis.  A small optimizer then shifts where the line
     crosses each gate (within the opening) and how wide each turn is, to minimize lap time at
     the speed you asked for while staying clear of every frame, stand, cone and the boundary.

  2. SPEED PROFILE.  Speed along the path = min(--speed, sqrt(lateral_accel / curvature)),
     smoothed with forward/backward passes so it brakes before corners and accelerates out.
     Faster --speed  =>  wider, flatter lines and slower hairpins automatically.
  
  3. FLY.  Take off to gate height, then stream position + velocity + acceleration setpoints
     at 50 Hz on a fixed clock.  No stop-and-go: the trajectory is a function of time.
  
  4. WATCH.  Every tick the drone position is checked against the real gate geometry.  A hit is
     logged (which gate, which part, speed) and, by default, the motors are killed like a real
     crash.  An IMU shock detector backs this up.  Gate passes and clearances are reported.
"""
import argparse
import asyncio
import bisect
import json
import math
import time
from pathlib import Path
from utility.vector_helpers import vadd, vsub, vmul, vdot, vnorm, vunit, vlerp, wrap180, slew

FT = 0.3048
TRACK_L_FT = 165.0
G = 9.81

# Extra via-points where the racing line loops between gates: gate -> [(X_ft, Y_ft, heading_deg_ENU)]
# After gate 5 the line swings west, hairpins at its western apex (heading south), and returns east.
VIA_AFTER_GATE_FT = {5: [(30.4, 82.4, -90.0)]}

PX4_PARAMS_INT = {
    "NAV_DLL_ACT": 0,       # no action on data-link loss (no QGC connected)
    "COM_RCL_EXCEPT": 4,    # no RC-loss failsafe in offboard mode
    "COM_RC_IN_MODE": 4,    # no RC / stick input required
}


def px4_float_params(a):
    """Loosen PX4's default speed/acceleration/tilt limits so it can follow the trajectory."""
    return {
        "MPC_XY_VEL_MAX": max(12.0, a.speed * 1.4),
        "MPC_Z_VEL_MAX_UP": 8.0,
        "MPC_Z_VEL_MAX_DN": 8.0,
        "MPC_ACC_HOR_MAX": max(10.0, a.lat_accel * 1.6),
        "MPC_ACC_HOR": max(8.0, a.lat_accel * 1.4),
        "MPC_TILTMAX_AIR": 60.0,
    }

# ----------------------------------------------------------------------------
# World geometry (matches the SDF the generator writes) - used for planning AND collisions
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Knots -> spline -> speed profile
# ----------------------------------------------------------------------------
def build_knots(meta, double_mode, turn_out=2.0):
    """Ordered knots: dict(pos, dir, c, opt, perp, label, gate, req) in ENU meters."""
    st = meta["start"]
    sx, sy = st["x"], st["y"]
    gates = meta["gates_in_order"]
    knots = []

    def K(pos, d, label, gate=None, opt=False, c=1.0, perp=(0, 0, 0), req=1):
        knots.append(dict(pos=pos, dir=d, c=c, opt=opt, perp=perp, label=label, gate=gate, req=req))

    z1 = gates[0]["z_path"]
    g1 = gates[0]
    K((sx, sy, z1), vunit((g1["x"] - sx, g1["y"] - sy, 0)), "start")

    for g in gates:
        yaw = math.radians(g["yaw_deg"])
        d = (math.cos(yaw), math.sin(yaw), 0.0)
        perp = (-math.sin(yaw), math.cos(yaw), 0.0)
        x, y, n = g["x"], g["y"], str(g["gate"])
        if g.get("double_stack") and double_mode == "upper-lower":
            zu, zl = g["z_upper"], g["z_lower"]
            r, zc = (zu - zl) / 2.0, (zu + zl) / 2.0

            def along(t): return (x + t * d[0], y + t * d[1])
            K((x, y, zu), d, f"GATE{n}U", f"{n}U", req=1)
            K((*along(turn_out), zu), d, f"loop{n}.0", c=1.0)
            K((*along(turn_out + r), zc), (0, 0, -1), f"loop{n}.90", c=1.11)
            K((*along(turn_out), zl), vmul(d, -1), f"loop{n}.180", c=1.0)
            K((x, y, zl), vmul(d, -1), f"GATE{n}L", f"{n}L", req=-1)
        else:
            dbl = g.get("double_stack")
            z = g["z_upper"] if (dbl and double_mode == "upper") else g["z_path"]
            gid = (f"{n}U" if double_mode == "upper" else f"{n}L") if dbl else n
            K((x, y, z), d, f"GATE{n}", gid, opt=not dbl, perp=perp)
        for i, (xf, yf, hd) in enumerate(VIA_AFTER_GATE_FT.get(g["gate"], [])):
            h = math.radians(hd)
            K((xf * FT, (TRACK_L_FT - yf) * FT, knots[-1]["pos"][2]), (math.cos(h), math.sin(h), 0.0),
              f"via{n}.{i}", c=0.9)

    last = knots[-1]["pos"]
    K((sx, sy, gates[-1]["z_path"]), vunit(vsub((sx, sy, gates[-1]["z_path"]), last)), "home_above")
    return knots


def hermite_sample(P0, M0, P1, M1, u):
    u2, u3 = u * u, u * u * u
    h00, h10, h01, h11 = 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2
    d00, d10, d01, d11 = 6 * u2 - 6 * u, 3 * u2 - 4 * u + 1, -6 * u2 + 6 * u, 3 * u2 - 2 * u
    e00, e10, e01, e11 = 12 * u - 6, 6 * u - 4, -12 * u + 6, 6 * u - 2
    P = tuple(h00 * P0[i] + h10 * M0[i] + h01 * P1[i] + h11 * M1[i] for i in range(3))
    D1 = tuple(d00 * P0[i] + d10 * M0[i] + d01 * P1[i] + d11 * M1[i] for i in range(3))
    D2 = tuple(e00 * P0[i] + e10 * M0[i] + e01 * P1[i] + e11 * M1[i] for i in range(3))
    return P, D1, D2


def build_path(knots, offs, cs, ds):
    """Dense samples along the spline: pts, unit tangents, curvature vectors (dT/ds), arc length."""
    pts, tans, kvs = [], [], []
    n = len(knots)
    base = [vadd(k["pos"], vmul(k["perp"], offs[i])) for i, k in enumerate(knots)]
    for i in range(n - 1):
        P0, P1 = base[i], base[i + 1]
        L = vnorm(vsub(P1, P0))
        M0 = vmul(knots[i]["dir"], cs[i] * L)
        M1 = vmul(knots[i + 1]["dir"], cs[i + 1] * L)
        m = max(6, int(L / ds) + 1)
        last = (i == n - 2)
        for k in range(m + (1 if last else 0)):
            u = k / m
            P, D1, D2 = hermite_sample(P0, M0, P1, M1, u)
            sp = vnorm(D1)
            if sp < 1e-6:
                continue
            T = vmul(D1, 1.0 / sp)
            perp_acc = vsub(D2, vmul(T, vdot(D2, T)))
            pts.append(P)
            tans.append(T)
            kvs.append(vmul(perp_acc, 1.0 / (sp * sp)))
    s = [0.0]
    for a, b in zip(pts, pts[1:]):
        s.append(s[-1] + vnorm(vsub(b, a)))
    return pts, tans, kvs, s


def speed_profile(s, kvs, vmax, a_lat, a_acc, a_dec):
    n = len(s)
    v = []
    for kv in kvs:
        k = vnorm(kv)
        v.append(min(vmax, math.sqrt(a_lat / k)) if k > 1e-4 else vmax)
    v[0] = 0.0
    for i in range(1, n):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_acc * (s[i] - s[i - 1])))
    v[-1] = 0.0
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_dec * (s[i + 1] - s[i])))
    t = [0.0]
    for i in range(n - 1):
        ds = s[i + 1] - s[i]
        vs = v[i] + v[i + 1]
        t.append(t[-1] + (2 * ds / vs if vs > 1e-6 else 0.1))
    return v, t


class Trajectory:
    def __init__(self, pts, tans, kvs, s, v, t):
        self.pts, self.tans, self.kvs, self.s, self.v, self.t = pts, tans, kvs, s, v, t
        self.T = t[-1]

    def sample(self, tm):
        """pos, vel, acc, unit tangent, arc length at time tm (ENU)."""
        if tm <= 0.0:
            return self.pts[0], (0, 0, 0), (0, 0, 0), self.tans[0], 0.0
        if tm >= self.T:
            return self.pts[-1], (0, 0, 0), (0, 0, 0), self.tans[-1], self.s[-1]
        i = min(bisect.bisect_right(self.t, tm) - 1, len(self.t) - 2)
        dt = self.t[i + 1] - self.t[i]
        f = (tm - self.t[i]) / dt if dt > 1e-9 else 0.0
        pos = vlerp(self.pts[i], self.pts[i + 1], f)
        T = vunit(vlerp(self.tans[i], self.tans[i + 1], f))
        kv = vlerp(self.kvs[i], self.kvs[i + 1], f)
        sp = self.v[i] + (self.v[i + 1] - self.v[i]) * f
        at = (self.v[i + 1] - self.v[i]) / dt if dt > 1e-9 else 0.0
        vel = vmul(T, sp)
        acc = vadd(vmul(T, at), vmul(kv, sp * sp))
        return pos, vel, acc, T, self.s[i] + (self.s[i + 1] - self.s[i]) * f

    def tangent_at_s(self, sv):
        j = min(bisect.bisect_left(self.s, sv), len(self.s) - 1)
        return self.tans[j]


# ----------------------------------------------------------------------------
# Planner (minimum-time-ish): coordinate descent on gate crossing offsets + curve widths
# ----------------------------------------------------------------------------
class Planner:
    def __init__(self, meta, world, args):
        self.meta, self.world, self.args = meta, world, args
        self.knots = build_knots(meta, args.double_mode)
        n = len(self.knots)
        self.opt_idx = [i for i, k in enumerate(self.knots) if k["opt"]]
        self.offs = [0.0] * n
        self.cs = [k["c"] for k in self.knots]
        self.need = args.drone_radius + args.plan_margin
        self.omax = max(0.0, world.W / 2 - args.drone_radius - args.plan_margin - 0.05)

    def evaluate(self, offs, cs, ds=0.25):
        a = self.args
        pts, tans, kvs, s = build_path(self.knots, offs, cs, ds)
        v, t = speed_profile(s, kvs, a.speed, a.lat_accel, a.accel, a.accel * 1.25)
        pen, maxv = 0.0, 0.0
        bw, bl = self.world.boundary
        for i, p in enumerate(pts):
            d, _, _ = self.world.min_clear(p)
            vio = self.need - d
            if p[0] < 0.4: vio = max(vio, 0.4 - p[0])
            if p[0] > bw - 0.4: vio = max(vio, p[0] - (bw - 0.4))
            if p[1] < 0.4: vio = max(vio, 0.4 - p[1])
            if p[1] > bl - 0.4: vio = max(vio, p[1] - (bl - 0.4))
            if i > 5 and p[2] < 0.7: vio = max(vio, 0.7 - p[2])
            if vio > 0:
                pen += vio * (s[i] - s[i - 1] if i else 0.0)
                maxv = max(maxv, vio)
        return t[-1] + 200.0 * maxv + 40.0 * pen, t[-1], maxv

    def optimize(self, passes=4, verbose=True):
        offs, cs = list(self.offs), list(self.cs)
        best, T, vio = self.evaluate(offs, cs)
        if verbose:
            print(f"  initial plan: lap {T:.1f} s" + (f" (clearance violation {vio:.2f} m)" if vio > 0 else ""))
        for ps in range(passes):
            so, sc = 0.45 / (2 ** ps), 0.35 / (2 ** ps)
            for i in self.opt_idx:
                for var, step, lo, hi in (("o", so, -self.omax, self.omax), ("c", sc, 0.5, 2.0)):
                    for delta in (step, -step):
                        o2, c2 = list(offs), list(cs)
                        if var == "o":
                            o2[i] = min(hi, max(lo, o2[i] + delta))
                        else:
                            c2[i] = min(hi, max(lo, c2[i] + delta))
                        cost, T2, v2 = self.evaluate(o2, c2)
                        if cost < best - 1e-6:
                            best, offs, cs = cost, o2, c2
            if verbose:
                _, T, vio = self.evaluate(offs, cs)
                print(f"  pass {ps + 1}/{passes}: lap {T:.2f} s" + (f"  (violation {vio:.2f} m)" if vio > 0 else ""))
        self.offs, self.cs = offs, cs

    def final(self):
        a = self.args
        pts, tans, kvs, s = build_path(self.knots, self.offs, self.cs, 0.05)
        v, t = speed_profile(s, kvs, a.speed, a.lat_accel, a.accel, a.accel * 1.25)
        return Trajectory(pts, tans, kvs, s, v, t)


def plan_summary(traj, world, knots, args):
    mc = min(world.min_clear(p)[0] for p in traj.pts[::4])
    print(f"Plan: {traj.s[-1]:.1f} m, lap {traj.T:.1f} s, peak {max(traj.v):.1f} m/s (limit {args.speed}), "
          f"avg {traj.s[-1] / traj.T:.1f} m/s, min clearance {mc - args.drone_radius:+.2f} m beyond drone radius")
    gl = []
    for k in knots:
        if k["gate"]:
            j = min(range(len(traj.pts)), key=lambda i: vnorm(vsub(traj.pts[i], k["pos"])))
            gl.append((k["gate"], traj.v[j]))
    print("  planned speed through gates: " + "  ".join(f"{g}:{v:.1f}" for g, v in gl) + "  (m/s)")


def plot_plan(path, traj, world, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [1, 1.1]})
    bw, bl = world.boundary
    ax.plot([0, bw, bw, 0, 0], [0, 0, bl, bl, 0], "k-", lw=1)
    xs, ys = [p[0] for p in traj.pts], [p[1] for p in traj.pts]
    sc = ax.scatter(xs, ys, c=traj.v, s=3, cmap="turbo")
    for f in world.frames:
        dx, dy = -f["s"] * (world.W / 2 + 0.1), f["c"] * (world.W / 2 + 0.1)
        ax.plot([f["gx"] - dx, f["gx"] + dx], [f["gy"] - dy, f["gy"] + dy], "k-", lw=4, alpha=0.7)
        ax.annotate(f["id"], (f["gx"], f["gy"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    for cx, cy in world.cones:
        ax.plot(cx, cy, "o", color="orange", ms=8)
    ax.set_aspect("equal")
    ax.set_title("Planned line (color = speed, m/s)")
    fig.colorbar(sc, ax=ax, shrink=0.5)
    ax2.plot(traj.t, traj.v)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("speed (m/s)")
    ax2.grid(True)
    ax3 = ax2.twinx()
    ax3.plot(traj.t, [p[2] for p in traj.pts], "g", alpha=0.6)
    ax3.set_ylabel("height (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


# ----------------------------------------------------------------------------
# MAVSDK helpers
# ----------------------------------------------------------------------------
async def first(aiter):
    async for x in aiter:
        return x


async def wait_for(aiter, pred, timeout, what):
    async def _go():
        async for x in aiter:
            if pred(x):
                return x
    try:
        return await asyncio.wait_for(_go(), timeout)
    except asyncio.TimeoutError:
        raise SystemExit(f"Timed out waiting for {what}")


async def set_params(drone, args):
    for name, val in PX4_PARAMS_INT.items():
        try:
            await drone.param.set_param_int(name, val)
            print(f"  param {name} = {val}")
        except Exception as e:
            print(f"  (skipped {name}: {e})")
    for name, val in px4_float_params(args).items():
        try:
            await drone.param.set_param_float(name, float(val))
            print(f"  param {name} = {val:g}")
        except Exception as e:
            print(f"  (skipped {name}: {e})")


# ----------------------------------------------------------------------------
# Main flight
# ----------------------------------------------------------------------------
async def run(args):
    meta = json.loads(Path(args.gates).read_text())
    world = World(meta)
    st = meta["start"]
    sx, sy, sz = st["x"], st["y"], st["z"]

    print("Planning ...")
    planner = Planner(meta, world, args)
    if not args.no_optimize:
        t0 = time.time()
        planner.optimize()
        print(f"  optimized in {time.time() - t0:.1f} s")
    traj = planner.final()
    plan_summary(traj, world, planner.knots, args)
    if args.plot:
        plot_plan(args.plot, traj, world, meta)
        print(f"  plot saved to {args.plot}")
    if args.dry_run:
        return

    from mavsdk import System
    from mavsdk.offboard import (AccelerationNed, OffboardError, PositionNedYaw, VelocityNedYaw)

    drone = System()
    print(f"Connecting to {args.address} ...")
    await drone.connect(system_address=args.address)
    await wait_for(drone.core.connection_state(), lambda s: s.is_connected, args.connect_timeout, "PX4 connection")
    print("Connected.")
    for fn, hz in (("set_rate_position_velocity_ned", 50.0), ("set_rate_imu", 100.0)):
        try:
            await getattr(drone.telemetry, fn)(hz)
        except Exception:
            pass
    if not args.no_params:
        print("Setting PX4 params for GCS-less, faster flight:")
        await set_params(drone, args)

    print("Waiting for position estimate ...")
    await wait_for(drone.telemetry.health(),
                   lambda h: h.is_global_position_ok and h.is_home_position_ok, 90, "GPS/home health")

    state = {"n": 0.0, "e": 0.0, "d": 0.0, "vn": 0.0, "ve": 0.0, "vd": 0.0}
    flags = {"flying": False, "impact": None}

    async def track_state():
        async for pv in drone.telemetry.position_velocity_ned():
            p, v = pv.position, pv.velocity
            state.update(n=p.north_m, e=p.east_m, d=p.down_m, vn=v.north_m_s, ve=v.east_m_s, vd=v.down_m_s)

    async def watch_imu():
        if args.impact_g <= 0:
            return
        async for imu in drone.telemetry.imu():
            a = imu.acceleration_frd
            mag = math.sqrt(a.forward_m_s2 ** 2 + a.right_m_s2 ** 2 + a.down_m_s2 ** 2)
            if flags["flying"] and mag > args.impact_g * G and flags["impact"] is None:
                flags["impact"] = mag / G

    tasks = [asyncio.ensure_future(track_state()), asyncio.ensure_future(watch_imu())]
    await asyncio.sleep(1.0)
    n0, e0, d0 = state["n"], state["e"], state["d"]
    heading0 = (await first(drone.telemetry.heading())).heading_deg

    def to_ned(p): return (p[1] - sy + n0, p[0] - sx + e0, d0 - (p[2] - sz))
    def vec_ned(v): return (v[1], v[0], -v[2])
    def cur_enu(): return (state["e"] - e0 + sx, state["n"] - n0 + sy, sz - (state["d"] - d0))

    takeoff_alt = traj.pts[0][2] - sz
    crashed = False
    status, min_clear = {}, {}
    try:
        print(f"Arming, taking off to {takeoff_alt:.2f} m ...")
        await drone.action.set_takeoff_altitude(takeoff_alt)
        await drone.action.arm()
        await drone.action.takeoff()
        await wait_for(drone.telemetry.position(),
                       lambda p: p.relative_altitude_m >= takeoff_alt - 0.25, 30, "takeoff altitude")
        await asyncio.sleep(2.0)

        p0 = to_ned(traj.pts[0])
        yaw = heading0
        await drone.offboard.set_position_velocity_acceleration_ned(
            PositionNedYaw(p0[0], p0[1], p0[2], yaw), VelocityNedYaw(0, 0, 0, yaw), AccelerationNed(0, 0, 0))
        try:
            await drone.offboard.start()
        except OffboardError as e:
            raise SystemExit(f"Offboard start failed: {e._result.result}")
        print(f"Offboard active - flying {traj.s[-1]:.0f} m lap, planned {traj.T:.1f} s, speed limit {args.speed} m/s")

        req = [(k["gate"], k["req"]) for k in planner.knots if k["gate"]]
        for gid, _ in req:
            status[gid] = "NOT REACHED"
        prev = cur_enu()
        flags["flying"] = True
        dt = 1.0 / args.rate
        t_start = time.monotonic()
        next_tick = t_start
        t_log = 0.0
        warned_lag = False
        touching = set()
        while True:
            tm = time.monotonic() - t_start
            pos, vel, acc, T, s_here = traj.sample(tm)
            ahead = traj.tangent_at_s(s_here + args.look_ahead)
            if math.hypot(ahead[0], ahead[1]) > 0.3:       # hold heading through pure climbs/descents
                yaw = slew(yaw, math.degrees(math.atan2(ahead[0], ahead[1])), args.yaw_rate * dt)
            pn, vn_, an_ = to_ned(pos), vec_ned(vel), vec_ned(acc)
            await drone.offboard.set_position_velocity_acceleration_ned(
                PositionNedYaw(pn[0], pn[1], pn[2], yaw),
                VelocityNedYaw(vn_[0], vn_[1], vn_[2], yaw),
                AccelerationNed(an_[0], an_[1], an_[2]))

            # ---- monitoring ----------------------------------------------------------
            cur = cur_enu()
            speed_now = math.sqrt(state["vn"] ** 2 + state["ve"] ** 2 + state["vd"] ** 2)
            hit = None
            for k in range(5):                              # sweep between ticks so nothing tunnels
                q = vlerp(prev, cur, k / 4.0)
                d, oid, part = world.min_clear(q)
                if oid and d - args.drone_radius < min_clear.get(oid, (9e9,))[0]:
                    min_clear[oid] = (d - args.drone_radius, part)
                if d < args.drone_radius and hit is None:
                    hit = (oid, part, d, q)
            for gid, rq in req:                             # gate-plane crossings
                if status[gid].startswith("CLEARED"):
                    continue
                u0, v0, w0 = world.frame_local(gid, prev)
                u1, v1, w1 = world.frame_local(gid, cur)
                if u0 * u1 < 0 or (u0 != 0 and u1 == 0):
                    f = u0 / (u0 - u1) if u0 != u1 else 0.0
                    vc, wc = v0 + (v1 - v0) * f, w0 + (w1 - w0) * f
                    inside = abs(vc) < world.W / 2 and abs(wc) < world.H / 2
                    right_way = (u1 - u0) * rq > 0
                    if inside and right_way:
                        status[gid] = f"CLEARED t={tm:.1f}s v={speed_now:.1f}m/s"
                        print(f"  gate {gid:>3} cleared   t={tm:5.1f}s  speed {speed_now:4.1f} m/s")
                    elif inside and not right_way:
                        status[gid] = "WRONG DIRECTION"
                    elif abs(vc) < world.W / 2 + world.T + 0.6 and abs(wc) < world.H / 2 + world.T + 0.6:
                        status[gid] = f"MISSED opening (off-center v={vc:+.2f} w={wc:+.2f})"
            prev = cur

            if hit is None and flags["impact"] is not None:
                d, oid, part = world.min_clear(cur)
                print(f"\n*** IMPACT detected by IMU: {flags['impact']:.1f} g at t={tm:.1f}s "
                      f"(nearest: {oid} {part}, {d:.2f} m away) ***")
                hit = (oid, part, d, cur)
                flags["impact"] = None
            if hit is not None:
                oid, part, dd, q = hit
                if oid not in touching:                     # log each new contact once
                    touching.add(oid)
                    print(f"\n*** COLLISION: {oid} ({part}) at t={tm:.1f}s, speed {speed_now:.1f} m/s, "
                          f"pos ENU ({q[0]:.1f}, {q[1]:.1f}, {q[2]:.1f}) ***")
                    if oid:
                        status[oid] = status.get(oid, "") + " | HIT"
                if args.on_collision == "crash":
                    await drone.action.kill()
                    crashed = True
                    print("*** Motors killed - drone is down. ***")
                    break
                if args.on_collision == "land":
                    print("*** Aborting route and landing. ***")
                    break
            else:
                touching.clear()

            lag = math.sqrt((pn[0] - state["n"]) ** 2 + (pn[1] - state["e"]) ** 2 + (pn[2] - state["d"]) ** 2)
            if lag > 3.0 and not warned_lag:
                print(f"  warning: tracking lag {lag:.1f} m - lower --speed or --lat-accel")
                warned_lag = True
            if tm - t_log >= 2.0:
                t_log = tm
                print(f"  t={tm:5.1f}s  speed {speed_now:4.1f}/{vnorm(vel):4.1f} m/s  "
                      f"lag {lag:4.2f} m  alt {sz - (state['d'] - d0):4.2f} m")
            if tm > traj.T + 1.5 or tm > args.timeout:
                break
            next_tick += dt
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

        flags["flying"] = False
        print("\n=== Run report ===")
        if crashed:
            print("RESULT: CRASHED")
        else:
            done = sum(1 for v in status.values() if v.startswith("CLEARED"))
            print(f"RESULT: {'completed' if done == len(status) else 'incomplete'} - "
                  f"{done}/{len(status)} gate passes cleared")
        for gid in status:
            mc = min_clear.get(gid)
            cl = f"closest approach to frame {mc[0]:+.2f} m ({mc[1]})" if mc else ""
            print(f"  gate {gid:>3}: {status[gid]:<48} {cl}")
        print("  (clearance is measured beyond the drone's radius; negative = contact)")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        flags["flying"] = False
        for tk in tasks:
            tk.cancel()
        if crashed:
            await asyncio.sleep(1.5)
        else:
            try:
                await drone.offboard.stop()
            except Exception:
                pass
            print("Landing ...")
            try:
                await drone.action.land()
                await wait_for(drone.telemetry.armed(), lambda a: not a, 40, "disarm")
                print("Landed and disarmed.")
            except BaseException as e:
                print(f"Landing note: {e}")


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--gates", default=str(here / "gates.json"))
    ap.add_argument("--address", default="udpin://0.0.0.0:14540")
    ap.add_argument("--speed", type=float, default=3.0, help="top speed, m/s (corners are slower automatically)")
    ap.add_argument("--accel", type=float, default=4.0, help="max speed-up along the path, m/s^2 (braking is 1.25x)")
    ap.add_argument("--lat-accel", type=float, default=5.0, help="max cornering acceleration, m/s^2 (< ~9)")
    ap.add_argument("--drone-radius", type=float, default=0.35, help="collision radius of the drone, m")
    ap.add_argument("--plan-margin", type=float, default=0.35, help="extra planned clearance from any frame, m")
    ap.add_argument("--on-collision", choices=["crash", "land", "continue"], default="crash",
                    help="crash = kill motors (like a real hit), land = abort+land, continue = just log")
    ap.add_argument("--impact-g", type=float, default=4.0, help="IMU shock threshold in g (0 disables)")
    ap.add_argument("--look-ahead", type=float, default=3.0, help="point the nose this many m ahead on the path")
    ap.add_argument("--yaw-rate", type=float, default=150.0, help="max heading slew, deg/s")
    ap.add_argument("--double-mode", choices=["upper-lower", "upper", "lower"], default="upper-lower")
    ap.add_argument("--rate", type=float, default=50.0, help="setpoint rate, Hz")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--connect-timeout", type=float, default=120.0)
    ap.add_argument("--no-params", action="store_true", help="don't touch PX4 parameters")
    ap.add_argument("--no-optimize", action="store_true", help="skip the line optimizer (gate-center line)")
    ap.add_argument("--plot", metavar="PNG", help="save a plot of the planned line and speed profile")
    ap.add_argument("--dry-run", action="store_true", help="plan only; don't connect")
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()