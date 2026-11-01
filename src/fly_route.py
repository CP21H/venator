#!/usr/bin/env python3
"""
Fly a fixed route through the drone-racing gates using MAVSDK-Python + PX4 offboard
mode. No QGroundControl needed.

Setup (classic MAVSDK-Python API; v4+ changed the API, so pin it):
    pip install "mavsdk<4"

Usage:
    python3 fly_route.py --dry-run            # print the route, don't connect
    python3 fly_route.py                      # fly it (PX4 SITL must be running)
    python3 fly_route.py --speed 3.0 --approach 2.0

How it works
  * Reads gates.json (written by make_track_world.py): gate centers, flight
    directions, start pad.
  * Builds a polyline:  climb -> [pre-gate point -> gate center -> post-gate point] x N
    The pre/post points sit on the gate's flight axis so the drone crosses each
    opening head-on.
  * A "carrot" moves along that polyline at --speed. Every 50 ms we send PX4 the
    carrot position + velocity feed-forward + heading. The carrot waits if the
    drone falls more than --lead meters behind it.
  * Local NED frame: PX4's origin is where the drone spawned (the start pad), so
    gate NED = gate ENU - start ENU.
"""
import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

# Extra via-points where the racing line loops out between gates (feet, PDF frame).
# After gate 5 the line swings west before coming back east through gate 6.
VIA_AFTER_GATE_FT = {5: [(30.4, 82.4)]}
TRACK_L_FT = 165.0
FT = 0.3048

# PX4 parameters needed to fly with no GCS / RC attached (SITL only!)
PX4_PARAMS_INT = {
    "NAV_DLL_ACT": 0,       # no action on data-link loss (no QGC connected)
    "COM_RCL_EXCEPT": 4,    # don't trigger RC-loss failsafe in offboard mode
    "COM_RC_IN_MODE": 4,    # no RC / stick input required
}


# ----------------------------------------------------------------------------
# Route building
# ----------------------------------------------------------------------------
def build_route(gates_json, approach, double_mode="upper-lower", turn_out=2.0):
    """Returns (route, meta). route = list of dicts {x,y,z,label,gate} in Gazebo ENU meters.

    Gate ids are strings ("1".."10"; the double gate becomes "9U" and "9L").
    double_mode:
      upper-lower : through the TOP opening, a half-loop (split-S) past the gate, then back
                    through the BOTTOM opening in the opposite direction
      lower / upper : fly one opening only
    """
    meta = json.loads(Path(gates_json).read_text())
    st = meta["start"]
    sx, sy, sz = st["x"], st["y"], st["z"]
    gates = meta["gates_in_order"]

    route = [{"x": sx, "y": sy, "z": sz, "label": "start", "gate": None}]
    route.append({"x": sx, "y": sy, "z": gates[0]["z_path"], "label": "climb", "gate": None})

    def add(x, y, z, label, gate=None):
        route.append({"x": x, "y": y, "z": z, "label": label, "gate": gate})

    for g in gates:
        yaw = math.radians(g["yaw_deg"])
        dx, dy = math.cos(yaw), math.sin(yaw)          # flight direction through the gate
        x, y = g["x"], g["y"]
        n = str(g["gate"])

        if g.get("double_stack") and double_mode == "upper-lower":
            zu, zl = g["z_upper"], g["z_lower"]
            A = max(approach, 2.5)                      # longer, gentler climb into the top opening
            r = (zu - zl) / 2.0
            zc = (zu + zl) / 2.0
            along = lambda t: (x + t * dx, y + t * dy)  # t>0 = beyond the gate in flight direction
            add(*along(-A), zu, f"pre{n}U", f"{n}U")
            add(x, y, zu, f"GATE{n}U", f"{n}U")
            add(*along(turn_out), zu, f"post{n}U", f"{n}U")
            for deg in (30, 60, 90, 120, 150):          # vertical half-loop, bulging past the gate
                th = math.radians(deg)
                add(*along(turn_out + r * math.sin(th)), zc + r * math.cos(th), f"loop{n}.{deg}")
            add(*along(turn_out), zl, f"pre{n}L", f"{n}L")      # now heading back toward the gate
            add(x, y, zl, f"GATE{n}L", f"{n}L")
            add(*along(-A), zl, f"post{n}L", f"{n}L")
        else:
            z = g["z_upper"] if (g.get("double_stack") and double_mode == "upper") else g["z_path"]
            add(x - approach * dx, y - approach * dy, z, f"pre{n}", n)
            add(x, y, z, f"GATE{n}", n)
            add(x + approach * dx, y + approach * dy, z, f"post{n}", n)

        for i, (xf, yf) in enumerate(VIA_AFTER_GATE_FT.get(g["gate"], [])):
            add(xf * FT, (TRACK_L_FT - yf) * FT, route[-1]["z"], f"via{n}.{i}")
    add(sx, sy, gates[-1]["z_path"], "home_above")
    return route, meta


def enu_to_ned(route, meta, n0, e0, d0):
    st = meta["start"]
    out = []
    for p in route:
        out.append({**p,
                    "n": (p["y"] - st["y"]) + n0,
                    "e": (p["x"] - st["x"]) + e0,
                    "d": d0 - (p["z"] - st["z"])})
    return out


class Polyline:
    def __init__(self, pts):
        self.pts = pts
        self.cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + self._dist(a, b))
        self.total = self.cum[-1]

    @staticmethod
    def _dist(a, b):
        return math.sqrt((a["n"] - b["n"]) ** 2 + (a["e"] - b["e"]) ** 2 + (a["d"] - b["d"]) ** 2)

    def at(self, s):
        """Position and unit tangent at arc length s."""
        s = max(0.0, min(self.total, s))
        for i in range(len(self.pts) - 1):
            if s <= self.cum[i + 1] or i == len(self.pts) - 2:
                a, b = self.pts[i], self.pts[i + 1]
                seg = self.cum[i + 1] - self.cum[i]
                t = 0.0 if seg < 1e-9 else (s - self.cum[i]) / seg
                pos = (a["n"] + t * (b["n"] - a["n"]), a["e"] + t * (b["e"] - a["e"]), a["d"] + t * (b["d"] - a["d"]))
                if seg < 1e-9:
                    return pos, (0.0, 0.0, 0.0)
                tan = ((b["n"] - a["n"]) / seg, (b["e"] - a["e"]) / seg, (b["d"] - a["d"]) / seg)
                return pos, tan
        raise RuntimeError("unreachable")


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def slew(cur, tgt, max_step):
    diff = wrap180(tgt - cur)
    return wrap180(cur + max(-max_step, min(max_step, diff)))


def print_route(route, ned=None):
    print(f"{'#':>2} {'label':<11} {'x(E)':>7} {'y(N)':>7} {'z':>5}" + ("   n      e      d" if ned else ""))
    for i, p in enumerate(route):
        line = f"{i:>2} {p['label']:<11} {p['x']:7.2f} {p['y']:7.2f} {p['z']:5.2f}"
        if ned:
            line += f"  {ned[i]['n']:6.2f} {ned[i]['e']:6.2f} {ned[i]['d']:6.2f}"
        print(line)


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


async def set_params(drone):
    for name, val in PX4_PARAMS_INT.items():
        try:
            await drone.param.set_param_int(name, val)
            print(f"  param {name} = {val}")
        except Exception as e:  # param may not exist in this PX4 version
            print(f"  (skipped {name}: {e})")


async def run(args):
    route, meta = build_route(args.gates, args.approach, args.double_mode)

    if args.dry_run:
        ned = enu_to_ned(route, meta, 0, 0, 0)
        print_route(route, ned)
        pl = Polyline(ned)
        print(f"\npath length {pl.total:.1f} m  ->  ~{pl.total / args.speed:.0f} s at {args.speed} m/s")
        return

    from mavsdk import System
    from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityNedYaw

    drone = System()
    print(f"Connecting to {args.address} ...")
    await drone.connect(system_address=args.address)
    await wait_for(drone.core.connection_state(), lambda s: s.is_connected, args.connect_timeout, "PX4 connection")
    print("Connected.")

    try:
        await drone.telemetry.set_rate_position_velocity_ned(20.0)
    except Exception:
        pass

    if not args.no_params:
        print("Setting PX4 params for GCS-less flight:")
        await set_params(drone)

    print("Waiting for position estimate ...")
    await wait_for(drone.telemetry.health(),
                   lambda h: h.is_global_position_ok and h.is_home_position_ok, 90, "GPS/home health")

    # latest local-frame state, updated in the background
    state = {"n": 0.0, "e": 0.0, "d": 0.0}

    async def track_state():
        async for pv in drone.telemetry.position_velocity_ned():
            state["n"], state["e"], state["d"] = pv.position.north_m, pv.position.east_m, pv.position.down_m

    state_task = asyncio.ensure_future(track_state())
    await asyncio.sleep(1.0)
    n0, e0, d0 = state["n"], state["e"], state["d"]
    heading0 = (await first(drone.telemetry.heading())).heading_deg

    ned = enu_to_ned(route, meta, n0, e0, d0)
    pl = Polyline(ned)
    print(f"Route: {len(ned)} points, {pl.total:.1f} m, ~{pl.total / args.speed:.0f} s at {args.speed} m/s")

    # arc-length window of each gate pass (pre -> post) for miss-distance reporting
    windows, gate_s, gate_pt, gate_order = {}, {}, {}, []
    for i, p in enumerate(ned):
        if p["label"].startswith("pre") and p["gate"]:
            windows[p["gate"]] = [pl.cum[i], None]
        elif p["label"].startswith("post") and p["gate"]:
            windows[p["gate"]][1] = pl.cum[i]
        elif p["label"].startswith("GATE"):
            gate_s[p["gate"]] = pl.cum[i]
            gate_pt[p["gate"]] = p
            gate_order.append(p["gate"])
    best = {n: 1e9 for n in gate_order}
    passed = set()

    takeoff_alt = route[1]["z"] - meta["start"]["z"]   # height above the start pad
    try:
        print(f"Arming, taking off to {takeoff_alt:.2f} m ...")
        await drone.action.set_takeoff_altitude(takeoff_alt)
        await drone.action.arm()
        await drone.action.takeoff()
        await wait_for(drone.telemetry.position(),
                       lambda p: p.relative_altitude_m >= takeoff_alt - 0.3, 30, "takeoff altitude")
        await asyncio.sleep(2.0)

        hover = (state["n"], state["e"], state["d"])
        await drone.offboard.set_position_velocity_ned(
            PositionNedYaw(*hover, heading0), VelocityNedYaw(0, 0, 0, heading0))
        try:
            await drone.offboard.start()
        except OffboardError as e:
            raise SystemExit(f"Offboard start failed: {e._result.result}")
        print("Offboard active - flying route.")

        dt, s, yaw = 0.05, 0.0, heading0
        t_start = time.time()
        t_log = 0.0
        end_wait = None
        while True:
            pos, tan = pl.at(s)
            dn, de, dd = pos[0] - state["n"], pos[1] - state["e"], pos[2] - state["d"]
            err = math.sqrt(dn * dn + de * de + dd * dd)
            advancing = err < args.lead and s < pl.total
            if advancing:
                s = min(pl.total, s + args.speed * dt)
            vel = tuple(args.speed * c for c in tan) if advancing else (0.0, 0.0, 0.0)
            # camera-forward heading: look at a point a few meters AHEAD on the path, so the
            # nose turns toward the upcoming gate before the drone gets there
            ahead, _ = pl.at(s + args.look_ahead)
            bn, be = ahead[0] - pos[0], ahead[1] - pos[1]
            if math.hypot(bn, be) > 0.5:               # skip during pure climbs/descents
                yaw = slew(yaw, math.degrees(math.atan2(be, bn)), args.yaw_rate * dt)
            await drone.offboard.set_position_velocity_ned(
                PositionNedYaw(pos[0], pos[1], pos[2], yaw), VelocityNedYaw(vel[0], vel[1], vel[2], yaw))

            # bookkeeping: gate crossings + miss distances
            for n, p in gate_pt.items():
                s0, s1 = windows[n]
                if s0 - 0.5 <= s <= s1 + 0.5:
                    dist = math.sqrt((p["n"] - state["n"]) ** 2 + (p["e"] - state["e"]) ** 2 + (p["d"] - state["d"]) ** 2)
                    best[n] = min(best[n], dist)
                if s >= gate_s[n] and n not in passed:
                    passed.add(n)
                    print(f"  carrot passed gate {n:>3}  (t={time.time() - t_start:5.1f}s)")

            if time.time() - t_log > 2.0:
                t_log = time.time()
                print(f"  s={s:5.1f}/{pl.total:.1f} m  lag={err:4.2f} m  pos=({state['n']:.1f},{state['e']:.1f},{state['d']:.1f})")

            if s >= pl.total:
                end_wait = end_wait or time.time()
                if err < 0.5 or time.time() - end_wait > 8.0:
                    break
            if time.time() - t_start > args.timeout:
                print("Route timeout - aborting.")
                break
            await asyncio.sleep(dt)

        print("\nGate pass report (closest approach of drone to gate center):")
        half = meta["gate_opening_m"] / 2
        for n in gate_order:
            ok = "OK " if best[n] < half - 0.25 else "RISK"
            print(f"  gate {n:>3}: {best[n]:5.2f} m   {ok}")
        print(f"  (opening half-width = {half:.2f} m; RISK = closer to the frame than ~0.25 m margin or missed)")
    except KeyboardInterrupt:
        print("\nInterrupted - landing.")
    finally:
        state_task.cancel()
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
    ap.add_argument("--speed", type=float, default=2.0, help="carrot speed along path, m/s")
    ap.add_argument("--approach", type=float, default=1.5, help="pre/post-gate point distance, m")
    ap.add_argument("--lead", type=float, default=1.5, help="max carrot lead over the drone, m")
    ap.add_argument("--yaw-rate", type=float, default=120.0, help="max heading slew, deg/s")
    ap.add_argument("--look-ahead", type=float, default=3.0,
                    help="point the nose at the path this many meters ahead (camera-forward heading)")
    ap.add_argument("--double-mode", choices=["upper-lower", "upper", "lower"], default="upper-lower",
                    help="how to fly the stacked double gate")
    ap.add_argument("--timeout", type=float, default=300.0, help="abort route after this many s")
    ap.add_argument("--connect-timeout", type=float, default=120.0)
    ap.add_argument("--no-params", action="store_true", help="don't touch PX4 parameters")
    ap.add_argument("--dry-run", action="store_true", help="print the route and exit")
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()