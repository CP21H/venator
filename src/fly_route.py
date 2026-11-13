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
from utility.world import World
from utility.trajectory import Trajectory
from utility.utility import build_path, speed_profile, plan_summary, plot_plan, list_args
from utility.planner import Planner
from utility.mavsdk_helpers import wait_for, set_params, first

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
    args = list_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()