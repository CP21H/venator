import math
import argparse
from pathlib import Path
from utility.vector_helpers import vadd, vsub, vmul, vdot, vnorm, vunit

FT = 0.3048
TRACK_L_FT = 165.0
G = 9.81

# Extra via-points where the racing line loops between gates: gate -> [(X_ft, Y_ft, heading_deg_ENU)]
# After gate 5 the line swings west, hairpins at its western apex (heading south), and returns east.
VIA_AFTER_GATE_FT = {5: [(30.4, 82.4, -90.0)]}

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

def list_args():
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

    return args