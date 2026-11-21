from utility.utility import build_knots, build_path, speed_profile
from utility.trajectory import Trajectory

# ----------------------------------------------------------------------------
# Planner (minimum-time-ish): coordinate descent on gate crossing offsets + curve widths
# ----------------------------------------------------------------------------
# Finds a fast, safe line through the gates
# Each normal gate has two knobs:
#   offs[i]     slide the crossing point sideways within the opening
#   cs[i]       tangent scale (how wide the curve is around that gate)
# Planner tries small changes to one knob at a time and keeps any change that lowers the cost
class Planner:
    def __init__(self, meta, world, args):
        self.meta, self.world, self.args = meta, world, args
        # Knots / waypoints to pass through
        self.knots = build_knots(meta, args.double_mode) 
        n = len(self.knots)
        # Indices of the knots the optimizer is allowed to adjust
        self.opt_idx = [i for i, k in enumerate(self.knots) if k["opt"]] 
        # Start with every crossing at gate center
        self.offs = [0.0] * n
        # Start with each knots default tangent scale
        self.cs = [k["c"] for k in self.knots]
        # Required clearance, drone must stay away from solid object
        self.need = args.drone_radius + args.plan_margin
        # Largest sideways shift allowed at a gate: half the opening minus drone radius and margins
        self.omax = max(0.0, world.W / 2 - args.drone_radius - args.plan_margin - 0.05)

    # Continue here: CC538
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