import bisect
from utility.vector_helpers import vlerp, vmul, vadd, vsub, vnorm, vunit

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