#!/usr/bin/env python3
"""
Generate a Gazebo (gz sim / Harmonic) world + QGroundControl plan for a PX4 SITL
autonomous drone-racing track inside a warehouse. Pure stdlib.

Source of truth: the contestant PDF (feet, origin top-left, +Y down, rotation =
clockwise from "up the map").  Converted here to Gazebo ENU meters with the
origin at the BOTTOM-LEFT corner of the track boundary (x east, y north):
    x = X_ft * 0.3048
    y = (165 - Y_ft) * 0.3048
    gazebo_yaw = 90 deg - pdf_rotation

Usage:
    python3 make_track_world.py
    python3 make_track_world.py --gate-open 3.0 --lat 39.81 --lon -82.93 --alt 220
    python3 make_track_world.py --no-hall
Outputs: drone_track.sdf, gates.json, track.plan
"""
import argparse
import json
import math
from pathlib import Path

FT = 0.3048
TRACK_W_FT, TRACK_L_FT = 85.0, 165.0

# --------------------------------------------------------------------------
# GATES from the PDF:  (number, X_ft, Y_ft, rotation_deg, stacked_double?)
# --------------------------------------------------------------------------
GATES_FT = [
    (1, 12.0, 71.0, 0, False),     # START gate
    (2, 16.0, 39.0, 0, False),
    (3, 40.0, 17.0, 90, False),
    (4, 72.0, 40.0, 180, False),
    (5, 66.0, 70.0, 215, False),
    (6, 41.0, 86.0, 90, False),
    (7, 70.5, 106.6, 180, False),
    (8, 60.0, 129.0, 215, False),
    (9, 39.7, 147.7, 180, True),   # stacked DOUBLE gate (southernmost)
    (10, 13.0, 110.0, 0, False),
]
# Orange cones from the PDF map (X_ft, Y_ft) - read off the picture (~3.69 px/ft)
CONES_FT = [(0, 0), (52.5, 0), (0, 60), (52.5, 60), (0, 120), (52.5, 120)]
# Drone start pad (yellow dotted marker on the map), just behind gate 1
START_FT = (9.5, 96.0)

TRACK_BBOX = (-2.0, -2.0, TRACK_W_FT * FT + 2, TRACK_L_FT * FT + 2)

# Warehouse (estimates from photos - measure the real hall if you can)
HALL = dict(x0=-30.0, x1=56.0, y0=-30.0, y1=80.0, h=12.0)
COLUMN_SPACING = 20.0
LIGHT_SPACING = 22.0

COLORS = {
    "yellow": (0.95, 0.68, 0.10, 1), "blue": (0.15, 0.55, 0.90, 1),
    "red": (0.88, 0.14, 0.08, 1), "orange": (0.95, 0.45, 0.05, 1),
    "concrete": (0.55, 0.55, 0.57, 1), "wall": (0.38, 0.40, 0.42, 1),
    "ceiling": (0.12, 0.12, 0.13, 1), "steel": (0.45, 0.50, 0.55, 1),
    "white": (1, 1, 1, 1),
}


def to_enu(x_ft, y_ft):
    return round(x_ft * FT, 3), round((TRACK_L_FT - y_ft) * FT, 3)


def c(name):
    return " ".join(str(v) for v in COLORS[name])


def box_visual(name, size, pose, color, emissive=None, collision=True, rough=None):
    sx, sy, sz = size
    px, py, pz = pose[:3]
    yaw = pose[5] if len(pose) > 5 else 0.0
    pose_s = f"{px:.4f} {py:.4f} {pz:.4f} 0 0 {yaw:.5f}"
    em = f"<emissive>{c(emissive)}</emissive>" if emissive else ""
    pbr = ""
    if rough is not None:
        pbr = f"<pbr><metal><metalness>0.0</metalness><roughness>{rough}</roughness></metal></pbr>"
    col = ""
    if collision:
        col = (f"<collision name='{name}_col'><pose>{pose_s}</pose>"
               f"<geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>")
    return (f"<visual name='{name}'><pose>{pose_s}</pose>"
            f"<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>"
            f"<material><ambient>{c(color)}</ambient><diffuse>{c(color)}</diffuse>{em}{pbr}</material>"
            f"</visual>{col}")


def frame_parts(prefix, zrel, W, H, T, D, color):
    """One square frame in the local y-z plane, normal along local +x, centered at z=zrel."""
    return [
        box_visual(f"{prefix}left", (D, T, H + 2 * T), (0, -(W / 2 + T / 2), zrel), color),
        box_visual(f"{prefix}right", (D, T, H + 2 * T), (0, (W / 2 + T / 2), zrel), color),
        box_visual(f"{prefix}top", (D, W, T), (0, 0, zrel + H / 2 + T / 2), color),
        box_visual(f"{prefix}bottom", (D, W, T), (0, 0, zrel - (H / 2 + T / 2)), color),
        box_visual(f"{prefix}led_l", (D + 0.02, 0.03, H), (0, -(W / 2 - 0.015), zrel), "white", emissive="white", collision=False),
        box_visual(f"{prefix}led_r", (D + 0.02, 0.03, H), (0, (W / 2 - 0.015), zrel), "white", emissive="white", collision=False),
    ]


def gate_model(name, x, y, zc, yaw_deg, color, opening, bar, depth, stacked):
    W = H = opening
    T, D = bar, depth
    parts = frame_parts("a_", 0.0, W, H, T, D, color)
    if stacked:
        # second frame directly on top, sharing the dividing bar
        parts += frame_parts("b_", H + T, W, H, T, D, color)
    stand_h = zc - H / 2 - T
    if stand_h > 0.05:
        parts.append(box_visual("stand", (0.12, 0.12, stand_h), (0, 0, -(zc - stand_h / 2)), "steel"))
    return (f"<model name='{name}'><static>true</static>"
            f"<pose>{x} {y} {zc} 0 0 {math.radians(yaw_deg):.5f}</pose>"
            f"<link name='link'>{''.join(parts)}</link></model>")


def cone_model(name, x, y, color):
    r, l = 0.2, 0.7
    geom = f"<cone><radius>{r}</radius><length>{l}</length></cone>"
    return (f"<model name='{name}'><static>true</static><pose>{x} {y} {l / 2} 0 0 0</pose>"
            f"<link name='link'><visual name='v'><geometry>{geom}</geometry>"
            f"<material><ambient>{c(color)}</ambient><diffuse>{c(color)}</diffuse></material></visual>"
            f"<collision name='c'><geometry>{geom}</geometry></collision></link></model>")


def hall_model():
    x0, x1, y0, y1, h = (HALL[k] for k in ("x0", "x1", "y0", "y1", "h"))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lx, ly = x1 - x0, y1 - y0
    t = 0.3
    p = [box_visual("floor", (lx, ly, 0.2), (cx, cy, -0.1), "concrete", rough=0.25),
         box_visual("wall_s", (lx, t, h), (cx, y0, h / 2), "wall"),
         box_visual("wall_n", (lx, t, h), (cx, y1, h / 2), "wall"),
         box_visual("wall_w", (t, ly, h), (x0, cy, h / 2), "wall"),
         box_visual("wall_e", (t, ly, h), (x1, cy, h / 2), "wall"),
         box_visual("ceiling", (lx, ly, 0.2), (cx, cy, h + 0.1), "ceiling")]
    n, y = 0, y0 + 5
    while y < y1:
        p.append(box_visual(f"truss_y{n}", (lx, 0.25, 0.6), (cx, y, h - 0.3), "steel", collision=False))
        y += 6.0
        n += 1
    bx0, by0, bx1, by1 = TRACK_BBOX
    n = 0
    xs = [x0 + 6 + i * COLUMN_SPACING for i in range(int((lx - 12) // COLUMN_SPACING) + 1)]
    ys = [y0 + 6 + j * COLUMN_SPACING for j in range(int((ly - 12) // COLUMN_SPACING) + 1)]
    for xx in xs:
        for yy in ys:
            if bx0 - 2 < xx < bx1 + 2 and by0 - 2 < yy < by1 + 2:
                continue
            p.append(box_visual(f"col{n}", (0.6, 0.6, h), (xx, yy, h / 2), "steel"))
            n += 1
    lights, n = [], 0
    xx = x0 + LIGHT_SPACING / 2
    while xx < x1:
        yy = y0 + LIGHT_SPACING / 2
        while yy < y1:
            p.append(box_visual(f"panel{n}", (3.0, 0.4, 0.05), (xx, yy, h - 0.05), "white", emissive="white", collision=False))
            lights.append(
                f"<light name='lamp{n}' type='point'><pose>{xx} {yy} {h - 0.8} 0 0 0</pose>"
                f"<cast_shadows>false</cast_shadows><diffuse>1 1 0.95 1</diffuse><specular>0.2 0.2 0.2 1</specular>"
                f"<attenuation><range>35</range><constant>0.4</constant><linear>0.02</linear><quadratic>0.001</quadratic></attenuation>"
                f"</light>")
            n += 1
            yy += LIGHT_SPACING
        xx += LIGHT_SPACING
    return (f"<model name='warehouse'><static>true</static><link name='link'>{''.join(p)}</link></model>"
            + "".join(lights))


def header(lat, lon, alt):
    return f"""<?xml version="1.0" ?>
<sdf version="1.9">
<world name="drone_track">
  <physics type="ode">
    <max_step_size>0.004</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>250</real_time_update_rate>
  </physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
  <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
  <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
  <plugin filename="gz-sim-air-speed-system" name="gz::sim::systems::AirSpeed"/>
  <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>
  <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
  <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
    <render_engine>ogre2</render_engine>
  </plugin>
  <gravity>0 0 -9.8</gravity>
  <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
  <atmosphere type="adiabatic"/>
  <scene>
    <ambient>0.45 0.45 0.45 1</ambient>
    <background>0.2 0.2 0.22 1</background>
    <shadows>false</shadows>
  </scene>
  <!-- World origin (0,0,0) = bottom-left corner of the track boundary. -->
  <spherical_coordinates>
    <surface_model>EARTH_WGS84</surface_model>
    <world_frame_orientation>ENU</world_frame_orientation>
    <latitude_deg>{lat}</latitude_deg>
    <longitude_deg>{lon}</longitude_deg>
    <elevation>{alt}</elevation>
  </spherical_coordinates>
"""


OPEN_GROUND = """
  <model name="ground_plane"><static>true</static><link name="link">
    <collision name="c"><geometry><plane><normal>0 0 1</normal><size>300 300</size></plane></geometry></collision>
    <visual name="v"><geometry><plane><normal>0 0 1</normal><size>300 300</size></plane></geometry>
    <material><ambient>0.5 0.5 0.5 1</ambient><diffuse>0.5 0.5 0.5 1</diffuse></material></visual>
  </link></model>
  <light name="sun" type="directional"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose>
    <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.5 0.1 -0.9</direction></light>
"""


def enu_to_latlon(x, y, lat0, lon0):
    """Local ENU meters -> lat/lon (fine over tens of meters)."""
    dlat = y / 111132.92
    dlon = x / (111412.84 * math.cos(math.radians(lat0)))
    return lat0 + dlat, lon0 + dlon


def write_plan(path, wps, bounds, lat0, lon0, alt0, start_xy):
    """QGroundControl .plan: takeoff + one waypoint per gate + boundary geofence."""
    items, jid = [], 1
    slat, slon = enu_to_latlon(*start_xy, lat0, lon0)
    items.append({"autoContinue": True, "command": 22, "doJumpId": jid, "frame": 3,
                  "params": [0, 0, 0, None, slat, slon, wps[0]["z"]], "type": "SimpleItem"})
    for w in wps:
        jid += 1
        lat, lon = enu_to_latlon(w["x"], w["y"], lat0, lon0)
        items.append({"autoContinue": True, "command": 16, "doJumpId": jid, "frame": 3,
                      "params": [0, 0, 0, None, lat, lon, w["z"]], "type": "SimpleItem"})
    jid += 1
    items.append({"autoContinue": True, "command": 20, "doJumpId": jid, "frame": 2,
                  "params": [0, 0, 0, 0, 0, 0, 0], "type": "SimpleItem"})
    poly = [list(enu_to_latlon(x, y, lat0, lon0)) for x, y in bounds]
    plan = {
        "fileType": "Plan", "version": 1, "groundStation": "QGroundControl",
        "geoFence": {"circles": [], "polygons": [{"inclusion": True, "polygon": poly, "version": 1}], "version": 2},
        "rallyPoints": {"points": [], "version": 2},
        "mission": {"cruiseSpeed": 5, "hoverSpeed": 5, "firmwareType": 12, "vehicleType": 2,
                    "globalPlanAltitudeMode": 1, "plannedHomePosition": [slat, slon, alt0],
                    "items": items, "version": 2},
    }
    Path(path).write_text(json.dumps(plan, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="drone_track.sdf")
    ap.add_argument("--gate-open", type=float, default=2.7,
                    help="square inner opening, m (PLACEHOLDER - the PDF doesn't give it; set the real size)")
    ap.add_argument("--bar", type=float, default=0.2, help="frame bar thickness, m")
    ap.add_argument("--depth", type=float, default=0.2, help="frame depth along flight direction, m")
    ap.add_argument("--no-hall", action="store_true")
    ap.add_argument("--double-pass", choices=["lower", "upper"], default="lower",
                    help="which opening of the stacked double gate the route flies through")
    ap.add_argument("--lat", type=float, default=47.397971057728974,
                    help="latitude of the track's bottom-left corner (world origin)")
    ap.add_argument("--lon", type=float, default=8.546163739800146)
    ap.add_argument("--alt", type=float, default=289.0, help="elevation AMSL, m")
    a = ap.parse_args()

    zc = a.gate_open / 2 + a.bar + 0.3     # lowest gate sits ~0.3 m above the floor
    body = [header(a.lat, a.lon, a.alt)]
    body.append(OPEN_GROUND if a.no_hall else hall_model())

    meta_gates, wps = [], []
    for num, xf, yf, rot, stacked in GATES_FT:
        x, y = to_enu(xf, yf)
        yaw = (90 - rot + 180) % 360 - 180
        color = "yellow" if num == 1 else "blue"
        body.append(gate_model(f"gate{num:02d}", x, y, zc, yaw, color, a.gate_open, a.bar, a.depth, stacked))
        # NOTE: the middle of a stacked pair is the divider bar - fly through one opening instead
        z_mid = zc + ((a.gate_open + a.bar) if (stacked and a.double_pass == "upper") else 0)
        meta_gates.append({"gate": num, "x": x, "y": y, "z_center": round(zc, 3), "z_path": round(z_mid, 3),
                           "z_lower": round(zc, 3),
                           "z_upper": round(zc + a.gate_open + a.bar, 3) if stacked else None,
                           "yaw_deg": yaw, "pdf_rotation_deg": rot, "double_stack": stacked})
        wps.append({"x": x, "y": y, "z": round(z_mid, 2)})

    for i, (xf, yf) in enumerate(CONES_FT):
        x, y = to_enu(xf, yf)
        body.append(cone_model(f"cone{i}", x, y, "orange"))
    body.append("</world></sdf>")
    Path(a.out).write_text("\n".join(body))

    sx, sy = to_enu(*START_FT)
    meta = {"origin_latlon_alt": [a.lat, a.lon, a.alt], "start": {"x": sx, "y": sy, "z": 0.2, "yaw_rad": round(math.pi / 2, 4)},
            "gate_opening_m": a.gate_open, "gates_in_order": meta_gates}
    Path("gates.json").write_text(json.dumps(meta, indent=2))

    w, l = TRACK_W_FT * FT, TRACK_L_FT * FT
    write_plan("track.plan", wps, [(0, 0), (w, 0), (w, l), (0, l)], a.lat, a.lon, a.alt, (sx, sy))

    print(f"wrote {a.out}, gates.json, track.plan   (gate opening {a.gate_open} m)")
    print(f'PX4_GZ_MODEL_POSE="{sx},{sy},0.2,0,0,1.571"')


if __name__ == "__main__":
    main()