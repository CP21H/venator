#!/usr/bin/env bash
# One-command launcher: PX4 SITL + Gazebo (drone_track world) + fixed-route flight script.
#
#   ./run_sim.sh                      # GUI on, fly the route at 2 m/s
#   ./run_sim.sh --headless           # no Gazebo window
#   ./run_sim.sh -- --speed 3.0       # everything after -- goes to fly_route.py
#   ./run_sim.sh --sim-only           # start the sim and leave it running (fly it yourself)
#
# Env overrides:  PX4_DIR=~/vscode/PX4-Autopilot   VEHICLE=gz_x500_mono_cam   PYTHON=python3
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/vscode/PX4-Autopilot}"
VEHICLE="${VEHICLE:-gz_x500_mono_cam}"
PYTHON="${PYTHON:-python3}"
WORLD_NAME="drone_track"
HEADLESS=0
SIM_ONLY=0
FLY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless) HEADLESS=1; shift ;;
    --sim-only) SIM_ONLY=1; shift ;;
    --) shift; FLY_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

[[ -d "$PX4_DIR" ]] || { echo "PX4_DIR not found: $PX4_DIR (set PX4_DIR=...)"; exit 1; }
[[ -f "$HERE/drone_track.sdf" && -f "$HERE/gates.json" ]] || { echo "Run make_track_world.py first."; exit 1; }

# 1. install the world and read the start pose from gates.json
cp "$HERE/drone_track.sdf" "$PX4_DIR/Tools/simulation/gz/worlds/${WORLD_NAME}.sdf"
POSE="$("$PYTHON" - "$HERE/gates.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))["start"]
print(f'{s["x"]},{s["y"]},{s["z"]},0,0,{s["yaw_rad"]}')
EOF
)"
echo "World: $WORLD_NAME   spawn pose: $POSE   vehicle: $VEHICLE"

# 2. clean up leftovers from previous runs, and again on exit
kill_sim() {
  pkill -f "build/px4_sitl_default/bin/px4" 2>/dev/null
  pkill -f "gz sim" 2>/dev/null
  pkill -f "gz-sim" 2>/dev/null
}
cleanup() {
  echo; echo "Shutting down simulation ..."
  [[ -n "${PX4_PID:-}" ]] && kill "$PX4_PID" 2>/dev/null
  kill_sim
}
trap cleanup EXIT INT TERM
kill_sim; sleep 1

# 3. start PX4 + Gazebo in the background. PX4's shell wants an open stdin,
#    so we feed it from `tail -f /dev/null` instead of letting it see EOF.
LOG="$(mktemp -t px4_sim.XXXXXX)"
export PX4_GZ_WORLD="$WORLD_NAME" PX4_GZ_MODEL_POSE="$POSE"
[[ $HEADLESS -eq 1 ]] && export HEADLESS=1
( cd "$PX4_DIR" && tail -f /dev/null | make px4_sitl "$VEHICLE" ) >"$LOG" 2>&1 &
PX4_PID=$!
echo "PX4 log: $LOG"

# 4. wait for PX4 to finish starting (first build can take several minutes)
echo -n "Waiting for PX4 to start "
for i in $(seq 1 600); do
  if grep -q "Startup script returned successfully" "$LOG" 2>/dev/null; then echo " ready."; break; fi
  if ! kill -0 "$PX4_PID" 2>/dev/null; then echo; echo "PX4 exited early. Last log lines:"; tail -n 25 "$LOG"; exit 1; fi
  if grep -q "Error Code" "$LOG" 2>/dev/null && grep -q "Unable to read file" "$LOG"; then
    echo; echo "Gazebo could not load the world:"; grep -E "Err|Error" "$LOG" | head; exit 1
  fi
  echo -n "."; sleep 1
done
grep -q "Startup script returned successfully" "$LOG" || { echo; echo "Timed out waiting for PX4."; tail -n 25 "$LOG"; exit 1; }
grep -E "Starting gazebo with world|world: " "$LOG" | head -2
sleep 3   # let Gazebo finish spawning the vehicle

# 5. fly (or just leave the sim up)
if [[ $SIM_ONLY -eq 1 ]]; then
  echo "Simulation running. Press Ctrl-C to stop."
  wait "$PX4_PID"
else
  "$PYTHON" "$HERE/fly_route.py" --gates "$HERE/gates.json" ${FLY_ARGS[@]+"${FLY_ARGS[@]}"}
  echo "Flight script finished."
fi
