import asyncio
from utility.utility import PX4_PARAMS_INT, px4_float_params

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