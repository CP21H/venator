import asyncio
from utility.utility import PX4_PARAMS_INT, px4_float_params

async def first(aiter):
    """
    Consumes the first item from an asynchronous iterator and returns it.
    
    Args:
        aiter: An asynchronous iterator or generator.
        
    Returns:
        The first item yielded by `aiter`.
    """
    async for x in aiter:
        return x


async def wait_for(aiter, pred, timeout, what):
    """
    Waits for an item from an asynchronous iterator that satisfies a predicate 
    within a specified timeout.
    
    Args:
        aiter: An asynchronous iterator to scan.
        pred: A predicate function that takes an item and returns True if it matches.
        timeout: Maximum time to wait in seconds.
        what: A string description of what is being waited for (used in error messages).
        
    Returns:
        The first item where `pred(x)` evaluates to True.
        
    Raises:
        SystemExit: If the timeout is reached before a matching item is found.
    """
    async def _go():
        async for x in aiter:
            if pred(x):
                return x
    try:
        return await asyncio.wait_for(_go(), timeout)
    except asyncio.TimeoutError:
        raise SystemExit(f"Timed out waiting for {what}")


async def set_params(drone, args):
    """
    Configures integer and floating-point parameters on a drone via MAVSDK/PX4.
    
    Args:
        drone: The initialized drone client object (e.g., MAVSDK system).
        args: Command-line arguments or configuration objects passed to `px4_float_params`.
    """
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