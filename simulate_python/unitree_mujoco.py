import os
import shutil
import sys
import time
import threading
from threading import Thread

# On macOS, launch_passive requires mjpython. Re-exec with mjpython once (use env var to
# avoid infinite re-exec when mjpython invokes python and sys.executable stays "python").
if sys.platform == "darwin" and os.environ.get("MUJOCO_MJPYTHON_REELEXEC") != "1":
    mjpython_path = shutil.which("mjpython")
    if not mjpython_path:
        bin_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidate = os.path.join(bin_dir, "mjpython")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            mjpython_path = candidate
    if mjpython_path:
        os.environ["MUJOCO_MJPYTHON_REELEXEC"] = "1"
        os.execv(mjpython_path, [mjpython_path] + sys.argv)
    print(
        "On macOS, MuJoCo's passive viewer requires mjpython.\n"
        "Run: mjpython unitree_mujoco.py\n"
        "Or set the Python interpreter to mjpython in your IDE.",
        file=sys.stderr,
    )
    sys.exit(1)

# So play.py and sim use the same LCM URL when sim is started by play.
for i, a in enumerate(sys.argv):
    if a == "--lcm-url" and i + 1 < len(sys.argv):
        os.environ["LCM_DEFAULT_URL"] = sys.argv[i + 1]
        break

import mujoco
import mujoco.viewer

from unitree_lcm_bridge import ChannelFactoryInitialize, UnitreeSdk2Bridge, ElasticBand

import config

# On macOS, pygame must run on the main thread. The sim thread does not call pygame;
# main_loop() (main thread) reads the joystick and sets external state on the bridge.
USE_JOYSTICK = config.USE_JOYSTICK
JOYSTICK_ON_MAIN_THREAD = USE_JOYSTICK and sys.platform == "darwin"
if USE_JOYSTICK and not JOYSTICK_ON_MAIN_THREAD:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
        import pygame
        pygame.init()
        pygame.joystick.init()

# Bridge reference so main_loop (main thread) can set external joystick state on macOS.
bridge_ref = [None]
locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data, data_lock=locker)
    bridge_ref[0] = unitree

    if USE_JOYSTICK:
        if JOYSTICK_ON_MAIN_THREAD:
            unitree.SetJoystickMapping(config.JOYSTICK_TYPE)
        else:
            unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


def main_loop():
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    viewer_thread.start()
    sim_thread.start()

    while True:
        if USE_JOYSTICK:
            if JOYSTICK_ON_MAIN_THREAD and bridge_ref[0] is not None:
                from unitree_lcm_bridge import WirelessController_
                w = WirelessController_()
                bridge_ref[0].set_external_wireless_controller(w)
            else:
                if not JOYSTICK_ON_MAIN_THREAD:
                    import pygame
                    pygame.event.pump()
        viewer_thread.join(timeout=0.02)
        sim_thread.join(timeout=0.02)
        if not viewer_thread.is_alive() and not sim_thread.is_alive():
            break


if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        import traceback
        print("Simulator exited with error:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
