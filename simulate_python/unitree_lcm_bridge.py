"""
Unitree LCM bridge for environments where unitree_sdk2py is not available (e.g. Apple Silicon).

Same API: ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
RecurrentThread, UnitreeSdk2Bridge, ElasticBand, and Go2 message types (LowCmd_, LowState_, etc.).
"""

import json
import os
import struct
import sys
import threading
import time
from typing import Any, Callable, Optional


try:
    import lcm
except ImportError:
    lcm = None  # type: ignore

# Topic names (same as Unitree; LCM channel uses '/' -> '_')
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"

MOTOR_SENSOR_NUM = 3
NUM_MOTOR_IDL_GO = 20
NUM_MOTOR_IDL_HG = 35

# ---------- Message classes (Unitree Go2 IDL compatible) ----------


class IMUState_:
    def __init__(self):
        self.quaternion = [0.0] * 4
        self.gyroscope = [0.0] * 3
        self.accelerometer = [0.0] * 3
        self.rpy = [0.0] * 3
        self.temperature = 0


class MotorState_:
    def __init__(self):
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.ddq = 0.0
        self.tau_est = 0.0
        self.q_raw = 0.0
        self.dq_raw = 0.0
        self.ddq_raw = 0.0
        self.temperature = 0
        self.lost = 0
        self.reserve = [0, 0]


class MotorCmd_:
    def __init__(self):
        self.mode = 0x01
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0
        self.reserve = [0, 0, 0]


class LowState_:
    """Unitree Go2 LowState compatible (bridge uses motor_state, imu_state, wireless_remote)."""

    def __init__(self):
        self.head = [0, 0]
        self.level_flag = 0
        self.frame_reserve = 0
        self.sn = [0, 0]
        self.version = [0, 0]
        self.bandwidth = 0
        self.imu_state = IMUState_()
        self.motor_state = [MotorState_() for _ in range(20)]
        self.foot_force = [0] * 4
        self.foot_force_est = [0] * 4
        self.tick = 0
        self.wireless_remote = bytearray(40)
        self.bit_flag = 0
        self.adc_reel = 0.0
        self.temperature_ntc1 = 0
        self.temperature_ntc2 = 0
        self.power_v = 0.0
        self.power_a = 0.0
        self.fan_frequency = [0] * 4
        self.reserve = 0
        self.crc = 0


class LowCmd_:
    """Unitree Go2 LowCmd compatible."""

    def __init__(self):
        self.head = [0xFE, 0xEF]
        self.level_flag = 0xFF
        self.frame_reserve = 0
        self.sn = [0, 0]
        self.version = [0, 0]
        self.bandwidth = 0
        self.motor_cmd = [MotorCmd_() for _ in range(20)]
        self.wireless_remote = bytearray(40)
        self.led = bytearray(12)
        self.fan = bytearray(2)
        self.gpio = 0
        self.reserve = 0
        self.crc = 0


class SportModeState_:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]


class WirelessController_:
    def __init__(self):
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.keys = 0


# Aliases for test/legacy code compatibility
unitree_go_msg_dds__SportModeState_ = SportModeState_
unitree_go_msg_dds__WirelessController_ = WirelessController_
unitree_go_msg_dds__LowCmd_ = LowCmd_
unitree_go_msg_dds__LowState_ = LowState_

# ---------- Serialization (JSON-based LCM payload) ----------


def _msg_to_dict(obj: Any) -> Any:
    if isinstance(obj, bytearray):
        return list(obj)
    if isinstance(obj, (list, tuple)):
        return [_msg_to_dict(x) for x in obj]
    if hasattr(obj, "__dict__") and not isinstance(
        obj, (dict, str, bytes, int, float, type(None), bool)
    ):
        d = {}
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            d[k] = _msg_to_dict(v)
        return d
    return obj


def _dict_to_motor_cmd(d: dict) -> MotorCmd_:
    m = MotorCmd_()
    m.mode = d.get("mode", 0x01)
    m.q = float(d.get("q", 0))
    m.dq = float(d.get("dq", 0))
    m.tau = float(d.get("tau", 0))
    m.kp = float(d.get("kp", 0))
    m.kd = float(d.get("kd", 0))
    m.reserve = d.get("reserve", [0, 0, 0])
    return m


def _dict_to_low_cmd(d: dict) -> LowCmd_:
    cmd = LowCmd_()
    cmd.head = d.get("head", [0xFE, 0xEF])
    cmd.level_flag = d.get("level_flag", 0xFF)
    cmd.gpio = d.get("gpio", 0)
    cmd.crc = d.get("crc", 0)
    motor_cmd = d.get("motor_cmd", [])
    for i, mc in enumerate(motor_cmd):
        if i >= 20:
            break
        if isinstance(mc, dict):
            cmd.motor_cmd[i] = _dict_to_motor_cmd(mc)
    return cmd


def _dict_to_motor_state(d: dict) -> MotorState_:
    m = MotorState_()
    m.mode = d.get("mode", 0)
    m.q = float(d.get("q", 0))
    m.dq = float(d.get("dq", 0))
    m.tau_est = float(d.get("tau_est", 0))
    return m


def _dict_to_imu_state(d: dict) -> IMUState_:
    imu = IMUState_()
    imu.quaternion = list(d.get("quaternion", [0] * 4))[:4]
    imu.gyroscope = list(d.get("gyroscope", [0] * 3))[:3]
    imu.accelerometer = list(d.get("accelerometer", [0] * 3))[:3]
    return imu


def _dict_to_low_state(d: dict) -> LowState_:
    s = LowState_()
    motor_state = d.get("motor_state", [])
    for i, ms in enumerate(motor_state):
        if i >= 20:
            break
        if isinstance(ms, dict):
            s.motor_state[i] = _dict_to_motor_state(ms)
    imu = d.get("imu_state")
    if isinstance(imu, dict):
        s.imu_state = _dict_to_imu_state(imu)
    wr = d.get("wireless_remote")
    if wr is not None:
        s.wireless_remote = bytearray(wr[:40]) if isinstance(wr, (list, bytearray)) else bytearray(40)
    return s


def _dict_to_sport_mode_state(d: dict) -> SportModeState_:
    s = SportModeState_()
    s.position = list(d.get("position", [0, 0, 0]))[:3]
    s.velocity = list(d.get("velocity", [0, 0, 0]))[:3]
    return s


def _dict_to_wireless_controller(d: dict) -> WirelessController_:
    w = WirelessController_()
    w.lx = float(d.get("lx", 0))
    w.ly = float(d.get("ly", 0))
    w.rx = float(d.get("rx", 0))
    w.ry = float(d.get("ry", 0))
    w.keys = int(d.get("keys", 0))
    return w


_TYPE_FROM_DICT = {
    "LowCmd_": _dict_to_low_cmd,
    "LowState_": _dict_to_low_state,
    "SportModeState_": _dict_to_sport_mode_state,
    "WirelessController_": _dict_to_wireless_controller,
}


def _serialize(msg: Any, type_name: str) -> bytes:
    d = {"__type__": type_name}
    for k, v in vars(msg).items():
        if k.startswith("_"):
            continue
        d[k] = _msg_to_dict(v)
    return json.dumps(d, default=lambda x: list(x) if isinstance(x, bytearray) else x).encode("utf-8")


def _deserialize(data: bytes, type_name: str) -> Any:
    d = json.loads(data.decode("utf-8"))
    t = d.pop("__type__", type_name)
    fn = _TYPE_FROM_DICT.get(t)
    if fn:
        return fn(d)
    return d


# ---------- CRC (LowCmd CRC32 compatible; LCM version returns 0) ----------


class CRC:
    """Unitree LowCmd CRC. LCM version returns 0 (verification skipped)."""

    def Crc(self, cmd: LowCmd_) -> int:
        # Matching C++ CRC32 would require same binary layout; use 0 for LCM.
        return 0


# ---------- LCM channel wrapper ----------


_lcm_instance: Optional["lcm.LCM"] = None
_domain_id: int = 1
_interface: str = "lo"


def ChannelFactoryInitialize(domain_id: int = 1, interface: str = "lo") -> None:
    global _lcm_instance, _domain_id, _interface
    _domain_id = domain_id
    _interface = interface
    if lcm is None:
        raise RuntimeError("LCM is not installed. Run: pip install lcm")
    if _lcm_instance is None:
        url = os.environ.get("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=255")
        try:
            _lcm_instance = lcm.LCM(url)
        except Exception:
            _lcm_instance = lcm.LCM()


def _get_lcm() -> "lcm.LCM":
    if _lcm_instance is None:
        ChannelFactoryInitialize(1, "lo")
    return _lcm_instance


def process_lcm_messages(timeout_ms: int = 0) -> None:
    """Process pending LCM messages (e.g. call from main loop so callbacks run). timeout_ms=0 for non-blocking."""
    if _lcm_instance is None:
        return
    try:
        _lcm_instance.handle_timeout(timeout_ms)
    except Exception:
        pass


def _topic_to_channel(topic: str) -> str:
    return topic.replace("/", "_")


class ChannelPublisher:
    """Unitree ChannelPublisher API compatible (LCM publish)."""

    def __init__(self, topic: str, msg_type: type):
        self._topic = topic
        self._channel = _topic_to_channel(topic)
        self._msg_type_name = msg_type.__name__
        self._lc = None

    def Init(self) -> None:
        self._lc = _get_lcm()

    def Write(self, msg: Any) -> None:
        if self._lc is None:
            self.Init()
        data = _serialize(msg, self._msg_type_name)
        self._lc.publish(self._channel, data)


class ChannelSubscriber:
    """Unitree ChannelSubscriber API compatible (LCM subscribe)."""

    def __init__(self, topic: str, msg_type: type):
        self._topic = topic
        self._channel = _topic_to_channel(topic)
        self._msg_type = msg_type
        self._msg_type_name = msg_type.__name__
        self._handler: Optional[Callable] = None
        self._subscription = None
        self._lc = None

    def Init(self, handler: Callable, priority: int = 10) -> None:
        self._handler = handler
        self._lc = _get_lcm()

        def _callback(channel: str, data: bytes) -> None:
            try:
                msg = _deserialize(data, self._msg_type_name)
                if not isinstance(msg, self._msg_type):
                    msg = _deserialize(data, self._msg_type_name)
                self._handler(msg)
            except Exception as e:
                print(f"[LCM] callback error: {e}", file=sys.stderr)

        self._subscription = self._lc.subscribe(self._channel, _callback)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._subscription is not None and self._lc is not None:
            self._lc.unsubscribe(self._subscription)


# ---------- RecurrentThread ----------


class RecurrentThread:
    """Thread that periodically calls target() (Unitree RecurrentThread compatible)."""

    def __init__(self, interval: float, target: Callable, name: str = "recurrent"):
        self._interval = interval
        self._target = target
        self._name = name
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def Start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self._target()
            except Exception as e:
                print(f"[{self._name}] error: {e}", file=sys.stderr)
            elapsed = time.perf_counter() - t0
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                self._stop.wait(timeout=sleep_time)
            else:
                self._stop.wait(timeout=0.001)

    def Stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ---------- UnitreeSdk2Bridge (LCM-based) ----------


class UnitreeSdk2Bridge:
    """Bridges MuJoCo sim with Unitree protocol (LCM). Joystick, rt/lowstate, rt/lowcmd, rt/sportmodestate, rt/wirelesscontroller."""

    def __init__(self, mj_model, mj_data, data_lock=None):
        import mujoco
        self._mujoco = mujoco
        self.mj_model = mj_model
        self.mj_data = mj_data
        self._data_lock = data_lock

        self.num_motor = self.mj_model.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.have_imu = False
        self.have_frame_sensor = False
        self.dt = self.mj_model.opt.timestep
        self.idl_type = self.num_motor > NUM_MOTOR_IDL_GO
        self.joystick = None
        self._external_wireless_controller = None
        self._external_wireless_lock = threading.Lock()
        self.axis_id = {}
        self.button_id = {}

        for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
            name = self._mujoco.mj_id2name(
                self.mj_model, self._mujoco._enums.mjtObj.mjOBJ_SENSOR, i
            )
            if name == "imu_quat":
                self.have_imu = True
            if name == "frame_pos":
                self.have_frame_sensor = True

        self.low_state = LowState_()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishLowState, name="sim_lowstate"
        )
        self.lowStateThread.Start()

        self.high_state = SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishHighState, name="sim_highstate"
        )
        self.HighStateThread.Start()

        self.wireless_controller = WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(TOPIC_WIRELESS_CONTROLLER, WirelessController_)
        self.wireless_controller_puber.Init()
        self.WirelessControllerThread = RecurrentThread(
            interval=0.01,
            target=self.PublishWirelessController,
            name="sim_wireless_controller",
        )
        self.WirelessControllerThread.Start()

        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)

        self.key_map = {
            "R1": 0, "L1": 1, "start": 2, "select": 3, "R2": 4, "L2": 5,
            "F1": 6, "F2": 7, "A": 8, "B": 9, "X": 10, "Y": 11,
            "up": 12, "right": 13, "down": 14, "left": 15,
        }

    def LowCmdHandler(self, msg: LowCmd_) -> None:
        if self.mj_data is None:
            return
        if self._data_lock is not None:
            self._data_lock.acquire()
        try:
            for i in range(self.num_motor):
                self.mj_data.ctrl[i] = (
                    msg.motor_cmd[i].tau
                    + msg.motor_cmd[i].kp * (msg.motor_cmd[i].q - self.mj_data.sensordata[i])
                    + msg.motor_cmd[i].kd * (
                        msg.motor_cmd[i].dq - self.mj_data.sensordata[i + self.num_motor]
                    )
                )
        finally:
            if self._data_lock is not None:
                self._data_lock.release()

    def PublishLowState(self) -> None:
        if self.mj_data is None:
            return
        for i in range(self.num_motor):
            self.low_state.motor_state[i].q = float(self.mj_data.sensordata[i])
            self.low_state.motor_state[i].dq = float(self.mj_data.sensordata[i + self.num_motor])
            self.low_state.motor_state[i].tau_est = float(
                self.mj_data.sensordata[i + 2 * self.num_motor]
            )

        if self.have_frame_sensor:
            off = self.dim_motor_sensor
            self.low_state.imu_state.quaternion[0] = float(self.mj_data.sensordata[off + 0])
            self.low_state.imu_state.quaternion[1] = float(self.mj_data.sensordata[off + 1])
            self.low_state.imu_state.quaternion[2] = float(self.mj_data.sensordata[off + 2])
            self.low_state.imu_state.quaternion[3] = float(self.mj_data.sensordata[off + 3])
            self.low_state.imu_state.gyroscope[0] = float(self.mj_data.sensordata[off + 4])
            self.low_state.imu_state.gyroscope[1] = float(self.mj_data.sensordata[off + 5])
            self.low_state.imu_state.gyroscope[2] = float(self.mj_data.sensordata[off + 6])
            self.low_state.imu_state.accelerometer[0] = float(self.mj_data.sensordata[off + 7])
            self.low_state.imu_state.accelerometer[1] = float(self.mj_data.sensordata[off + 8])
            self.low_state.imu_state.accelerometer[2] = float(self.mj_data.sensordata[off + 9])

        # Do not call pygame.event.get() here: on macOS it must run on the main thread only.
        # The main thread (unitree_mujoco) pumps events; we only read joystick state.
        if self.joystick is not None:
            axis_id = getattr(self, "axis_id", {})
            button_id = getattr(self, "button_id", {})
            if axis_id and button_id:
                self.low_state.wireless_remote[2] = int(
                    "".join([
                        str(int(self.joystick.get_axis(axis_id.get("LT", 2)) > 0)),
                        str(int(self.joystick.get_axis(axis_id.get("RT", 5)) > 0)),
                        str(int(self.joystick.get_button(button_id.get("SELECT", 6)))),
                        str(int(self.joystick.get_button(button_id.get("START", 7)))),
                        str(int(self.joystick.get_button(button_id.get("LB", 4)))),
                        str(int(self.joystick.get_button(button_id.get("RB", 5)))),
                    ]), 2
                )
                self.low_state.wireless_remote[3] = int(
                    "".join([
                        str(int(self.joystick.get_hat(0)[0] < 0)),
                        str(int(self.joystick.get_hat(0)[1] < 0)),
                        str(int(self.joystick.get_hat(0)[0] > 0)),
                        str(int(self.joystick.get_hat(0)[1] > 0)),
                        str(int(self.joystick.get_button(button_id.get("Y", 3)))),
                        str(int(self.joystick.get_button(button_id.get("X", 2)))),
                        str(int(self.joystick.get_button(button_id.get("B", 1)))),
                        str(int(self.joystick.get_button(button_id.get("A", 0)))),
                    ]), 2
                )
                sticks = [
                    self.joystick.get_axis(axis_id.get("LX", 0)),
                    self.joystick.get_axis(axis_id.get("RX", 3)),
                    -self.joystick.get_axis(axis_id.get("RY", 4)),
                    -self.joystick.get_axis(axis_id.get("LY", 1)),
                ]
                packs = [struct.pack("f", x) for x in sticks]
                self.low_state.wireless_remote[4:8] = packs[0]
                self.low_state.wireless_remote[8:12] = packs[1]
                self.low_state.wireless_remote[12:16] = packs[2]
                self.low_state.wireless_remote[20:24] = packs[3]

        self.low_state_puber.Write(self.low_state)

    def PublishHighState(self) -> None:
        if self.mj_data is None:
            return
        off = self.dim_motor_sensor
        self.high_state.position[0] = float(self.mj_data.sensordata[off + 10])
        self.high_state.position[1] = float(self.mj_data.sensordata[off + 11])
        self.high_state.position[2] = float(self.mj_data.sensordata[off + 12])
        self.high_state.velocity[0] = float(self.mj_data.sensordata[off + 13])
        self.high_state.velocity[1] = float(self.mj_data.sensordata[off + 14])
        self.high_state.velocity[2] = float(self.mj_data.sensordata[off + 15])
        self.high_state_puber.Write(self.high_state)

    def set_external_wireless_controller(self, w: WirelessController_) -> None:
        """Set wireless controller state from outside (e.g. main thread on macOS). Thread-safe."""
        with self._external_wireless_lock:
            self._external_wireless_controller = w

    def PublishWirelessController(self) -> None:
        with self._external_wireless_lock:
            external = self._external_wireless_controller
        if external is not None:
            self.wireless_controller.keys = external.keys
            self.wireless_controller.lx = external.lx
            self.wireless_controller.ly = external.ly
            self.wireless_controller.rx = external.rx
            self.wireless_controller.ry = external.ry
            self.wireless_controller_puber.Write(self.wireless_controller)
            return
        if self.joystick is None:
            # macOS: external not set yet; publish default so play.py receives (rcvd=yes).
            self.wireless_controller_puber.Write(self.wireless_controller)
            return
        # Do not call pygame.event.get() here: on macOS it must run on the main thread only.
        key_map = self.key_map
        axis_id = getattr(self, "axis_id", {})
        button_id = getattr(self, "button_id", {})
        if not axis_id or not button_id:
            return
        n_buttons = self.joystick.get_numbuttons()
        n_axes = self.joystick.get_numaxes()
        r1 = self.joystick.get_button(button_id["RB"]) if n_buttons > button_id["RB"] else 0
        if button_id.get("RB_alt") is not None and n_buttons > button_id["RB_alt"]:
            r1 = r1 or self.joystick.get_button(button_id["RB_alt"])
        rt_axis = self.joystick.get_axis(axis_id["RT"]) if n_axes > axis_id["RT"] else -1.0
        rt_btn = button_id.get("RT")
        r2 = 1 if rt_axis > 0.3 else 0
        if rt_btn is not None and n_buttons > rt_btn:
            r2 = r2 or self.joystick.get_button(rt_btn)
        key_state = [0] * 16
        key_state[key_map["R1"]] = r1
        key_state[key_map["L1"]] = self.joystick.get_button(button_id["LB"]) if n_buttons > button_id["LB"] else 0
        key_state[key_map["start"]] = self.joystick.get_button(button_id["START"]) if n_buttons > button_id["START"] else 0
        key_state[key_map["select"]] = self.joystick.get_button(button_id["SELECT"]) if n_buttons > button_id["SELECT"] else 0
        key_state[key_map["R2"]] = r2
        key_state[key_map["L2"]] = 1 if (n_axes > axis_id["LT"] and self.joystick.get_axis(axis_id["LT"]) > 0.3) else 0
        key_state[key_map["A"]] = self.joystick.get_button(button_id["A"]) if n_buttons > button_id["A"] else 0
        key_state[key_map["B"]] = self.joystick.get_button(button_id["B"])
        key_state[key_map["X"]] = self.joystick.get_button(button_id["X"])
        key_state[key_map["Y"]] = self.joystick.get_button(button_id["Y"])
        key_state[key_map["up"]] = 1 if self.joystick.get_hat(0)[1] > 0 else 0
        key_state[key_map["right"]] = 1 if self.joystick.get_hat(0)[0] > 0 else 0
        key_state[key_map["down"]] = 1 if self.joystick.get_hat(0)[1] < 0 else 0
        key_state[key_map["left"]] = 1 if self.joystick.get_hat(0)[0] < 0 else 0
        key_value = sum(b << i for i, b in enumerate(key_state))
        self.wireless_controller.keys = key_value
        self.wireless_controller.lx = self.joystick.get_axis(axis_id["LX"])
        self.wireless_controller.ly = -self.joystick.get_axis(axis_id["LY"])
        self.wireless_controller.rx = self.joystick.get_axis(axis_id["RX"])
        self.wireless_controller.ry = -self.joystick.get_axis(axis_id["RY"])
        self.wireless_controller_puber.Write(self.wireless_controller)

    def SetJoystickMapping(self, js_type: str = "xbox") -> None:
        """Set axis_id and button_id only (for external joystick reader, e.g. main thread on macOS)."""
        if js_type == "xbox":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 3, "RY": 4, "LT": 2, "RT": 5, "DX": 6, "DY": 7,
            }
            # RB: some controllers use 5, others 10 (e.g. Xbox on macOS). RT: axis 5 or button 7.
            self.button_id = {
                "X": 2, "Y": 3, "B": 1, "A": 0, "LB": 4, "RB": 5, "RB_alt": 10,
                "SELECT": 6, "START": 7, "RT": 7,
            }
        elif js_type == "switch":
            self.axis_id = {
                "LX": 0, "LY": 1, "RX": 2, "RY": 3, "LT": 5, "RT": 4, "DX": 6, "DY": 7,
            }
            self.button_id = {
                "X": 3, "Y": 4, "B": 1, "A": 0, "LB": 6, "RB": 7, "SELECT": 10, "START": 11,
            }
        else:
            raise ValueError(f"Unsupported gamepad type: {js_type}")

    def build_wireless_controller_from_joystick(self, joystick: Any) -> WirelessController_:
        """Build WirelessController_ from a pygame Joystick. Call only from main thread (e.g. macOS)."""
        key_map = self.key_map
        axis_id = self.axis_id
        button_id = self.button_id
        if not axis_id or not button_id:
            return self.wireless_controller
        n_buttons = joystick.get_numbuttons()
        n_axes = joystick.get_numaxes()
        r1 = joystick.get_button(button_id["RB"]) if n_buttons > button_id["RB"] else 0
        if button_id.get("RB_alt") is not None and n_buttons > button_id["RB_alt"]:
            r1 = r1 or joystick.get_button(button_id["RB_alt"])
        key_state = [0] * 16
        key_state[key_map["R1"]] = r1
        key_state[key_map["L1"]] = joystick.get_button(button_id["LB"]) if n_buttons > button_id["LB"] else 0
        key_state[key_map["start"]] = joystick.get_button(button_id["START"]) if n_buttons > button_id["START"] else 0
        key_state[key_map["select"]] = joystick.get_button(button_id["SELECT"]) if n_buttons > button_id["SELECT"] else 0
        rt_axis = joystick.get_axis(axis_id["RT"]) if n_axes > axis_id["RT"] else -1.0
        rt_btn = button_id.get("RT")
        r2 = 1 if rt_axis > 0.3 else 0
        if rt_btn is not None and n_buttons > rt_btn:
            r2 = r2 or joystick.get_button(rt_btn)
        key_state[key_map["R2"]] = r2
        key_state[key_map["L2"]] = 1 if (n_axes > axis_id["LT"] and joystick.get_axis(axis_id["LT"]) > 0.3) else 0
        key_state[key_map["A"]] = joystick.get_button(button_id["A"])
        key_state[key_map["B"]] = joystick.get_button(button_id["B"])
        key_state[key_map["X"]] = joystick.get_button(button_id["X"])
        key_state[key_map["Y"]] = joystick.get_button(button_id["Y"])
        key_state[key_map["up"]] = 1 if (joystick.get_numhats() > 0 and joystick.get_hat(0)[1] > 0) else 0
        key_state[key_map["right"]] = 1 if (joystick.get_numhats() > 0 and joystick.get_hat(0)[0] > 0) else 0
        key_state[key_map["down"]] = 1 if (joystick.get_numhats() > 0 and joystick.get_hat(0)[1] < 0) else 0
        key_state[key_map["left"]] = 1 if (joystick.get_numhats() > 0 and joystick.get_hat(0)[0] < 0) else 0
        w = WirelessController_()
        w.keys = sum(b << i for i, b in enumerate(key_state))
        w.lx = joystick.get_axis(axis_id["LX"])
        w.ly = -joystick.get_axis(axis_id["LY"])
        w.rx = joystick.get_axis(axis_id["RX"])
        w.ry = -joystick.get_axis(axis_id["RY"])
        return w

    def SetupJoystick(self, device_id: int = 0, js_type: str = "xbox") -> None:
        import pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("No gamepad detected.")
            sys.exit(1)
        self.joystick = pygame.joystick.Joystick(device_id)
        self.joystick.init()
        self.SetJoystickMapping(js_type)

    def PrintSceneInformation(self) -> None:
        mujoco = self._mujoco
        print("\n<<------------- Link ------------->>")
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                print("link_index:", i, ", name:", name)
        print("\n<<------------- Joint ------------->>")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                print("joint_index:", i, ", name:", name)
        print("\n<<------------- Actuator ------------->>")
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                print("actuator_index:", i, ", name:", name)
        print("\n<<------------- Sensor ------------->>")
        idx = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name:
                print("sensor_index:", idx, ", name:", name, ", dim:", self.mj_model.sensor_dim[i])
            idx += self.mj_model.sensor_dim[i]
        print()


class ElasticBand:
    def __init__(self):
        import numpy as np
        self._np = np
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0.0, 0.0, 3.0])
        self.length = 0.0
        self.enable = True

    def Advance(self, x, dx):
        np = self._np
        dx_vec = self.point - x
        distance = np.linalg.norm(dx_vec)
        if distance < 1e-8:
            return np.zeros(3)
        direction = dx_vec / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key: int) -> None:
        try:
            import mujoco.glfw
            glfw = mujoco.glfw.glfw
            if key == glfw.KEY_7:
                self.length -= 0.1
            if key == glfw.KEY_8:
                self.length += 0.1
            if key == glfw.KEY_9:
                self.enable = not self.enable
        except Exception:
            pass


# LCM receive thread: LCM.handle() is blocking, so run it in a background thread after subscribe.
# ChannelSubscriber.Init starts the lcm.handle() loop in a background thread.
def _lcm_handle_loop() -> None:
    lc = _lcm_instance
    if lc is None:
        return
    while True:
        try:
            lc.handle()
        except Exception as e:
            print(f"[LCM] handle error: {e}", file=sys.stderr)
        time.sleep(0.001)


def _start_lcm_handle_thread() -> None:
    t = threading.Thread(target=_lcm_handle_loop, daemon=True)
    t.start()


# Start handle thread when a subscriber exists; start once on first subscriber Init.
_lcm_handle_started = False


def _ensure_lcm_handle_thread() -> None:
    """Start a background thread that runs LCM.handle(). Skip if LCM_NO_HANDLE_THREAD=1 (caller will use process_lcm_messages() in main loop)."""
    global _lcm_handle_started
    if _lcm_handle_started:
        return
    if os.environ.get("LCM_NO_HANDLE_THREAD", "").strip() == "1":
        return
    _lcm_handle_started = True
    _start_lcm_handle_thread()


# ChannelSubscriber.Init calls _ensure_lcm_handle_thread()
_original_subscriber_init = ChannelSubscriber.Init


def _subscriber_init_with_handle(self, handler, priority=10):
    _ensure_lcm_handle_thread()
    _original_subscriber_init(self, handler, priority)


ChannelSubscriber.Init = _subscriber_init_with_handle
