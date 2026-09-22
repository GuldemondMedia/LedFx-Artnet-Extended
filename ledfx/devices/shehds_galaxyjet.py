import colorsys
import logging
import math
import threading
import time

import numpy as np
import voluptuous as vol
from stupidArtnet import StupidArtnet

from ledfx.devices import NetworkedDevice
from ledfx.utils import check_if_ip_is_broadcast

_LOGGER = logging.getLogger(__name__)

CHANNEL_COUNT = 23

MOVEMENT_MODES = (
    "Static",
    "Sweep",
    "Mirror sweep",
    "Circle",
    "Infinity (figure 8)",
    "BPM synced sweep",
    "Bar synced circle",
)

PRISM_MODES = (
    "Off",
    "On",
    "Toggle on beat",
)

BRIGHTNESS_MODES = (
    "Follow LEDFx effect",
    "Music",
    "Effect x music",
    "Fixed",
)

COLOR_MODES = (
    "Follow LEDFx effect",
    "Fixed color",
    "Change on beat",
    "Rainbow wheel",
)

GOBO_MODES = (
    "Open",
    "Fixed gobo",
    "Change on beat",
)

STROBE_MODES = (
    "Off",
    "Beat pulse",
    "Music reactive",
)

COLOR_SLOTS = {
    "Open": 2,
    "Open / Red": 7,
    "Red": 12,
    "Red / Yellow": 17,
    "Yellow": 22,
    "Yellow / Dark blue": 27,
    "Dark blue": 32,
    "Dark blue / Green 1": 37,
    "Green 1": 42,
    "Green 1 / Orange": 47,
    "Orange": 52,
    "Orange / Purple": 57,
    "Purple": 62,
    "Purple / Green 2": 67,
    "Green 2": 72,
    "Green 2 / Light blue": 77,
    "Light blue": 82,
    "Light blue / Pink": 87,
    "Pink": 92,
    "Pink / Open": 97,
}

SOLID_COLOR_SEQUENCE = (
    2,
    12,
    22,
    32,
    42,
    52,
    62,
    72,
    82,
    92,
)

# Hue anchors used when an LEDFx RGB effect is mapped onto the physical
# GalaxyJet color wheel. The values are the centres of the documented
# color-wheel ranges instead of arbitrary 0-255 positions.
HUE_COLOR_SLOTS = (
    (0.000, 12),  # red
    (0.083, 52),  # orange
    (0.167, 22),  # yellow
    (0.320, 42),  # green 1
    (0.410, 72),  # green 2
    (0.530, 82),  # light blue
    (0.667, 32),  # dark blue
    (0.780, 62),  # purple
    (0.920, 92),  # pink
)

GOBO_SLOTS = {
    "Open": 5,
    "Gobo 1": 15,
    "Gobo 2": 25,
    "Gobo 3": 35,
    "Gobo 4": 45,
    "Gobo 5": 55,
    "Gobo 6": 65,
}

GOBO_SEQUENCE = tuple(GOBO_SLOTS.values())

# CH6 shutter/strobe documented ranges.
SHUTTER_OPEN = 105
SHUTTER_STROBE_MIN = 4
SHUTTER_STROBE_MAX = 103


class ShehdsGalaxyJetDevice(NetworkedDevice):
    """SHEHDS GalaxyJet 300 3-in-1 moving head, 23CH mode."""

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Required(
                "pixel_count",
                description=(
                    "Number of GalaxyJet fixtures. One LEDFx pixel controls "
                    "one complete moving head."
                ),
                default=4,
            ): vol.All(int, vol.Range(min=1, max=22)),
            vol.Optional(
                "universe",
                description="Art-Net universe",
                default=0,
            ): vol.All(int, vol.Range(min=0)),
            vol.Optional(
                "dmx_start_address",
                description=(
                    "DMX start address of fixture 1 when using regular "
                    "fixture spacing"
                ),
                default=1,
            ): vol.All(int, vol.Range(min=1, max=490)),
            vol.Optional(
                "fixture_spacing",
                description=(
                    "Channels between fixture start addresses. Use 23 for "
                    "fixtures directly after each other."
                ),
                default=23,
            ): vol.All(int, vol.Range(min=23, max=512)),
            vol.Optional(
                "fixture_addresses",
                description=(
                    "Optional exact start addresses, e.g. 1,30,60,100. "
                    "When filled in, this overrides start address + spacing."
                ),
                default="",
            ): str,
            vol.Optional(
                "music_sensitivity",
                description=(
                    "Overall sensitivity of movement/brightness/strobe to "
                    "LEDFx's existing music analysis"
                ),
                default=1.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.25, max=2.0)),
            vol.Optional(
                "movement_mode",
                description="Movement pattern",
                default="Sweep",
            ): vol.All(str, vol.In(MOVEMENT_MODES)),
            vol.Optional(
                "movement_speed",
                description=(
                    "Base movement speed: 0 is very slow, 100 is fast"
                ),
                default=45,
            ): vol.All(int, vol.Range(min=0, max=100)),
            vol.Optional(
                "movement_music_reactivity",
                description=(
                    "How much louder lows/bass make movement faster"
                ),
                default=60,
            ): vol.All(int, vol.Range(min=0, max=100)),
            vol.Optional(
                "movement_smoothness",
                description=(
                    "Mechanical smoothing: higher values make pan/tilt "
                    "targets calmer and easier for the motors to follow"
                ),
                default=60,
            ): vol.All(int, vol.Range(min=0, max=100)),
            vol.Optional(
                "movement_spread",
                description=(
                    "Phase spread across fixtures: 0 moves together, "
                    "100 spreads all fixtures across the pattern"
                ),
                default=100,
            ): vol.All(int, vol.Range(min=0, max=100)),
            vol.Optional(
                "pan_center",
                description="Pan centre position",
                default=127,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "tilt_center",
                description="Tilt centre position",
                default=127,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "pan_range",
                description="Maximum pan movement around the centre",
                default=160,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "tilt_range",
                description="Maximum tilt movement around the centre",
                default=96,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "pan_invert",
                description="Invert pan direction",
                default=False,
            ): bool,
            vol.Optional(
                "tilt_invert",
                description="Invert tilt direction",
                default=False,
            ): bool,
            vol.Optional(
                "brightness_mode",
                description="How the beam brightness is controlled",
                default="Follow LEDFx effect",
            ): vol.All(str, vol.In(BRIGHTNESS_MODES)),
            vol.Optional(
                "master_brightness",
                description="Maximum brightness / fixed brightness",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "color_mode",
                description="Physical color-wheel behaviour",
                default="Follow LEDFx effect",
            ): vol.All(str, vol.In(COLOR_MODES)),
            vol.Optional(
                "fixed_color",
                description="Physical color used in Fixed color mode",
                default="Open",
            ): vol.All(str, vol.In(COLOR_SLOTS.keys())),
            vol.Optional(
                "color_change_interval",
                description=(
                    "Minimum seconds between physical color-wheel changes"
                ),
                default=0.60,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=5.0)),
            vol.Optional(
                "rainbow_speed",
                description=(
                    "Built-in rainbow-wheel speed: 0 slow, 100 fast"
                ),
                default=50,
            ): vol.All(int, vol.Range(min=0, max=100)),
            vol.Optional(
                "gobo_mode",
                description="Gobo wheel behaviour",
                default="Open",
            ): vol.All(str, vol.In(GOBO_MODES)),
            vol.Optional(
                "fixed_gobo",
                description="Gobo used in Fixed gobo mode",
                default="Open",
            ): vol.All(str, vol.In(GOBO_SLOTS.keys())),
            vol.Optional(
                "gobo_change_interval",
                description=(
                    "Minimum seconds between physical gobo-wheel changes"
                ),
                default=0.80,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=5.0)),
            vol.Optional(
                "strobe_mode",
                description="Shutter/strobe behaviour",
                default="Off",
            ): vol.All(str, vol.In(STROBE_MODES)),
            vol.Optional(
                "strobe_strength",
                description=(
                    "Maximum strobe rate for beat/music strobe modes"
                ),
                default=50,
            ): vol.All(int, vol.Range(min=0, max=100)),
            vol.Optional(
                "prism_mode",
                description=(
                    "Prism behaviour. Toggle on beat alternates the prism "
                    "in and out using LEDFx beat detection."
                ),
                default="Off",
            ): vol.All(str, vol.In(PRISM_MODES)),
            vol.Optional(
                "prism_rotation_speed",
                description=(
                    "Prism rotation speed: 0 is stopped, 100 is fastest"
                ),
                default=0,
            ): vol.All(int, vol.Range(min=0, max=100)),
            vol.Optional(
                "zoom",
                description="Fixed zoom position",
                default=128,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "focus",
                description="Fixed focus position",
                default=128,
            ): vol.All(int, vol.Range(min=0, max=255)),
        }
    )

    def __init__(self, ledfx, config):
        super().__init__(ledfx, config)
        self._device_type = "SHEHDS GalaxyJet 300"
        self._artnet = None
        self._fixture_audio = None
        self._audio_lock = threading.Lock()
        self._audio = {
            "lows": 0.0,
            "highs": 0.0,
            "beat_oscillator": 0.0,
            "bar_oscillator": 0.0,
            "beat_time": 0.0,
            "beat_count": 0,
        }
        self._epoch = time.monotonic()
        self._last_frame_time = self._epoch
        self._movement_phase = 0.0
        self._smoothed_speed = None
        self._position_state = {}
        self._held_values = {}
        self._fixture_addresses = self._calculate_addresses()

    def config_updated(self, config):
        self._fixture_addresses = self._calculate_addresses()
        self._position_state.clear()
        self._held_values.clear()
        self._smoothed_speed = None
        self._last_frame_time = time.monotonic()
        was_active = self._active
        if was_active:
            self.deactivate()
            self.activate()

    def _calculate_addresses(self):
        exact = str(self._config.get("fixture_addresses", "")).strip()
        if exact:
            addresses = []
            for part in exact.replace(";", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    value = int(part)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid fixture address: {part}"
                    ) from exc
                addresses.append(value)

            if len(addresses) != self.pixel_count:
                raise ValueError(
                    "fixture_addresses must contain exactly one start "
                    "address per fixture"
                )
        else:
            first = int(self._config.get("dmx_start_address", 1))
            spacing = int(self._config.get("fixture_spacing", 23))
            addresses = [
                first + index * spacing
                for index in range(self.pixel_count)
            ]

        for address in addresses:
            if address < 1 or address + CHANNEL_COUNT - 1 > 512:
                raise ValueError(
                    f"Fixture at DMX {address} does not fit in one universe"
                )

        ordered = sorted(addresses)
        for previous, current in zip(ordered, ordered[1:]):
            if current < previous + CHANNEL_COUNT:
                raise ValueError(
                    "GalaxyJet fixture DMX ranges overlap. Use at least "
                    "23 channels spacing."
                )

        return tuple(address - 1 for address in addresses)

    def _subscribe_audio(self):
        if self._fixture_audio is not None:
            return
        try:
            from ledfx.effects.audio import AudioAnalysisSource

            if (
                not self._ledfx.audio
                or not isinstance(self._ledfx.audio, AudioAnalysisSource)
            ):
                self._ledfx.audio = AudioAnalysisSource(
                    self._ledfx,
                    self._ledfx.config.get("audio", {}),
                )

            self._fixture_audio = self._ledfx.audio
            self._fixture_audio.subscribe(self._audio_updated)
        except Exception:
            self._fixture_audio = None
            _LOGGER.exception(
                "Unable to subscribe %s to LEDFx audio analysis", self.name
            )

    def _unsubscribe_audio(self):
        if self._fixture_audio is None:
            return
        try:
            self._fixture_audio.unsubscribe(self._audio_updated)
        except Exception:
            _LOGGER.exception(
                "Unable to unsubscribe %s from LEDFx audio", self.name
            )
        self._fixture_audio = None

    def _audio_updated(self):
        audio = self._fixture_audio
        if audio is None:
            return

        now = time.monotonic()
        try:
            lows = float(audio.lows_power())
            highs = float(audio.high_power())
            beat_oscillator = float(audio.beat_oscillator())
            bar_oscillator = float(audio.bar_oscillator()) / 4.0
            beat_now = bool(audio.volume_beat_now())
        except Exception:
            _LOGGER.debug(
                "LEDFx audio analysis unavailable for GalaxyJet",
                exc_info=True,
            )
            return

        with self._audio_lock:
            self._audio["lows"] = max(0.0, min(1.0, lows))
            self._audio["highs"] = max(0.0, min(1.0, highs))
            self._audio["beat_oscillator"] = max(
                0.0, min(1.0, beat_oscillator)
            )
            self._audio["bar_oscillator"] = max(
                0.0, min(1.0, bar_oscillator)
            )
            if beat_now:
                self._audio["beat_time"] = now
                self._audio["beat_count"] += 1

    def _audio_snapshot(self):
        with self._audio_lock:
            return dict(self._audio)

    def _music_level(self, value):
        sensitivity = float(self._config.get("music_sensitivity", 1.0))
        return max(0.0, min(1.0, float(value) * sensitivity))

    @staticmethod
    def _coarse_fine(value):
        value = min(255.0, max(0.0, float(value)))
        full = int(round(value * 257.0))
        return (full >> 8) & 0xFF, full & 0xFF

    @staticmethod
    def _nearest_effect_color(hue, saturation):
        if saturation < 0.08:
            return COLOR_SLOTS["Open"]

        best_value = HUE_COLOR_SLOTS[0][1]
        best_distance = 1.0
        for anchor, value in HUE_COLOR_SLOTS:
            distance = abs(hue - anchor)
            distance = min(distance, 1.0 - distance)
            if distance < best_distance:
                best_distance = distance
                best_value = value
        return best_value

    def _hold_value(self, key, desired, interval, now):
        desired = int(desired)
        state = self._held_values.get(key)
        if state is None:
            self._held_values[key] = {
                "value": desired,
                "changed": now,
            }
            return desired

        if (
            desired != state["value"]
            and now - state["changed"] >= interval
        ):
            state["value"] = desired
            state["changed"] = now

        return int(state["value"])

    def _smooth_position(self, key, target, now):
        smoothness = int(self._config.get("movement_smoothness", 60))
        if smoothness <= 0:
            self._position_state[key] = (float(target), now)
            return float(target)

        tau = 0.05 + (smoothness / 100.0) * 0.85
        state = self._position_state.get(key)
        if state is None:
            self._position_state[key] = (float(target), now)
            return float(target)

        value, last_time = state
        delta = max(0.0, min(0.25, now - last_time))
        alpha = 1.0 - math.exp(-delta / tau)
        value += (float(target) - value) * alpha
        self._position_state[key] = (value, now)
        return value

    def _update_movement_phase(self, audio, now):
        delta = max(0.0, min(0.25, now - self._last_frame_time))
        self._last_frame_time = now

        base_speed = int(self._config.get("movement_speed", 45)) / 100.0
        reactivity = (
            int(self._config.get("movement_music_reactivity", 60)) / 100.0
        )
        lows = self._music_level(audio.get("lows", 0.0))

        target_speed = base_speed + (
            1.0 - base_speed
        ) * lows * reactivity

        if self._smoothed_speed is None:
            self._smoothed_speed = target_speed
        else:
            smoothness = int(
                self._config.get("movement_smoothness", 60)
            )
            tau = 0.05 + (smoothness / 100.0) * 0.55
            alpha = 1.0 - math.exp(-delta / tau) if delta else 0.0
            self._smoothed_speed += (
                target_speed - self._smoothed_speed
            ) * alpha

        # 0 -> 12 seconds per cycle, 1 -> 0.8 seconds per cycle.
        period = 12.0 - self._smoothed_speed * 11.2
        period = max(0.8, period)
        self._movement_phase = (
            self._movement_phase + delta * math.tau / period
        ) % math.tau

    def _movement_position(self, fixture_index, audio, now):
        mode = self._config.get("movement_mode", "Sweep")
        pan_center = float(self._config.get("pan_center", 127))
        tilt_center = float(self._config.get("tilt_center", 127))
        pan_range = float(self._config.get("pan_range", 160))
        tilt_range = float(self._config.get("tilt_range", 96))

        spread = int(self._config.get("movement_spread", 100)) / 100.0
        fixture_phase = (
            fixture_index / max(1, self.pixel_count)
        ) * math.tau * spread

        phase = self._movement_phase + fixture_phase
        if mode == "BPM synced sweep":
            phase = (
                audio.get("beat_oscillator", 0.0) * math.tau
                + fixture_phase
            )
        elif mode == "Bar synced circle":
            phase = (
                audio.get("bar_oscillator", 0.0) * math.tau
                + fixture_phase
            )

        pan = pan_center
        tilt = tilt_center

        if mode in ("Sweep", "BPM synced sweep"):
            pan += math.sin(phase) * pan_range / 2.0
            tilt += math.sin(phase * 0.5) * tilt_range / 2.0
        elif mode == "Mirror sweep":
            direction = -1.0 if fixture_index % 2 else 1.0
            pan += direction * math.sin(phase) * pan_range / 2.0
            tilt += math.cos(phase) * tilt_range / 2.0
        elif mode in ("Circle", "Bar synced circle"):
            pan += math.sin(phase) * pan_range / 2.0
            tilt += math.cos(phase) * tilt_range / 2.0
        elif mode == "Infinity (figure 8)":
            # Lissajous 1:2 path: a horizontal figure-eight. This sweeps
            # both sides of a court instead of orbiting one centre point.
            pan += math.sin(phase) * pan_range / 2.0
            tilt += math.sin(phase * 2.0) * tilt_range / 2.0

        pan = min(255.0, max(0.0, pan))
        tilt = min(255.0, max(0.0, tilt))

        pan = self._smooth_position(
            ("pan", fixture_index), pan, now
        )
        tilt = self._smooth_position(
            ("tilt", fixture_index), tilt, now
        )

        if self._config.get("pan_invert", False):
            pan = 255.0 - pan
        if self._config.get("tilt_invert", False):
            tilt = 255.0 - tilt

        return pan, tilt

    def _dimmer(self, effect_brightness, audio):
        effect_level = max(
            0.0, min(1.0, effect_brightness / 255.0)
        )
        music_level = self._music_level(audio.get("lows", 0.0))
        mode = self._config.get(
            "brightness_mode", "Follow LEDFx effect"
        )

        if mode == "Music":
            level = music_level
        elif mode == "Effect x music":
            level = effect_level * music_level
        elif mode == "Fixed":
            level = 1.0
        else:
            level = effect_level

        maximum = float(self._config.get("master_brightness", 255))
        return level * maximum

    def _color(self, fixture_index, hue, saturation, audio, now):
        mode = self._config.get("color_mode", "Follow LEDFx effect")

        if mode == "Fixed color":
            desired = COLOR_SLOTS[
                self._config.get("fixed_color", "Open")
            ]
        elif mode == "Change on beat":
            desired = SOLID_COLOR_SEQUENCE[
                audio.get("beat_count", 0) % len(SOLID_COLOR_SEQUENCE)
            ]
        elif mode == "Rainbow wheel":
            speed = int(self._config.get("rainbow_speed", 50)) / 100.0
            # 100-180 = rainbow fast -> slow.
            return int(round(180 - speed * 80))
        else:
            desired = self._nearest_effect_color(hue, saturation)

        interval = float(
            self._config.get("color_change_interval", 0.60)
        )
        return self._hold_value(
            ("color", fixture_index),
            desired,
            interval,
            now,
        )

    def _gobo(self, fixture_index, audio, now):
        mode = self._config.get("gobo_mode", "Open")

        if mode == "Fixed gobo":
            desired = GOBO_SLOTS[
                self._config.get("fixed_gobo", "Open")
            ]
        elif mode == "Change on beat":
            desired = GOBO_SEQUENCE[
                audio.get("beat_count", 0) % len(GOBO_SEQUENCE)
            ]
        else:
            desired = GOBO_SLOTS["Open"]

        interval = float(
            self._config.get("gobo_change_interval", 0.80)
        )
        return self._hold_value(
            ("gobo", fixture_index),
            desired,
            interval,
            now,
        )

    def _shutter(self, audio, now):
        mode = self._config.get("strobe_mode", "Off")
        if mode == "Off":
            return SHUTTER_OPEN

        strength = int(self._config.get("strobe_strength", 50)) / 100.0
        max_strobe = int(
            round(
                SHUTTER_STROBE_MIN
                + strength
                * (SHUTTER_STROBE_MAX - SHUTTER_STROBE_MIN)
            )
        )

        if mode == "Beat pulse":
            if now - audio.get("beat_time", 0.0) <= 0.12:
                return max_strobe
            return SHUTTER_OPEN

        highs = self._music_level(audio.get("highs", 0.0))
        if highs < 0.45:
            return SHUTTER_OPEN
        level = (highs - 0.45) / 0.55
        return int(
            round(
                SHUTTER_STROBE_MIN
                + level * (max_strobe - SHUTTER_STROBE_MIN)
            )
        )

    def _prism(self, audio):
        mode = self._config.get("prism_mode", "Off")
        if mode == "On":
            prism = 128
        elif mode == "Toggle on beat":
            prism = 128 if audio.get("beat_count", 0) % 2 else 0
        else:
            prism = 0

        speed = int(self._config.get("prism_rotation_speed", 0))
        if prism == 0 or speed <= 0:
            rotation = 191
        else:
            # Documented clockwise rotation range is 193-255,
            # slow to fast. 191-192 is stop.
            rotation = int(round(193 + (speed / 100.0) * 62))

        return prism, rotation

    def activate(self):
        if self._destination is None:
            super().activate()
            return

        if self._artnet is not None:
            self._artnet.close()

        broadcast = check_if_ip_is_broadcast(self._config["ip_address"])
        self._artnet = StupidArtnet(
            target_ip=self.destination,
            universe=self._config["universe"],
            packet_size=512,
            fps=self._config["refresh_rate"],
            even_packet_size=True,
            broadcast=broadcast,
            port=6454,
        )

        super().activate()
        self._subscribe_audio()
        self._last_frame_time = time.monotonic()

    def deactivate(self):
        self._unsubscribe_audio()
        if self._artnet is not None:
            try:
                self._artnet.blackout()
                self._artnet.close()
            finally:
                self._artnet = None
        super().deactivate()

    def flush(self, data):
        if self._artnet is None:
            return

        if not self.lock.acquire(blocking=False):
            return

        try:
            rgb = np.asarray(data)
            if rgb.ndim == 1:
                rgb = rgb.reshape((-1, 3))
            else:
                rgb = rgb.reshape((-1, rgb.shape[-1]))

            rgb = np.clip(rgb[:, :3], 0, 255)
            if rgb.shape[0] < self.pixel_count:
                padded = np.zeros((self.pixel_count, 3), dtype=float)
                padded[: rgb.shape[0]] = rgb
                rgb = padded
            else:
                rgb = rgb[: self.pixel_count]

            packet = np.zeros(512, dtype=np.uint8)
            audio = self._audio_snapshot()
            now = time.monotonic()
            self._update_movement_phase(audio, now)

            for fixture_index, start in enumerate(
                self._fixture_addresses
            ):
                red, green, blue = rgb[fixture_index]
                brightness = float(max(red, green, blue))

                if brightness <= 0:
                    hue = 0.0
                    saturation = 0.0
                else:
                    hue, saturation, _ = colorsys.rgb_to_hsv(
                        red / 255.0,
                        green / 255.0,
                        blue / 255.0,
                    )

                pan, tilt = self._movement_position(
                    fixture_index, audio, now
                )
                pan_coarse, pan_fine = self._coarse_fine(pan)
                tilt_coarse, tilt_fine = self._coarse_fine(tilt)

                dimmer = self._dimmer(brightness, audio)
                dimmer_coarse, dimmer_fine = self._coarse_fine(
                    dimmer
                )

                color = self._color(
                    fixture_index,
                    hue,
                    saturation,
                    audio,
                    now,
                )
                gobo = self._gobo(fixture_index, audio, now)
                shutter = self._shutter(audio, now)
                prism, prism_rotation = self._prism(audio)

                focus_coarse, focus_fine = self._coarse_fine(
                    self._config.get("focus", 128)
                )

                fixture = np.array(
                    [
                        pan_coarse,                    # CH1  Pan
                        pan_fine,                      # CH2  Pan fine
                        tilt_coarse,                   # CH3  Tilt
                        tilt_fine,                     # CH4  Tilt fine
                        128,                           # CH5  P/T speed
                        shutter,                       # CH6  Shutter/strobe
                        dimmer_coarse,                 # CH7  Dimmer
                        dimmer_fine,                   # CH8  Dimmer fine
                        color,                         # CH9  Color wheel
                        0,                             # CH10 Color fine
                        gobo,                          # CH11 Gobo 1
                        0,                             # CH12 Gobo 2 open
                        0,                             # CH13 Gobo 2 rotation
                        prism,                         # CH14 Prism
                        prism_rotation,                # CH15 Prism rotation
                        0,                             # CH16 Frost off
                        self._config.get("zoom", 128), # CH17 Zoom
                        focus_coarse,                  # CH18 Focus
                        focus_fine,                    # CH19 Focus fine
                        0,                             # CH20 Never reset
                        0,                             # CH21 Ring strobe off
                        0,                             # CH22 Ring mode off
                        0,                             # CH23 Ring speed off
                    ],
                    dtype=np.uint8,
                )

                packet[start : start + CHANNEL_COUNT] = fixture

            self._artnet.set(packet)
            self._artnet.show()
        finally:
            self.lock.release()
