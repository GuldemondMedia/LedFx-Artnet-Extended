import colorsys
import logging
import math
import threading
import time

import numpy as np
import voluptuous as vol
from stupidArtnet import StupidArtnet

from ledfx.devices import NetworkedDevice
from ledfx.devices.utils.rgbw_conversion import (
    RGB_MAPPING,
    WHITE_FUNCS_MAPPING,
    OutputMode,
)
from ledfx.utils import check_if_ip_is_broadcast, extract_uint8_seq

_LOGGER = logging.getLogger(__name__)

RGB_PIXEL_PROFILE = "RGB / RGBW pixels"
SHEHDS_GALAXYJET_300_3IN1_23CH = (
    "SHEHDS GalaxyJet 300 3-in-1 (23CH)"
)

FIXTURE_PROFILES = (
    RGB_PIXEL_PROFILE,
    SHEHDS_GALAXYJET_300_3IN1_23CH,
)

AUDIO_POWER_SOURCES = (
    "Volume",
    "Beat",
    "Bass",
    "Lows",
    "Mids",
    "Highs",
)

AUDIO_OSCILLATOR_SOURCES = (
    "Beat oscillator",
    "Bar oscillator",
)

AUDIO_EVENT_SOURCES = (
    "Volume beat",
    "BPM beat",
    "Onset",
)

CONTROL_SOURCES = (
    "Off",
    "Effect brightness",
    "Effect hue",
    *AUDIO_POWER_SOURCES,
    *AUDIO_OSCILLATOR_SOURCES,
    *AUDIO_EVENT_SOURCES,
)

MOVEMENT_MODES = (
    "Static",
    "Sweep",
    "Mirror sweep",
    "Circle",
    "BPM synced sweep",
    "Bar synced circle",
    "Audio position",
)

MOVEMENT_BEAT_ACTIONS = (
    "None",
    "Reverse direction",
    "Jump 180 degrees",
    "Reset phase",
)

DIMMER_MODES = (
    "Effect brightness",
    "Audio source",
    "Effect x audio",
    "Max effect/audio",
    "Fixed",
)

COLOR_MODES = (
    "Effect hue",
    "Fixed DMX",
    "Timed cycle",
    "Audio source",
    "Beat cycle",
)

WHEEL_MODES = (
    "Fixed DMX",
    "Timed cycle",
    "Audio source",
    "Effect brightness",
    "Beat cycle",
)

RAW_MODES = (
    "Fixed DMX",
    "Audio source",
)

STROBE_MODES = (
    "Open",
    "Fixed DMX",
    "Audio source",
    "Beat pulse",
)

BINARY_MODES = (
    "Off",
    "On",
    "Audio threshold",
    "Beat toggle",
)

AUX_MODES = (
    "Fixed DMX",
    "Audio source",
    "Effect brightness",
)

SHEHDS_CHANNEL_COUNT = 23


class ArtNetDevice(NetworkedDevice):
    """Art-Net device support, including optional DMX fixture profiles."""

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Required(
                "pixel_count",
                description=(
                    "Number of individual pixels. With the SHEHDS profile, "
                    "each pixel represents one moving head."
                ),
                default=1,
            ): vol.All(int, vol.Range(min=1)),
            vol.Optional(
                "universe",
                description="DMX universe for the device",
                default=0,
            ): vol.All(int, vol.Range(min=0)),
            vol.Optional(
                "packet_size",
                description="Size of each DMX universe",
                default=510,
            ): vol.All(int, vol.Range(min=1, max=512)),
            vol.Optional(
                "pre_amble",
                description="Channel bytes to insert before the RGB data",
                default="",
            ): str,
            vol.Optional(
                "post_amble",
                description="Channel bytes to insert after the RGB data",
                default="",
            ): str,
            vol.Optional(
                "pixels_per_device",
                description=(
                    "Number of pixels to consume per device. Pre and post "
                    "ambles are repeated per device. By default (0), all "
                    "pixels are used by one instance."
                ),
                default=0,
            ): vol.All(int, vol.Range(min=0)),
            vol.Optional(
                "dmx_start_address",
                description=(
                    "First DMX address. For fixture profiles, this is used "
                    "when fixture_addresses is left blank."
                ),
                default=1,
            ): vol.All(int, vol.Range(min=1, max=512)),
            vol.Optional(
                "even_packet_size",
                description="Whether to use even packet size",
                default=True,
            ): bool,
            vol.Optional(
                "rgb_order",
                description=(
                    "RGB data order mode for RGB/RGBW pixel hardware"
                ),
                default="RGB",
            ): vol.All(str, vol.In(RGB_MAPPING)),
            vol.Optional(
                "white_mode",
                description=(
                    "White-channel handling mode for RGB/RGBW hardware"
                ),
                default="None",
            ): vol.All(str, vol.In(WHITE_FUNCS_MAPPING.keys())),
            vol.Optional(
                "fixture_profile",
                description=(
                    "Select a DMX fixture profile. Leave on RGB / RGBW "
                    "pixels for normal Art-Net LED output."
                ),
                default=RGB_PIXEL_PROFILE,
            ): vol.All(str, vol.In(FIXTURE_PROFILES)),
            vol.Optional(
                "fixture_addresses",
                description=(
                    "Comma-separated 1-based fixture start addresses, e.g. "
                    "1,24,47,70. Blank uses contiguous 23CH addresses."
                ),
                default="",
            ): str,
            vol.Optional(
                "audio_control_enabled",
                description=(
                    "Use LEDFx's existing audio analysis to drive fixture "
                    "controls. This does not create a second music detector."
                ),
                default=True,
            ): bool,
            vol.Optional(
                "audio_gain",
                description=(
                    "Gain applied to LEDFx volume/frequency power signals"
                ),
                default=1.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=4.0)),
            vol.Optional(
                "audio_curve",
                description=(
                    "Response curve for LEDFx volume/frequency power. "
                    "Below 1 is more sensitive; above 1 is less sensitive."
                ),
                default=1.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=4.0)),
            vol.Optional(
                "audio_pulse_hold",
                description=(
                    "Seconds that beat/onset events remain active for DMX "
                    "mapping"
                ),
                default=0.10,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.02, max=1.0)),
            vol.Optional(
                "movement_mode",
                description="Pan/tilt movement pattern",
                default="Sweep",
            ): vol.All(str, vol.In(MOVEMENT_MODES)),
            vol.Optional(
                "pan",
                description="Pan centre / static position (0-255)",
                default=127,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "tilt",
                description="Tilt centre / static position (0-255)",
                default=127,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "pan_range",
                description="Pan travel around the centre (0-255)",
                default=160,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "tilt_range",
                description="Tilt travel around the centre (0-255)",
                default=96,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "pan_invert",
                description="Invert generated pan",
                default=False,
            ): bool,
            vol.Optional(
                "tilt_invert",
                description="Invert generated tilt",
                default=False,
            ): bool,
            vol.Optional(
                "fixture_phase_spread",
                description=(
                    "Movement phase spread between fixtures. 0 moves all "
                    "heads together; 1 spreads them over a full cycle."
                ),
                default=1.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "movement_period",
                description="Normal seconds for one generated movement cycle",
                default=6.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.20, max=120.0)),
            vol.Optional(
                "movement_speed_source",
                description=(
                    "LEDFx/effect signal that changes generated movement "
                    "speed"
                ),
                default="Lows",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "movement_slow_period",
                description=(
                    "Movement cycle time when the selected speed signal is "
                    "quiet/low"
                ),
                default=10.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.20, max=120.0)),
            vol.Optional(
                "movement_fast_period",
                description=(
                    "Movement cycle time when the selected speed signal is "
                    "loud/high"
                ),
                default=1.2,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.10, max=120.0)),
            vol.Optional(
                "movement_speed_amount",
                description=(
                    "How strongly the selected signal changes movement "
                    "speed; 0 uses movement_period only"
                ),
                default=1.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "movement_beat_source",
                description="Event used for movement beat actions",
                default="Volume beat",
            ): vol.All(str, vol.In(AUDIO_EVENT_SOURCES)),
            vol.Optional(
                "movement_beat_action",
                description="Optional action each detected beat/onset",
                default="None",
            ): vol.All(str, vol.In(MOVEMENT_BEAT_ACTIONS)),
            vol.Optional(
                "pan_source",
                description=(
                    "Optional signal mapped into pan in addition to the "
                    "selected movement pattern"
                ),
                default="Off",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "pan_source_amount",
                description="Amount of pan modulation from pan_source",
                default=0.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "tilt_source",
                description=(
                    "Optional signal mapped into tilt in addition to the "
                    "selected movement pattern"
                ),
                default="Off",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "tilt_source_amount",
                description="Amount of tilt modulation from tilt_source",
                default=0.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "pt_speed_mode",
                description="Control SHEHDS CH5 pan/tilt motor speed",
                default="Fixed DMX",
            ): vol.All(str, vol.In(RAW_MODES)),
            vol.Optional(
                "pt_speed_value",
                description="Fixed SHEHDS CH5 pan/tilt speed DMX value",
                default=128,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "pt_speed_source",
                description="Signal mapped to SHEHDS CH5 motor speed",
                default="Lows",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "pt_speed_min",
                description="Minimum CH5 value for audio mapping",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "pt_speed_max",
                description="Maximum CH5 value for audio mapping",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "dimmer_mode",
                description="How fixture dimmer follows LEDFx/music",
                default="Effect brightness",
            ): vol.All(str, vol.In(DIMMER_MODES)),
            vol.Optional(
                "dimmer_source",
                description="Audio/effect source for audio dimmer modes",
                default="Lows",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "dimmer_fixed",
                description="Fixed dimmer value when dimmer_mode is Fixed",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "dimmer_floor",
                description="Minimum output dimmer",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "master_dimmer",
                description="Maximum output dimmer",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "color_mode",
                description="How the fixture color wheel is controlled",
                default="Effect hue",
            ): vol.All(str, vol.In(COLOR_MODES)),
            vol.Optional(
                "color_value",
                description="Fixed raw color-wheel DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "color_source",
                description="Signal mapped to the color wheel",
                default="Beat oscillator",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "color_min",
                description="Minimum color-wheel DMX value for mapping",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "color_max",
                description=(
                    "Maximum color-wheel DMX value for mapping. Default "
                    "keeps automatic mapping in the lower half of the wheel."
                ),
                default=127,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "color_cycle_values",
                description=(
                    "Comma-separated raw color DMX values for timed/beat "
                    "cycle modes"
                ),
                default="0",
            ): str,
            vol.Optional(
                "color_cycle_period",
                description="Seconds between timed color changes",
                default=1.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.05, max=120.0)),
            vol.Optional(
                "color_beat_source",
                description="Event used by Beat cycle color mode",
                default="Volume beat",
            ): vol.All(str, vol.In(AUDIO_EVENT_SOURCES)),
            vol.Optional(
                "gobo1_mode",
                description="How gobo wheel 1 is controlled",
                default="Fixed DMX",
            ): vol.All(str, vol.In(WHEEL_MODES)),
            vol.Optional(
                "gobo1_value",
                description="Fixed raw DMX value for gobo wheel 1",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "gobo1_source",
                description="Signal mapped to gobo wheel 1",
                default="Beat",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "gobo1_min",
                description="Minimum gobo 1 DMX value for mapping",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "gobo1_max",
                description="Maximum gobo 1 DMX value for mapping",
                default=127,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "gobo1_cycle_values",
                description="Comma-separated raw gobo 1 DMX cycle values",
                default="0",
            ): str,
            vol.Optional(
                "gobo1_cycle_period",
                description="Seconds between timed gobo 1 changes",
                default=2.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.05, max=120.0)),
            vol.Optional(
                "gobo1_beat_source",
                description="Event used by gobo 1 Beat cycle",
                default="Volume beat",
            ): vol.All(str, vol.In(AUDIO_EVENT_SOURCES)),
            vol.Optional(
                "gobo2_mode",
                description="How rotating gobo wheel 2 is controlled",
                default="Fixed DMX",
            ): vol.All(str, vol.In(WHEEL_MODES)),
            vol.Optional(
                "gobo2_value",
                description="Fixed raw DMX value for gobo wheel 2",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "gobo2_source",
                description="Signal mapped to gobo wheel 2",
                default="Mids",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "gobo2_min",
                description="Minimum gobo 2 DMX value for mapping",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "gobo2_max",
                description="Maximum gobo 2 DMX value for mapping",
                default=127,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "gobo2_cycle_values",
                description="Comma-separated raw gobo 2 DMX cycle values",
                default="0",
            ): str,
            vol.Optional(
                "gobo2_cycle_period",
                description="Seconds between timed gobo 2 changes",
                default=2.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.05, max=120.0)),
            vol.Optional(
                "gobo2_beat_source",
                description="Event used by gobo 2 Beat cycle",
                default="Volume beat",
            ): vol.All(str, vol.In(AUDIO_EVENT_SOURCES)),
            vol.Optional(
                "gobo2_rotation_mode",
                description="How gobo wheel 2 rotation is controlled",
                default="Fixed DMX",
            ): vol.All(str, vol.In(RAW_MODES)),
            vol.Optional(
                "gobo2_rotation_value",
                description="Fixed raw gobo 2 rotation DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "gobo2_rotation_source",
                description="Signal mapped to gobo 2 rotation",
                default="Highs",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "gobo2_rotation_min",
                description="Minimum gobo 2 rotation DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "gobo2_rotation_max",
                description="Maximum gobo 2 rotation DMX value",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "strobe_mode",
                description=(
                    "Fixture shutter/strobe control. Beat/audio modes can "
                    "flash the light; use appropriate care."
                ),
                default="Open",
            ): vol.All(str, vol.In(STROBE_MODES)),
            vol.Optional(
                "strobe_value",
                description="Fixed raw strobe DMX value",
                default=250,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "strobe_source",
                description="Signal mapped to strobe rate",
                default="Highs",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "strobe_min",
                description="Slow strobe DMX value",
                default=4,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "strobe_max",
                description="Fast strobe DMX value",
                default=103,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "strobe_threshold",
                description=(
                    "Below this source level the shutter stays open"
                ),
                default=0.15,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "strobe_beat_source",
                description="Event used for Beat pulse strobe",
                default="Volume beat",
            ): vol.All(str, vol.In(AUDIO_EVENT_SOURCES)),
            vol.Optional(
                "strobe_beat_value",
                description="Raw strobe DMX value during a beat pulse",
                default=80,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "prism_mode",
                description="Prism insertion control",
                default="Off",
            ): vol.All(str, vol.In(BINARY_MODES)),
            vol.Optional(
                "prism_on_value",
                description="Raw DMX value used when prism is on",
                default=128,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "prism_source",
                description="Signal used for prism audio threshold",
                default="Beat",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "prism_threshold",
                description="Prism audio threshold",
                default=0.6,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "prism_beat_source",
                description="Event used for prism Beat toggle",
                default="Volume beat",
            ): vol.All(str, vol.In(AUDIO_EVENT_SOURCES)),
            vol.Optional(
                "prism_rotation_mode",
                description="Prism rotation control",
                default="Fixed DMX",
            ): vol.All(str, vol.In(RAW_MODES)),
            vol.Optional(
                "prism_rotation_value",
                description=(
                    "Fixed raw prism rotation value; 191-192 is stop in "
                    "the SHEHDS manual."
                ),
                default=191,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "prism_rotation_source",
                description="Signal mapped to prism rotation",
                default="Highs",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "prism_rotation_min",
                description="Minimum prism rotation DMX value",
                default=128,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "prism_rotation_max",
                description="Maximum prism rotation DMX value",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "frost_mode",
                description="Frost insertion control",
                default="Off",
            ): vol.All(str, vol.In(BINARY_MODES)),
            vol.Optional(
                "frost_on_value",
                description="Raw DMX value used when frost is on",
                default=128,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "frost_source",
                description="Signal used for frost audio threshold",
                default="Lows",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "frost_threshold",
                description="Frost audio threshold",
                default=0.7,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "frost_beat_source",
                description="Event used for frost Beat toggle",
                default="Volume beat",
            ): vol.All(str, vol.In(AUDIO_EVENT_SOURCES)),
            vol.Optional(
                "zoom_mode",
                description="Zoom control mode",
                default="Fixed DMX",
            ): vol.All(str, vol.In(AUX_MODES)),
            vol.Optional(
                "zoom_value",
                description="Fixed zoom DMX value",
                default=128,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "zoom_source",
                description="Signal mapped to zoom",
                default="Lows",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "zoom_min",
                description="Minimum zoom DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "zoom_max",
                description="Maximum zoom DMX value",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "focus_mode",
                description="Focus control mode",
                default="Fixed DMX",
            ): vol.All(str, vol.In(AUX_MODES)),
            vol.Optional(
                "focus_value",
                description="Fixed focus DMX value",
                default=128,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "focus_source",
                description="Signal mapped to focus",
                default="Mids",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "focus_min",
                description="Minimum focus DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "focus_max",
                description="Maximum focus DMX value",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "ring_strobe_mode",
                description="Ring strobe control",
                default="Fixed DMX",
            ): vol.All(str, vol.In(STROBE_MODES)),
            vol.Optional(
                "ring_strobe_value",
                description="Fixed ring strobe DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "ring_strobe_source",
                description="Signal mapped to ring strobe",
                default="Highs",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "ring_strobe_min",
                description="Minimum ring strobe DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "ring_strobe_max",
                description="Maximum ring strobe DMX value",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "ring_strobe_beat_source",
                description="Event used for ring Beat pulse",
                default="Volume beat",
            ): vol.All(str, vol.In(AUDIO_EVENT_SOURCES)),
            vol.Optional(
                "ring_mode_control",
                description="Ring effect mode control",
                default="Fixed DMX",
            ): vol.All(str, vol.In(RAW_MODES)),
            vol.Optional(
                "ring_mode_value",
                description="Fixed ring effect mode DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "ring_mode_source",
                description="Signal mapped to ring effect mode",
                default="Beat oscillator",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "ring_mode_min",
                description="Minimum ring effect mode DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "ring_mode_max",
                description="Maximum ring effect mode DMX value",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "ring_speed_mode",
                description="Ring effect speed control",
                default="Fixed DMX",
            ): vol.All(str, vol.In(RAW_MODES)),
            vol.Optional(
                "ring_speed_value",
                description="Fixed ring effect speed DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "ring_speed_source",
                description="Signal mapped to ring effect speed",
                default="Lows",
            ): vol.All(str, vol.In(CONTROL_SOURCES)),
            vol.Optional(
                "ring_speed_min",
                description="Minimum ring effect speed DMX value",
                default=0,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional(
                "ring_speed_max",
                description="Maximum ring effect speed DMX value",
                default=255,
            ): vol.All(int, vol.Range(min=0, max=255)),
            vol.Optional("port", description="port", default=6454): int,
        }
    )

    def __init__(self, ledfx, config):
        super().__init__(ledfx, config)
        self._artnet = None
        self._device_type = "ArtNet"
        self._fixture_audio = None
        self._audio_lock = threading.Lock()
        self._audio_state = self._empty_audio_state()
        self._fixture_epoch = time.monotonic()
        self._movement_phase = 0.0
        self._movement_direction = 1
        self._movement_last_time = self._fixture_epoch
        self._handled_movement_events = {}
        self.config_use(config)
        self.init = True

    @staticmethod
    def _empty_audio_state():
        return {
            "Volume": 0.0,
            "Beat": 0.0,
            "Bass": 0.0,
            "Lows": 0.0,
            "Mids": 0.0,
            "Highs": 0.0,
            "Beat oscillator": 0.0,
            "Bar oscillator": 0.0,
            "event_times": {
                "Volume beat": 0.0,
                "BPM beat": 0.0,
                "Onset": 0.0,
            },
            "event_counts": {
                "Volume beat": 0,
                "BPM beat": 0,
                "Onset": 0,
            },
        }

    def config_updated(self, config):
        self.config_use(config)
        self.deactivate()
        self.init = True
        self.activate()

    def config_use(self, config):
        self.pre_amble = np.array(
            extract_uint8_seq(config.get("pre_amble", "")), dtype=np.uint8
        )
        self.post_amble = np.array(
            extract_uint8_seq(config.get("post_amble", "")), dtype=np.uint8
        )
        self.pixels_per_device = config.get("pixels_per_device", 0)
        self.dmx_start_address = config.get("dmx_start_address", 1) - 1
        self.rgb_mode = config.get("rgb_order", "RGB")
        self.white_mode = config.get("white_mode", "None")
        self.packet_size = config.get(
            "packet_size", self._config.get("packet_size", 510)
        )
        self.fixture_profile = config.get(
            "fixture_profile", RGB_PIXEL_PROFILE
        )
        self.fixture_addresses_text = config.get("fixture_addresses", "")
        self.audio_control_enabled = config.get(
            "audio_control_enabled", True
        )
        self.audio_gain = float(config.get("audio_gain", 1.0))
        self.audio_curve = float(config.get("audio_curve", 1.0))
        self.audio_pulse_hold = float(config.get("audio_pulse_hold", 0.10))

        self.color_cycle_values = self._parse_dmx_values(
            config.get("color_cycle_values", "0"),
            label="color_cycle_values",
        )
        self.gobo1_cycle_values = self._parse_dmx_values(
            config.get("gobo1_cycle_values", "0"),
            label="gobo1_cycle_values",
        )
        self.gobo2_cycle_values = self._parse_dmx_values(
            config.get("gobo2_cycle_values", "0"),
            label="gobo2_cycle_values",
        )

    @staticmethod
    def _parse_dmx_values(value, fallback=(0,), label="DMX values"):
        values = []
        for part in str(value or "").replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                number = int(part)
            except ValueError:
                _LOGGER.warning("Ignoring invalid %s entry: %s", label, part)
                continue
            if 0 <= number <= 255:
                values.append(number)
            else:
                _LOGGER.warning(
                    "Ignoring out-of-range %s entry: %s", label, number
                )
        return tuple(values) if values else tuple(fallback)

    def _fixture_addresses(self):
        addresses = []
        text = str(self.fixture_addresses_text or "").strip()
        if text:
            for part in text.replace(";", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    address = int(part)
                except ValueError:
                    _LOGGER.warning(
                        "Ignoring invalid fixture address: %s", part
                    )
                    continue
                if 1 <= address <= 512:
                    addresses.append(address - 1)
                else:
                    _LOGGER.warning(
                        "Ignoring fixture address outside 1-512: %s",
                        address,
                    )

        while len(addresses) < self.pixel_count:
            if addresses:
                next_address = addresses[-1] + SHEHDS_CHANNEL_COUNT
            else:
                next_address = (
                    self.dmx_start_address
                    + len(addresses) * SHEHDS_CHANNEL_COUNT
                )
            addresses.append(next_address)

        addresses = addresses[: self.pixel_count]

        if any(
            address < 0 or address + SHEHDS_CHANNEL_COUNT > 512
            for address in addresses
        ):
            raise ValueError(
                "SHEHDS fixture addresses must fit completely inside one "
                "512-channel DMX universe"
            )

        return addresses

    def _activate_fixture_audio(self):
        if (
            self.fixture_profile != SHEHDS_GALAXYJET_300_3IN1_23CH
            or not self.audio_control_enabled
            or self._fixture_audio is not None
        ):
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
            self._fixture_audio.subscribe(self._audio_data_updated)
            _LOGGER.info(
                "Art-Net fixture %s subscribed to LEDFx audio analysis",
                self.name,
            )
        except Exception:
            self._fixture_audio = None
            _LOGGER.exception(
                "Unable to subscribe Art-Net fixture %s to LEDFx audio",
                self.name,
            )

    def _deactivate_fixture_audio(self):
        if self._fixture_audio is None:
            return
        try:
            self._fixture_audio.unsubscribe(self._audio_data_updated)
        except Exception:
            _LOGGER.exception(
                "Unable to unsubscribe Art-Net fixture %s from LEDFx audio",
                self.name,
            )
        self._fixture_audio = None

    def _audio_data_updated(self):
        audio = self._fixture_audio
        if audio is None:
            return

        now = time.monotonic()
        try:
            values = {
                "Volume": float(audio.volume()),
                "Beat": float(audio.beat_power()),
                "Bass": float(audio.bass_power()),
                "Lows": float(audio.lows_power()),
                "Mids": float(audio.mids_power()),
                "Highs": float(audio.high_power()),
                "Beat oscillator": float(audio.beat_oscillator()),
                "Bar oscillator": float(audio.bar_oscillator()) / 4.0,
            }
            events = {
                "Volume beat": bool(audio.volume_beat_now()),
                "BPM beat": bool(audio.bpm_beat_now()),
                "Onset": bool(audio.onset()),
            }
        except Exception:
            _LOGGER.debug(
                "LEDFx audio analysis unavailable for Art-Net fixture",
                exc_info=True,
            )
            return

        with self._audio_lock:
            for key, value in values.items():
                if np.isnan(value):
                    value = 0.0
                self._audio_state[key] = max(0.0, min(1.0, value))

            for key, active in events.items():
                if active:
                    self._audio_state["event_times"][key] = now
                    self._audio_state["event_counts"][key] += 1

    def _audio_snapshot(self):
        with self._audio_lock:
            snapshot = {
                key: value
                for key, value in self._audio_state.items()
                if key not in ("event_times", "event_counts")
            }
            snapshot["event_times"] = dict(
                self._audio_state["event_times"]
            )
            snapshot["event_counts"] = dict(
                self._audio_state["event_counts"]
            )
        return snapshot

    def _shape_audio(self, value):
        value = max(0.0, min(1.0, float(value) * self.audio_gain))
        return max(0.0, min(1.0, value**self.audio_curve))

    def _source_value(
        self,
        source,
        effect_hue,
        effect_brightness,
        audio,
        now,
    ):
        if source == "Off":
            return 0.0
        if source == "Effect brightness":
            return max(0.0, min(1.0, effect_brightness / 255.0))
        if source == "Effect hue":
            return max(0.0, min(1.0, effect_hue))
        if source in AUDIO_POWER_SOURCES:
            return self._shape_audio(audio.get(source, 0.0))
        if source in AUDIO_OSCILLATOR_SOURCES:
            return max(0.0, min(1.0, audio.get(source, 0.0)))
        if source in AUDIO_EVENT_SOURCES:
            last_time = audio["event_times"].get(source, 0.0)
            return (
                1.0
                if now - last_time <= self.audio_pulse_hold
                else 0.0
            )
        return 0.0

    @staticmethod
    def _map_value(value, minimum, maximum):
        value = max(0.0, min(1.0, float(value)))
        return float(minimum) + value * (float(maximum) - float(minimum))

    @staticmethod
    def _coarse_fine(value):
        value = min(255.0, max(0.0, float(value)))
        full = int(round(value * 257.0))
        return (full >> 8) & 0xFF, full & 0xFF

    def activate(self):
        if self._artnet:
            _LOGGER.warning(
                "Art-Net sender already started for device %s",
                self.config["name"],
            )

        broadcast = check_if_ip_is_broadcast(self._config["ip_address"])
        self._artnet = StupidArtnet(
            target_ip=self._config["ip_address"],
            universe=self._config["universe"],
            packet_size=self.packet_size,
            fps=self._config["refresh_rate"],
            even_packet_size=self._config["even_packet_size"],
            broadcast=broadcast,
            port=self._config["port"],
        )

        super().activate()
        self._activate_fixture_audio()
        self.init = True

    def deactivate(self):
        self._deactivate_fixture_audio()
        super().deactivate()
        if not self._artnet:
            return

        self._artnet.blackout()
        self._artnet.close()
        self._artnet = None

    def do_once(self):
        self.output_mode = OutputMode(self.rgb_mode, self.white_mode)

        if self.fixture_profile == SHEHDS_GALAXYJET_300_3IN1_23CH:
            self.fixture_addresses = self._fixture_addresses()
            self.num_devices = self.pixel_count
            self.data_max = self.pixel_count
            self.channel_count = max(
                address + SHEHDS_CHANNEL_COUNT
                for address in self.fixture_addresses
            )
            self.universe_count = math.ceil(
                self.channel_count / self.packet_size
            )
            self.init = False
            return

        if (
            self.pixels_per_device == 0
            or self.pixels_per_device > self.pixel_count
        ):
            self.use_pixels_per_device = self.pixel_count
        else:
            self.use_pixels_per_device = self.pixels_per_device

        self.num_devices = self.pixel_count // self.use_pixels_per_device
        self.data_max = self.num_devices * self.use_pixels_per_device

        total_pixels_per_device = (
            self.pre_amble.size
            + (
                self.use_pixels_per_device
                * self.output_mode.channels_per_pixel
            )
            + self.post_amble.size
        )
        self.channel_count = (
            self.dmx_start_address
            + total_pixels_per_device * self.num_devices
        )
        self.universe_count = math.ceil(
            self.channel_count / self.packet_size
        )
        self.init = False

    def _advance_movement_phase(
        self,
        effect_hue,
        effect_brightness,
        audio,
        now,
    ):
        delta = max(0.0, min(0.25, now - self._movement_last_time))
        self._movement_last_time = now

        source = self._config.get("movement_speed_source", "Lows")
        amount = float(self._config.get("movement_speed_amount", 1.0))
        base_period = float(self._config.get("movement_period", 6.0))

        if source == "Off" or amount <= 0.0:
            period = base_period
        else:
            signal = self._source_value(
                source,
                effect_hue,
                effect_brightness,
                audio,
                now,
            )
            slow = float(self._config.get("movement_slow_period", 10.0))
            fast = float(self._config.get("movement_fast_period", 1.2))
            audio_period = slow + signal * (fast - slow)
            period = base_period + amount * (audio_period - base_period)

        period = max(0.10, period)

        event_source = self._config.get(
            "movement_beat_source", "Volume beat"
        )
        event_count = audio["event_counts"].get(event_source, 0)
        last_count = self._handled_movement_events.get(
            event_source, event_count
        )
        event_delta = event_count - last_count
        self._handled_movement_events[event_source] = event_count

        action = self._config.get("movement_beat_action", "None")
        if event_delta > 0:
            if action == "Reverse direction" and event_delta % 2:
                self._movement_direction *= -1
            elif action == "Jump 180 degrees":
                self._movement_phase += math.pi * event_delta
            elif action == "Reset phase":
                self._movement_phase = 0.0

        self._movement_phase += (
            self._movement_direction * delta * math.tau / period
        )
        self._movement_phase %= math.tau

    def _movement_values(
        self,
        fixture_index,
        effect_hue,
        effect_brightness,
        audio,
        now,
    ):
        centre_pan = float(self._config.get("pan", 127))
        centre_tilt = float(self._config.get("tilt", 127))
        pan_range = float(self._config.get("pan_range", 160))
        tilt_range = float(self._config.get("tilt_range", 96))
        spread = float(self._config.get("fixture_phase_spread", 1.0))
        fixture_phase = (
            (fixture_index / max(1, self.pixel_count))
            * math.tau
            * spread
        )
        phase = self._movement_phase + fixture_phase
        mode = self._config.get("movement_mode", "Sweep")

        pan = centre_pan
        tilt = centre_tilt

        if mode == "Sweep":
            pan += math.sin(phase) * pan_range / 2.0
            tilt += math.sin(phase * 0.5) * tilt_range / 2.0
        elif mode == "Mirror sweep":
            mirror = -1.0 if fixture_index % 2 else 1.0
            pan += mirror * math.sin(phase) * pan_range / 2.0
            tilt += math.cos(phase) * tilt_range / 2.0
        elif mode == "Circle":
            pan += math.sin(phase) * pan_range / 2.0
            tilt += math.cos(phase) * tilt_range / 2.0
        elif mode == "BPM synced sweep":
            synced = (
                audio.get("Beat oscillator", 0.0) * math.tau
                + fixture_phase
            )
            pan += math.sin(synced) * pan_range / 2.0
            tilt += math.sin(synced * 0.5) * tilt_range / 2.0
        elif mode == "Bar synced circle":
            synced = (
                audio.get("Bar oscillator", 0.0) * math.tau
                + fixture_phase
            )
            pan += math.sin(synced) * pan_range / 2.0
            tilt += math.cos(synced) * tilt_range / 2.0
        elif mode == "Audio position":
            pan_source = self._config.get("pan_source", "Lows")
            tilt_source = self._config.get("tilt_source", "Mids")
            pan_level = self._source_value(
                pan_source,
                effect_hue,
                effect_brightness,
                audio,
                now,
            )
            tilt_level = self._source_value(
                tilt_source,
                effect_hue,
                effect_brightness,
                audio,
                now,
            )
            pan = centre_pan + (pan_level - 0.5) * pan_range
            tilt = centre_tilt + (tilt_level - 0.5) * tilt_range

        if mode != "Audio position":
            pan_source = self._config.get("pan_source", "Off")
            pan_amount = float(
                self._config.get("pan_source_amount", 0.0)
            )
            if pan_source != "Off" and pan_amount > 0.0:
                level = self._source_value(
                    pan_source,
                    effect_hue,
                    effect_brightness,
                    audio,
                    now,
                )
                pan += (level - 0.5) * pan_range * pan_amount

            tilt_source = self._config.get("tilt_source", "Off")
            tilt_amount = float(
                self._config.get("tilt_source_amount", 0.0)
            )
            if tilt_source != "Off" and tilt_amount > 0.0:
                level = self._source_value(
                    tilt_source,
                    effect_hue,
                    effect_brightness,
                    audio,
                    now,
                )
                tilt += (level - 0.5) * tilt_range * tilt_amount

        pan = min(255.0, max(0.0, pan))
        tilt = min(255.0, max(0.0, tilt))

        if self._config.get("pan_invert", False):
            pan = 255.0 - pan
        if self._config.get("tilt_invert", False):
            tilt = 255.0 - tilt

        return pan, tilt

    def _raw_control(
        self,
        mode,
        fixed,
        source,
        minimum,
        maximum,
        effect_hue,
        effect_brightness,
        audio,
        now,
    ):
        if mode == "Audio source":
            level = self._source_value(
                source,
                effect_hue,
                effect_brightness,
                audio,
                now,
            )
            return int(round(self._map_value(level, minimum, maximum)))
        return int(fixed)

    def _dimmer_value(
        self,
        effect_hue,
        effect_brightness,
        audio,
        now,
    ):
        effect_level = max(
            0.0, min(1.0, effect_brightness / 255.0)
        )
        source = self._config.get("dimmer_source", "Lows")
        audio_level = self._source_value(
            source,
            effect_hue,
            effect_brightness,
            audio,
            now,
        )
        mode = self._config.get("dimmer_mode", "Effect brightness")

        if mode == "Audio source":
            level = audio_level
        elif mode == "Effect x audio":
            level = effect_level * audio_level
        elif mode == "Max effect/audio":
            level = max(effect_level, audio_level)
        elif mode == "Fixed":
            return float(self._config.get("dimmer_fixed", 255))
        else:
            level = effect_level

        floor = float(self._config.get("dimmer_floor", 0))
        ceiling = float(self._config.get("master_dimmer", 255))
        return self._map_value(level, floor, ceiling)

    def _cycle_value(
        self,
        values,
        timed_period,
        beat_source,
        use_beat,
        fixture_index,
        now,
        audio,
    ):
        if use_beat:
            step = audio["event_counts"].get(beat_source, 0)
        else:
            step = int(
                (now - self._fixture_epoch)
                / max(0.05, float(timed_period))
            )
        return values[(step + fixture_index) % len(values)]

    def _wheel_value(
        self,
        prefix,
        values,
        fixture_index,
        effect_hue,
        effect_brightness,
        audio,
        now,
    ):
        mode = self._config.get(f"{prefix}_mode", "Fixed DMX")
        fixed = int(self._config.get(f"{prefix}_value", 0))

        if mode == "Timed cycle":
            return self._cycle_value(
                values,
                self._config.get(f"{prefix}_cycle_period", 2.0),
                self._config.get(
                    f"{prefix}_beat_source", "Volume beat"
                ),
                False,
                fixture_index,
                now,
                audio,
            )
        if mode == "Beat cycle":
            return self._cycle_value(
                values,
                self._config.get(f"{prefix}_cycle_period", 2.0),
                self._config.get(
                    f"{prefix}_beat_source", "Volume beat"
                ),
                True,
                fixture_index,
                now,
                audio,
            )
        if mode == "Audio source":
            level = self._source_value(
                self._config.get(f"{prefix}_source", "Beat"),
                effect_hue,
                effect_brightness,
                audio,
                now,
            )
            return int(
                round(
                    self._map_value(
                        level,
                        self._config.get(f"{prefix}_min", 0),
                        self._config.get(f"{prefix}_max", 127),
                    )
                )
            )
        if mode == "Effect brightness":
            level = max(
                0.0, min(1.0, effect_brightness / 255.0)
            )
            return int(
                round(
                    self._map_value(
                        level,
                        self._config.get(f"{prefix}_min", 0),
                        self._config.get(f"{prefix}_max", 127),
                    )
                )
            )
        return fixed

    def _color_value(
        self,
        fixture_index,
        effect_hue,
        effect_saturation,
        effect_brightness,
        audio,
        now,
    ):
        mode = self._config.get("color_mode", "Effect hue")
        if mode == "Fixed DMX":
            return float(self._config.get("color_value", 0))
        if mode == "Timed cycle":
            return float(
                self._cycle_value(
                    self.color_cycle_values,
                    self._config.get("color_cycle_period", 1.0),
                    self._config.get(
                        "color_beat_source", "Volume beat"
                    ),
                    False,
                    fixture_index,
                    now,
                    audio,
                )
            )
        if mode == "Beat cycle":
            return float(
                self._cycle_value(
                    self.color_cycle_values,
                    self._config.get("color_cycle_period", 1.0),
                    self._config.get(
                        "color_beat_source", "Volume beat"
                    ),
                    True,
                    fixture_index,
                    now,
                    audio,
                )
            )

        if mode == "Audio source":
            level = self._source_value(
                self._config.get(
                    "color_source", "Beat oscillator"
                ),
                effect_hue,
                effect_brightness,
                audio,
                now,
            )
        else:
            level = 0.0 if effect_saturation < 0.08 else effect_hue

        return self._map_value(
            level,
            self._config.get("color_min", 0),
            self._config.get("color_max", 127),
        )

    def _event_active(self, source, audio, now):
        return self._source_value(
            source, 0.0, 0.0, audio, now
        ) >= 0.5

    def _binary_value(
        self,
        prefix,
        effect_hue,
        effect_brightness,
        audio,
        now,
    ):
        mode = self._config.get(f"{prefix}_mode", "Off")
        on_value = int(self._config.get(f"{prefix}_on_value", 128))

        if mode == "On":
            return on_value
        if mode == "Audio threshold":
            level = self._source_value(
                self._config.get(f"{prefix}_source", "Beat"),
                effect_hue,
                effect_brightness,
                audio,
                now,
            )
            return (
                on_value
                if level
                >= float(self._config.get(f"{prefix}_threshold", 0.6))
                else 0
            )
        if mode == "Beat toggle":
            source = self._config.get(
                f"{prefix}_beat_source", "Volume beat"
            )
            count = audio["event_counts"].get(source, 0)
            return on_value if count % 2 else 0
        return 0

    def _aux_value(
        self,
        prefix,
        effect_hue,
        effect_brightness,
        audio,
        now,
    ):
        mode = self._config.get(f"{prefix}_mode", "Fixed DMX")
        if mode == "Audio source":
            level = self._source_value(
                self._config.get(f"{prefix}_source", "Lows"),
                effect_hue,
                effect_brightness,
                audio,
                now,
            )
            return self._map_value(
                level,
                self._config.get(f"{prefix}_min", 0),
                self._config.get(f"{prefix}_max", 255),
            )
        if mode == "Effect brightness":
            level = max(
                0.0, min(1.0, effect_brightness / 255.0)
            )
            return self._map_value(
                level,
                self._config.get(f"{prefix}_min", 0),
                self._config.get(f"{prefix}_max", 255),
            )
        return float(self._config.get(f"{prefix}_value", 128))

    def _strobe_value(
        self,
        prefix,
        effect_hue,
        effect_brightness,
        audio,
        now,
        fixture_strobe=True,
    ):
        mode = self._config.get(f"{prefix}_mode", "Open")
        fixed = int(
            self._config.get(
                f"{prefix}_value", 250 if fixture_strobe else 0
            )
        )
        open_value = 250 if fixture_strobe else 0

        if mode == "Open":
            return open_value
        if mode == "Fixed DMX":
            return fixed
        if mode == "Beat pulse":
            source = self._config.get(
                f"{prefix}_beat_source", "Volume beat"
            )
            if self._event_active(source, audio, now):
                return int(
                    self._config.get(
                        f"{prefix}_beat_value",
                        self._config.get(f"{prefix}_max", 255),
                    )
                )
            return open_value
        if mode == "Audio source":
            level = self._source_value(
                self._config.get(f"{prefix}_source", "Highs"),
                effect_hue,
                effect_brightness,
                audio,
                now,
            )
            threshold = float(
                self._config.get(f"{prefix}_threshold", 0.0)
            )
            if fixture_strobe and level < threshold:
                return open_value
            return int(
                round(
                    self._map_value(
                        level,
                        self._config.get(f"{prefix}_min", 0),
                        self._config.get(f"{prefix}_max", 255),
                    )
                )
            )
        return fixed

    def _build_shehds_fixture_data(self, data):
        rgb = np.asarray(data)
        if rgb.ndim == 1:
            rgb = rgb.reshape((-1, 3))
        else:
            rgb = rgb.reshape((-1, rgb.shape[-1]))

        if rgb.shape[1] < 3:
            raise ValueError(
                "SHEHDS Art-Net fixture profile requires RGB pixel data"
            )

        rgb = np.clip(rgb[:, :3], 0, 255).astype(np.uint8)
        if rgb.shape[0] < self.pixel_count:
            padded = np.zeros((self.pixel_count, 3), dtype=np.uint8)
            padded[: rgb.shape[0]] = rgb
            rgb = padded
        else:
            rgb = rgb[: self.pixel_count]

        devices_data = np.zeros(self.channel_count, dtype=np.uint8)
        audio = self._audio_snapshot()
        now = time.monotonic()

        all_brightness = np.max(rgb, axis=1)
        overall_brightness = float(
            np.max(all_brightness) if len(all_brightness) else 0
        )
        overall_rgb = np.mean(rgb, axis=0)
        maximum = float(np.max(overall_rgb))
        minimum = float(np.min(overall_rgb))
        if maximum <= 0:
            overall_hue = 0.0
        else:
            overall_hue, _, _ = colorsys.rgb_to_hsv(
                *(overall_rgb / 255.0)
            )

        self._advance_movement_phase(
            overall_hue,
            overall_brightness,
            audio,
            now,
        )

        for fixture_index, start in enumerate(self.fixture_addresses):
            red, green, blue = [int(value) for value in rgb[fixture_index]]
            maximum = max(red, green, blue)

            if maximum == 0:
                hue = 0.0
                saturation = 0.0
            else:
                hue, saturation, _ = colorsys.rgb_to_hsv(
                    red / 255.0,
                    green / 255.0,
                    blue / 255.0,
                )
            brightness = float(maximum)

            pan, tilt = self._movement_values(
                fixture_index,
                hue,
                brightness,
                audio,
                now,
            )
            pan_coarse, pan_fine = self._coarse_fine(pan)
            tilt_coarse, tilt_fine = self._coarse_fine(tilt)

            pt_speed = self._raw_control(
                self._config.get("pt_speed_mode", "Fixed DMX"),
                self._config.get("pt_speed_value", 128),
                self._config.get("pt_speed_source", "Lows"),
                self._config.get("pt_speed_min", 0),
                self._config.get("pt_speed_max", 255),
                hue,
                brightness,
                audio,
                now,
            )

            dimmer = self._dimmer_value(
                hue, brightness, audio, now
            )
            dimmer_coarse, dimmer_fine = self._coarse_fine(dimmer)

            color = self._color_value(
                fixture_index,
                hue,
                saturation,
                brightness,
                audio,
                now,
            )
            color_coarse, color_fine = self._coarse_fine(color)

            gobo1 = self._wheel_value(
                "gobo1",
                self.gobo1_cycle_values,
                fixture_index,
                hue,
                brightness,
                audio,
                now,
            )
            gobo2 = self._wheel_value(
                "gobo2",
                self.gobo2_cycle_values,
                fixture_index,
                hue,
                brightness,
                audio,
                now,
            )
            gobo2_rotation = self._raw_control(
                self._config.get(
                    "gobo2_rotation_mode", "Fixed DMX"
                ),
                self._config.get("gobo2_rotation_value", 0),
                self._config.get(
                    "gobo2_rotation_source", "Highs"
                ),
                self._config.get("gobo2_rotation_min", 0),
                self._config.get("gobo2_rotation_max", 255),
                hue,
                brightness,
                audio,
                now,
            )

            strobe = self._strobe_value(
                "strobe",
                hue,
                brightness,
                audio,
                now,
                fixture_strobe=True,
            )
            prism = self._binary_value(
                "prism", hue, brightness, audio, now
            )
            prism_rotation = self._raw_control(
                self._config.get(
                    "prism_rotation_mode", "Fixed DMX"
                ),
                self._config.get("prism_rotation_value", 191),
                self._config.get(
                    "prism_rotation_source", "Highs"
                ),
                self._config.get("prism_rotation_min", 128),
                self._config.get("prism_rotation_max", 255),
                hue,
                brightness,
                audio,
                now,
            )
            frost = self._binary_value(
                "frost", hue, brightness, audio, now
            )
            zoom = self._aux_value(
                "zoom", hue, brightness, audio, now
            )
            focus = self._aux_value(
                "focus", hue, brightness, audio, now
            )
            focus_coarse, focus_fine = self._coarse_fine(focus)

            ring_strobe = self._strobe_value(
                "ring_strobe",
                hue,
                brightness,
                audio,
                now,
                fixture_strobe=False,
            )
            ring_mode = self._raw_control(
                self._config.get(
                    "ring_mode_control", "Fixed DMX"
                ),
                self._config.get("ring_mode_value", 0),
                self._config.get(
                    "ring_mode_source", "Beat oscillator"
                ),
                self._config.get("ring_mode_min", 0),
                self._config.get("ring_mode_max", 255),
                hue,
                brightness,
                audio,
                now,
            )
            ring_speed = self._raw_control(
                self._config.get(
                    "ring_speed_mode", "Fixed DMX"
                ),
                self._config.get("ring_speed_value", 0),
                self._config.get("ring_speed_source", "Lows"),
                self._config.get("ring_speed_min", 0),
                self._config.get("ring_speed_max", 255),
                hue,
                brightness,
                audio,
                now,
            )

            fixture = np.array(
                [
                    pan_coarse,
                    pan_fine,
                    tilt_coarse,
                    tilt_fine,
                    pt_speed,
                    strobe,
                    dimmer_coarse,
                    dimmer_fine,
                    color_coarse,
                    color_fine,
                    gobo1,
                    gobo2,
                    gobo2_rotation,
                    prism,
                    prism_rotation,
                    frost,
                    int(round(zoom)),
                    focus_coarse,
                    focus_fine,
                    0,
                    ring_strobe,
                    ring_mode,
                    ring_speed,
                ],
                dtype=np.uint8,
            )
            devices_data[
                start : start + SHEHDS_CHANNEL_COUNT
            ] = fixture

        return devices_data

    def _build_pixel_data(self, data):
        data = self.output_mode.apply(data)
        data = data.flatten()[
            : self.data_max * self.output_mode.channels_per_pixel
        ]

        devices_data = np.empty(self.channel_count, dtype=np.uint8)
        reshaped_data = data.reshape(
            (
                self.num_devices,
                self.use_pixels_per_device
                * self.output_mode.channels_per_pixel,
            )
        )

        pre_amble_repeated = np.tile(
            self.pre_amble, (self.num_devices, 1)
        )
        post_amble_repeated = np.tile(
            self.post_amble, (self.num_devices, 1)
        )

        full_device_data = np.concatenate(
            (pre_amble_repeated, reshaped_data, post_amble_repeated),
            axis=1,
        )

        devices_data[0 : self.dmx_start_address] = 0
        devices_data[self.dmx_start_address :] = full_device_data.ravel()
        return devices_data

    def _send_dmx_data(self, devices_data):
        if self._artnet is None:
            return

        for i in range(self.universe_count):
            start = i * self.packet_size
            end = start + self.packet_size
            packet = np.zeros(self.packet_size, dtype=np.uint8)
            packet[
                : min(self.packet_size, self.channel_count - start)
            ] = devices_data[start:end]
            self._artnet.set_universe(i + self._config["universe"])
            self._artnet.set(packet)
            self._artnet.show()

    def flush(self, data):
        """Flush the data to all the Art-Net channels."""
        if self.init:
            self.do_once()

        if self.lock.acquire(blocking=False):
            try:
                if (
                    self.fixture_profile
                    == SHEHDS_GALAXYJET_300_3IN1_23CH
                ):
                    devices_data = self._build_shehds_fixture_data(data)
                else:
                    devices_data = self._build_pixel_data(data)

                self._send_dmx_data(devices_data)
            finally:
                self.lock.release()
        else:
            _LOGGER.error(
                "Panic could not get lock %s", self.config["name"]
            )
