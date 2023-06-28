import argparse
import datetime
import json
import os
import sys
import time

from PIL import Image
import gradio as gr
from . import errors,shared_items,devices


demo = None

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default=os.path.dirname(os.path.dirname(os.path.realpath(__file__))), help="base path where all user data is stored",)
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--medvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a little speed for low VRM usage")
parser.add_argument("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage")
parser.add_argument("--lowram", action='store_true', help="load stable diffusion checkpoint weights to VRAM instead of RAM")

if os.environ.get('IGNORE_CMD_ARGS_ERRORS', None) is None:
    cmd_opts = parser.parse_args()
else:
    cmd_opts, _ = parser.parse_known_args()

cmd_opts.disable_safe_unpickle = False
cmd_opts.precision = "full"
cmd_opts.use_cpu = "" ## all?
cmd_opts.device_id = None

sd_upscalers = []

restricted_opts = {
    "samples_filename_pattern",
}

devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
    (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

device = devices.device
weight_load_location = None if cmd_opts.lowram else "cpu"

#batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram
xformers_available = False





class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.refresh = refresh


def options_section(section_identifier, options_dict):
    for k, v in options_dict.items():
        v.section = section_identifier

    return options_dict


options_templates = {}
options_templates.update(options_section(('upscaling', "Upscaling"), {
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for ESRGAN upscalers. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI. (Requires restart)", gr.CheckboxGroup, lambda: {"choices": shared_items.realesrgan_models_names()}),
    "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
}))


options_templates.update(options_section(('interrogate', "Interrogate Options"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "Interrogate: keep models in VRAM"),
    "interrogate_return_ranks": OptionInfo(False, "Interrogate: include ranks of model tags matches in results (Has no effect on caption-based interrogators)."),
    "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(24, "Interrogate: minimum description length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(48, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP: maximum number of lines in text file (0 = No limit)"),
#   "interrogate_clip_skip_categories": OptionInfo([], "CLIP: skip inquire categories", gr.CheckboxGroup, lambda: {"choices": modules.interrogate.category_types()}, refresh=modules.interrogate.category_types),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "Interrogate: deepbooru score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(True, "Interrogate: deepbooru sort alphabetically"),
    "deepbooru_use_spaces": OptionInfo(False, "use spaces for tags in deepbooru"),
    "deepbooru_escape": OptionInfo(True, "escape (\\) brackets in deepbooru (so they are used as literal brackets and not for emphasis)"),
    "deepbooru_filter_tags": OptionInfo("", "filter out those tags from deepbooru output (separated by comma)"),
}))

class Options:
    data = None
    data_labels = options_templates
    typemap = {int: float}

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value):
        if self.data is not None:
            if key in self.data or key in self.data_labels:
                assert not cmd_opts.freeze_settings, "changing settings is disabled"

                info = opts.data_labels.get(key, None)
                comp_args = info.component_args if info else None
                if isinstance(comp_args, dict) and comp_args.get('visible', True) is False:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                if cmd_opts.hide_ui_dir_config and key in restricted_opts:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                self.data[key] = value
                return

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def set(self, key, value):
        """sets an option and calls its onchange callback, returning True if the option changed and False otherwise"""

        oldval = self.data.get(key, None)
        if oldval == value:
            return False

        try:
            setattr(self, key, value)
        except RuntimeError:
            return False

        if self.data_labels[key].onchange is not None:
            try:
                self.data_labels[key].onchange()
            except Exception as e:
                errors.display(e, f"changing setting {key} to {value}")
                setattr(self, key, oldval)
                return False

        return True

    def get_default(self, key):
        """returns the default value for the key"""

        data_label = self.data_labels.get(key)
        if data_label is None:
            return None

        return data_label.default

    def save(self, filename):
        assert not cmd_opts.freeze_settings, "saving settings is disabled"

        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4)

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)

        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func

        if call:
            func()

    def dumpjson(self):
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        return json.dumps(d)

    def add_option(self, key, info):
        self.data_labels[key] = info

    def reorder(self):
        """reorder settings so that all items related to section always go together"""

        section_ids = {}
        settings_items = self.data_labels.items()
        for k, item in settings_items:
            if item.section not in section_ids:
                section_ids[item.section] = len(section_ids)

        self.data_labels = {k: v for k, v in sorted(settings_items, key=lambda x: section_ids[x[1].section])}

    def cast_value(self, key, value):
        """casts an arbitrary to the same type as this setting's value with key
        Example: cast_value("eta_noise_seed_delta", "12") -> returns 12 (an int rather than str)
        """

        if value is None:
            return None

        default_value = self.data_labels[key].default
        if default_value is None:
            default_value = getattr(self, key, None)
        if default_value is None:
            return None

        expected_type = type(default_value)
        if expected_type == bool and value == "False":
            value = False
        else:
            value = expected_type(value)

        return value
opts = Options()