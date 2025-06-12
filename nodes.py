from __future__ import annotations

import asyncio
import colorsys
import os
import sys
from dataclasses import dataclass

import comfy.utils
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# The submodule imports relative modules, so we need to ensure the Pixelization directory is in the path.
if "Pixelization" not in sys.modules:
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/Pixelization")
from .Pixelization.models import c2pGen
from .Pixelization.models.networks import define_G
from .Pixelization.test_pro import MLP_code


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    return torch.backends.mps.is_available()


def get_cuda_device_string():
    return "cuda"


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


basedir = os.path.dirname(os.path.realpath(__file__))
path_checkpoints = os.path.join(basedir, "checkpoints")
path_pixelart_vgg19 = os.path.join(path_checkpoints, "pixelart_vgg19.pth")
path_160_net_G_A = os.path.join(path_checkpoints, "160_net_G_A.pth")
path_alias_net = os.path.join(path_checkpoints, "alias_net.pth")


class TorchHijackForC2pGen:
    def __getattr__(self, item):
        if item == "load":
            return self.load

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def load(self, filename, *args, **kwargs):
        if filename == "./pixelart_vgg19.pth":
            filename = path_pixelart_vgg19

        return torch.load(filename, *args, **kwargs)


c2pGen.torch = TorchHijackForC2pGen()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        os.makedirs(path_checkpoints, exist_ok=True)

        models_missing = False

        if not os.path.exists(path_pixelart_vgg19):
            print(
                f"Missing {path_pixelart_vgg19} - download it from https://drive.google.com/uc?id=1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM"
            )
            models_missing = True

        if not os.path.exists(path_160_net_G_A):
            print(
                f"Missing {path_160_net_G_A} - download it from https://drive.google.com/uc?id=1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az"
            )
            models_missing = True

        if not os.path.exists(path_alias_net):
            print(
                f"Missing {path_alias_net} - download it from https://drive.google.com/uc?id=17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_"
            )
            models_missing = True

        if models_missing:
            error_message = "Missing checkpoints for pixelization - see console for download links."
            print(error_message)
            raise RuntimeError(error_message)

        with torch.no_grad():
            self.G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [0])
            self.alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [0])

            G_A_state = torch.load(path_160_net_G_A)
            for p in list(G_A_state.keys()):
                G_A_state["module." + str(p)] = G_A_state.pop(p)
            self.G_A_net.load_state_dict(G_A_state)

            alias_state = torch.load(path_alias_net)
            for p in list(alias_state.keys()):
                alias_state["module." + str(p)] = alias_state.pop(p)
            self.alias_net.load_state_dict(alias_state)


def rescale_image(img):
    """
    Preprocess the image for pixelization.

    Crops the image to a size that is divisible by 4.
    """
    orig_width, orig_height = img.size

    new_width = int(round(orig_width / 4) * 4)
    new_height = int(round(orig_height / 4) * 4)

    left = (orig_width - new_width) // 2
    top = (orig_height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    img = img.crop((left, top, right, bottom))

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return trans(img)[None, :, :, :]


@dataclass
class PixelizationOptions:
    pixel_size: int = 4  # Size of pixelation
    upscale_after: bool = True  # Upscale the pixelized image after processing
    original_img: Image.Image | None = None  # Original image for color copying
    copy_hue: bool = False  # Copy hue from the original image
    copy_sat: bool = False  # Copy saturation from the original image
    copy_val: bool = False  # Copy value (brightness) from the original image
    scale_value: float = 1.0  # Scale value for the pixelization (not used in this implementation)
    restore_dark: int = 15  # Restore dark pixels
    restore_bright: int = 1  # Restore bright pixels


def to_image(tensor, options: PixelizationOptions):
    img = tensor.data[0].cpu().float().numpy()
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)

    width = img.size[0] // 4
    height = img.size[1] // 4
    img = img.resize((width, height), resample=Image.Resampling.NEAREST)

    if options.original_img and (options.copy_hue or options.copy_sat):
        original_img = options.original_img.resize((width, height), resample=Image.Resampling.NEAREST)
        img = color_image(img, original_img, options)

    if options.upscale_after:
        img = img.resize(
            (
                img.size[0] * options.pixel_size,
                img.size[1] * options.pixel_size,
            ),
            resample=Image.Resampling.NEAREST,
        )

    return img


def color_image(img, original_img, options: PixelizationOptions):
    """
    Color the pixelized image based on the original image's hue and saturation.
    """
    img = img.convert("RGB")
    original_img = original_img.convert("RGB")

    colored_img = Image.new("RGB", img.size)

    print(img.width, img.height)

    for x in range(img.width):
        for y in range(img.height):
            pixel = original_img.getpixel((x, y))
            r, g, b = pixel
            original_h, original_s, original_v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

            pixel = img.getpixel((x, y))
            r, g, b = pixel
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

            if options.copy_val:
                if v < 0.5:
                    v = v * (100 - options.restore_dark) / 100 + original_v * options.restore_dark / 100
                else:
                    v = v * (100 - options.restore_bright) / 100 + original_v * options.restore_bright / 100

            r, g, b = colorsys.hsv_to_rgb(
                original_h if options.copy_hue else h,
                original_s if options.copy_sat else s,
                v,
            )
            colored_img.putpixel((x, y), (int(r * 255), int(g * 255), int(b * 255)))

    return colored_img


def tensor2pil(image):
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def wait_for_async(async_fn, loop=None):
    res = []

    async def run_async():
        r = await async_fn()
        res.append(r)

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    loop.run_until_complete(run_async())

    return res[0]


class Pixelization:
    def __init__(self):
        self.model = Model()
        self.device = get_optimal_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {"default": 4, "min": 1, "max": 32}),
                "upscale_after": ("BOOLEAN", {"default": True}),
                "copy_hue": ("BOOLEAN", {"default": False}),
                "copy_sat": ("BOOLEAN", {"default": False}),
                "copy_val": ("BOOLEAN", {"default": False}),
                "restore_dark": ("INT", {"default": 15, "min": 0, "max": 100}),
                "restore_bright": ("INT", {"default": 1, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "pixelize"

    # This folds the node into the same category as comfyui-post-processing-nodes.
    # Seems like the best place for it, pixelization is also post-processing.
    CATEGORY = "postprocessing/Effects"

    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = False

    async def run_pixelization(self, image, options):
        image = image.resize((image.width * 4 // options.pixel_size, image.height * 4 // options.pixel_size))

        with torch.no_grad():
            in_t = rescale_image(image).to(self.device)

            code = torch.asarray(MLP_code, device=self.device).reshape((1, 256, 1, 1))
            adain_params = self.model.G_A_net.module.MLP(code)

            feature = self.model.G_A_net.module.RGBEnc(in_t)
            images = self.model.G_A_net.module.RGBDec(feature, adain_params)
            out_t = self.model.alias_net(images)

            image = to_image(out_t, options)

        image = pil2tensor(image)

        return image

    def pixelize(
        self,
        image,
        pixel_size,
        upscale_after,
        copy_hue,
        copy_sat,
        copy_val,
        restore_dark,
        restore_bright,
    ):
        self.model.to(self.device)

        tensor = image * 255
        tensor = np.array(tensor, dtype=np.uint8)

        progressbar = comfy.utils.ProgressBar(tensor.shape[0])
        all_images = []
        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])

            pixelize_options = PixelizationOptions(
                pixel_size=pixel_size,
                upscale_after=upscale_after,
                original_img=image,
                copy_hue=copy_hue,
                copy_sat=copy_sat,
                copy_val=copy_val,
                restore_dark=restore_dark,
                restore_bright=restore_bright,
            )

            all_images.append(wait_for_async(lambda: self.run_pixelization(image, pixelize_options)))
            progressbar.update(1)

        return (all_images,)
