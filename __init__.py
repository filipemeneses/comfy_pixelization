import asyncio
import os
import sys

import comfy.utils
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

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


device = get_optimal_device()


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


def to_image(tensor, pixel_size, upscale_after):
    img = tensor.data[0].cpu().float().numpy()
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((img.size[0] // 4, img.size[1] // 4), resample=Image.Resampling.NEAREST)
    if upscale_after:
        img = img.resize((img.size[0] * pixel_size, img.size[1] * pixel_size), resample=Image.Resampling.NEAREST)

    return img


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
        if not hasattr(self, "model"):
            self.model = Model()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {"default": 4, "min": 1, "max": 32}),
                "upscale_after": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "pixelize"

    CATEGORY = "image"

    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = False

    async def run_pixelatization(self, image, pixel_size, upscale_after):
        image = image.resize((image.width * 4 // pixel_size, image.height * 4 // pixel_size))

        with torch.no_grad():
            in_t = rescale_image(image).to(device)

            code = torch.asarray(MLP_code, device=device).reshape((1, 256, 1, 1))
            adain_params = self.model.G_A_net.module.MLP(code)

            feature = self.model.G_A_net.module.RGBEnc(in_t)
            images = self.model.G_A_net.module.RGBDec(feature, adain_params)
            out_t = self.model.alias_net(images)

            image = to_image(out_t, pixel_size=pixel_size, upscale_after=upscale_after)

        image = pil2tensor(image)

        return image

    def pixelize(self, image, pixel_size, upscale_after):
        self.model.to(device)

        tensor = image * 255
        tensor = np.array(tensor, dtype=np.uint8)

        progressbar = comfy.utils.ProgressBar(tensor.shape[0])
        all_images = []
        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
            all_images.append(wait_for_async(lambda: self.run_pixelatization(image, pixel_size, upscale_after)))
            progressbar.update(1)

        return (all_images,)


NODE_CLASS_MAPPINGS = {"Pixelization": Pixelization}
