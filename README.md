
# ComfyUI_pixelization

ComfyUI node that pixelizes images

# Workflow preview
![](preview.png)


# Installation

1. Clone repository to `ComfyUI/custom_nodes/ComfyUI_pixelization`
2. Go to folder
3. Run `python ./install.py` 
4. Download checkpoints to `ComfyUI/custom_nodes/ComfyUI_pixelization/checkpoints`
5. Use node `Pixelization > Pixelization` to generate pixelated image

## Models

Download all three models from the table and place them into the `checkpoints` directory inside the extension at `ComfyUI/custom_nodes/ComfyUI_pixelization/checkpoints`.

| url                                                                                | filename           |
| ---------------------------------------------------------------------------------- | ------------------ |
| https://drive.google.com/file/d/1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM/view?usp=sharing | pixelart_vgg19.pth |
| https://drive.google.com/file/d/17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_/view?usp=sharing | alias_net.pth      |
| https://drive.google.com/file/d/1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az/view?usp=sharing | 160_net_G_A.pth    |


# Credits

* AUTOMATIC1111 extension: https://github.com/AUTOMATIC1111/stable-diffusion-webui-pixelization
* Original repo: https://github.com/WuZongWei6/Pixelization
* Code I used for reference: https://github.com/arenatemp/pixelization_inference
