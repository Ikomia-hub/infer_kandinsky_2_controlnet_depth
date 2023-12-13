<div align="center">
  <img src="images/einstein.jpg" alt="Algorithm icon">
  <h1 align="center">infer_kandinsky_2_controlnet_depth</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_kandinsky_2_controlnet_depth">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_kandinsky_2_controlnet_depth">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_kandinsky_2_controlnet_depth/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_kandinsky_2_controlnet_depth.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Kandinsky 2.2 is a text-conditional diffusion model based on unCLIP and latent diffusion, composed of a transformer-based image prior model, a unet diffusion model, and a decoder.

The addition of the ControlNet depth mechanism allows the model to effectively control the process of generating images. This leads to more accurate and visually appealing outputs and opens new possibilities for text-guided image manipulation.


*Note: This algorithm requires 10GB GPU RAM*

- Input

![controlnet depth input](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png)

- Output

![controlnet depth output](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/robot_cat_text2img.png)


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name = "infer_kandinsky_2_controlnet_depth", auto_connect=True)

# Run
wf.run_on(url="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png")

# Display the image
display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **prompt** (str) - default 'A robot, 4k photo' : Text prompt to guide the image generation .
- **negative_prompt** (str, *optional*) - default 'lowres, text, error, cropped, worst quality, low quality, ugly': The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
- **prior_num_inference_steps** (int) - default '25': Number of denoising steps of the prior model (CLIP).
- **prior_guidance_scale** (float) - default '4.0':  Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality. (minimum: 1; maximum: 20).
- **num_inference_steps** (int) - default '100': The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
- **guidance_scale** (float) - default '4.0':  Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality. (minimum: 1; maximum: 20).
- **height** (int) - default '768: The height in pixels of the generated image.
- **width** (int) - default '768: The width in pixels of the generated image.
- **seed** (int) - default '-1': Seed value. '-1' generates a random number between 0 and 191965535.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name = "infer_kandinsky_2_controlnet_depth", auto_connect=True)

algo.set_parameters({
    'prompt': 'A robot, 4k photo',
    'negative_prompt': 'lowres, text, error, cropped, worst quality, low quality, ugly',
    'prior_num_inference_steps': '25',
    'prior_guidance_scale': '4.0',
    'num_inference_steps': '100',
    'guidance_scale': '4.0',
    'seed': '-1',
    'width': '768',
    'height': '768',
    })

# Run
wf.run_on(url="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png")

# Display the image
display(algo.get_output(0).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name = "infer_kandinsky_2_controlnet_depth", auto_connect=True)

# Run
wf.run_on(url="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

