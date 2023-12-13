import copy
from ikomia import core, dataprocess
from diffusers import KandinskyV22PriorPipeline, KandinskyV22ControlnetPipeline
from transformers import pipeline, DPTForDepthEstimation, DPTImageProcessor
import torch
import numpy as np
import random
import os
from PIL import Image

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferKandinsky2ControlnetDepthParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.prompt = "A robot, 4k photo"
        self.prior_guidance_scale = 4.0
        self.guidance_scale = 1.0
        self.negative_prompt = "lowres, text, error, cropped, worst quality, low quality, ugly"
        self.height = 768
        self.width = 768
        self.prior_num_inference_steps = 25
        self.num_inference_steps = 10
        self.seed = -1
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.prompt = str(param_map["prompt"])
        self.guidance_scale = float(param_map["guidance_scale"])
        self.prior_guidance_scale = float(param_map["prior_guidance_scale"])
        self.negative_prompt = str(param_map["negative_prompt"])
        self.seed = int(param_map["seed"])
        self.height = int(param_map["height"])
        self.width = int(param_map["width"])
        self.num_inference_steps = int(param_map["num_inference_steps"])
        self.prior_num_inference_steps = int(param_map["prior_num_inference_steps"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["prompt"] = str(self.prompt)
        param_map["negative_prompt"] = str(self.negative_prompt)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["prior_guidance_scale"] = str(self.prior_guidance_scale)
        param_map["height"] = str(self.height)
        param_map["width"] = str(self.width)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["prior_num_inference_steps"] = str(self.prior_num_inference_steps)
        param_map["seed"] = str(self.seed)

        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferKandinsky2ControlnetDepth(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.add_input(dataprocess.CImageIO())
        self.add_output(dataprocess.CImageIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferKandinsky2ControlnetDepthParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.pipe_controlnet = None
        self.pipe_prior = None
        self.generator = None
        self.seed = None
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
        self.depth_model_name = "Intel/dpt-large"
        self.controlnet_model_name = "kandinsky-community/kandinsky-2-2-controlnet-depth"
        self.prior_model_name = "kandinsky-community/kandinsky-2-2-prior"
        self.depth_estimator = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def generate_seed(self, seed):
        if seed == -1:
            self.seed = random.randint(0, 191965535)
        else:
            self.seed = seed
        self.generator = torch.Generator(self.device).manual_seed(self.seed)

    def make_hint(self, image, depth_estimator):
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        detected_map = torch.from_numpy(image).float() / 255.0
        hint = detected_map.permute(2, 0, 1)
        return hint

    def load_pipeline(self, local_files_only=False, model_type=None):
        torch_tensor_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        if model_type == 'depth':
            self.image_processor = DPTImageProcessor.from_pretrained(
                self.depth_model_name,
                cache_dir=self.model_folder,
                local_files_only=local_files_only
                )
            pipeline = DPTForDepthEstimation.from_pretrained(
                self.depth_model_name,
                cache_dir=self.model_folder,
                local_files_only=local_files_only
                )

        if model_type == 'prior':
            pipeline = KandinskyV22PriorPipeline.from_pretrained(
                self.prior_model_name,
                torch_dtype=torch_tensor_dtype,
                cache_dir=self.model_folder,
                local_files_only=local_files_only
                )
            pipeline.to(self.device)

        if model_type == 'controlnet':
            pipeline = KandinskyV22ControlnetPipeline.from_pretrained(
                self.controlnet_model_name,
                torch_dtype=torch_tensor_dtype,
                cache_dir=self.model_folder,
                local_files_only=local_files_only
                )

            pipeline.to(self.device)

        return pipeline

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Get input
        src_img = self.get_input(0).get_image()

        img = Image.fromarray(src_img).resize((param.width, param.height))
        if param.update or self.depth_estimator is None:
            self.generate_seed(param.seed)
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            # We can use the `depth-estimation` pipeline from transformers to process the image and retrieve its depth map.
            try:
                model_depth_estimator = self.load_pipeline(local_files_only=True, model_type='depth')
            except Exception as e:
                model_depth_estimator = self.load_pipeline(local_files_only=False, model_type='depth')
            with torch.no_grad():
                self.depth_estimator = pipeline(
                                        "depth-estimation",
                                        model=model_depth_estimator,
                                        image_processor=self.image_processor
                )

        hint = self.make_hint(img, self.depth_estimator).unsqueeze(0).half().to(self.device)

        if self.pipe_prior is None:
            # Now, we load the prior pipeline and the text-to-image controlnet pipeline
            try:
                self.pipe_prior = self.load_pipeline(local_files_only=True, model_type='prior')
            except Exception as e:
                self.pipe_prior = self.load_pipeline(local_files_only=False, model_type='prior')

        if self.pipe_controlnet is None:
            try:
                self.pipe_controlnet = self.load_pipeline(local_files_only=True, model_type='controlnet')
            except Exception as e:
                self.pipe_controlnet = self.load_pipeline(local_files_only=False, model_type='controlnet')

        # We pass the prompt and negative prompt through the prior to generate image embeddings
        with torch.no_grad():
            image_emb, zero_image_emb = self.pipe_prior(
                                                prompt=param.prompt,
                                                negative_prompt=param.negative_prompt,
                                                num_inference_steps=param.prior_num_inference_steps,
                                                guidance_scale= param.prior_guidance_scale,
                                                generator=self.generator
            ).to_tuple()

        # Now we can pass the image embeddings and the depth image we extracted to the controlnet pipeline.
        # With Kandinsky 2.2, only prior pipelines accept `prompt` input.
        # You do not need to pass the prompt to the controlnet pipeline.
        with torch.no_grad():
            result = self.pipe_controlnet(
                image_embeds=image_emb,
                negative_image_embeds=zero_image_emb,
                hint=hint,
                num_inference_steps=param.num_inference_steps,
                generator=self.generator,
                height=param.height,
                width=param.width,
                guidance_scale= param.guidance_scale
            ).images[0]

        # Get and display output
        image = np.array(result)
        output_img = self.get_output(0)
        output_img.set_image(image)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferKandinsky2ControlnetDepthFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_kandinsky_2_controlnet_depth"
        self.info.short_description = "Kandinsky 2.2 controlnet depth diffusion model."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/einstein.jpg"
        self.info.authors = "A. Shakhmatov, A. Razzhigaev, A. Nikolich, V. Arkhipkin, I. Pavlov, A. Kuznetsov, D. Dimitrov"
        self.info.article = "https://aclanthology.org/2023.emnlp-demo.25/"
        self.info.journal = "ACL Anthology"
        self.info.year = 2023
        self.info.license = "Apache 2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://huggingface.co/kandinsky-community/kandinsky-2-2-controlnet-depth"
        # Code source repository
        self.info.repository = "https://github.com/ai-forever/Kandinsky-2"
        # Keywords used for search
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "IMAGE_GENERATION"
        self.info.keywords = "Latent Diffusion,Hugging Face,Kandinsky,Image mixing,Interpolation,Generative"


    def create(self, param=None):
        # Create algorithm object
        return InferKandinsky2ControlnetDepth(self.info.name, param)
