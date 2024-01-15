from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_kandinsky_2_controlnet_depth.infer_kandinsky_2_controlnet_depth_process import InferKandinsky2ControlnetDepthParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferKandinsky2ControlnetDepthWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferKandinsky2ControlnetDepthParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)

        # Negative prompt
        self.edit_negative_prompt = pyqtutils.append_edit(
                                                    self.grid_layout,
                                                    "Negative prompt",
                                                    self.parameters.negative_prompt
                                                    )

        # Number of inference steps - prior
        self.spin_prior_number_of_steps = pyqtutils.append_spin(
                                                    self.grid_layout,
                                                    "Prior number of steps",
                                                    self.parameters.prior_num_inference_steps,
                                                    min=1, step=1
                                                    )

        # Guidance scale
        self.spin_prior_guidance_scale = pyqtutils.append_double_spin(
                                                        self.grid_layout,
                                                        "Prior guidance scale",
                                                        self.parameters.prior_guidance_scale,
                                                        min=0, step=0.1, decimals=1
                                                    )

        # Number of inference steps
        self.spin_number_of_steps = pyqtutils.append_spin(
                                                    self.grid_layout,
                                                    "Number of steps",
                                                    self.parameters.num_inference_steps,
                                                    min=1, step=1
                                                    )
        # Guidance scale
        self.spin_guidance_scale = pyqtutils.append_double_spin(
                                                        self.grid_layout,
                                                        "Guidance scale",
                                                        self.parameters.guidance_scale,
                                                        min=0, step=0.1, decimals=1
                                                    )

        # Image height
        self.spin_height = pyqtutils.append_spin(
                                                self.grid_layout,
                                                "Image height",
                                                self.parameters.height,
                                                min=128, step=1
                                                )

         # width
        self.spin_width = pyqtutils.append_spin(
                                                self.grid_layout,
                                                "Image width",
                                                self.parameters.width,
                                                min=128, step=1
                                                )

        # Seed
        self.spin_seed = pyqtutils.append_spin(
                                            self.grid_layout,
                                            "Seed",
                                            self.parameters.seed,
                                            min=-1, step=1
                                            )

       # Set widget layout
        self.set_layout(layout_ptr)


    def on_apply(self):
        # Apply button clicked slot
        self.emit_apply(self.parameters)
        self.parameters.update = True
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.num_inference_steps = self.spin_number_of_steps.value()
        self.parameters.prior_num_inference_steps = self.spin_prior_number_of_steps.value()
        self.parameters.guidance_scale = self.spin_guidance_scale.value()
        self.parameters.prior_guidance_scale = self.spin_prior_guidance_scale.value()
        self.parameters.width = self.spin_width.value()
        self.parameters.height = self.spin_height.value()
        self.parameters.seed = self.spin_seed.value()
        self.parameters.negative_prompt = self.edit_negative_prompt.text()


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferKandinsky2ControlnetDepthWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_kandinsky_2_controlnet_depth"

    def create(self, param):
        # Create widget object
        return InferKandinsky2ControlnetDepthWidget(param, None)
