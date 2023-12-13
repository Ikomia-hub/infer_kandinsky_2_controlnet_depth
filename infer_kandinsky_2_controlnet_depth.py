from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_kandinsky_2_controlnet_depth.infer_kandinsky_2_controlnet_depth_process import InferKandinsky2ControlnetDepthFactory
        return InferKandinsky2ControlnetDepthFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_kandinsky_2_controlnet_depth.infer_kandinsky_2_controlnet_depth_widget import InferKandinsky2ControlnetDepthWidgetFactory
        return InferKandinsky2ControlnetDepthWidgetFactory()
