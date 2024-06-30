from typing import List, Optional, Tuple, Union
import torch
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

class DDPMPipelinenew(DiffusionPipeline):
    def __init__(self, unet, scheduler, num_classes: int):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.num_classes = num_classes

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 64,
        class_labels: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        # Ensure class_labels is on the same device as the model
        class_labels = class_labels.to(self.device)
        if class_labels.ndim == 0:
            class_labels = class_labels.unsqueeze(0).expand(batch_size)
        else:
            class_labels = class_labels.expand(batch_size)

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # Ensure the class labels are correctly broadcast to match the input tensor shape
            model_output = self.unet(image, t, class_labels).sample

            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
    
