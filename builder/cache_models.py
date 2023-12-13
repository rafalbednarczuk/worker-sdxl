import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def fetch_pretrained_model_local(model_class, model_path, **kwargs):
    '''
    Fetches a pretrained model from path
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_single_file(model_path, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines
    '''
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }

    pipe = fetch_pretrained_model_local(StableDiffusionXLPipeline,
                                        "/base_model.safetensors", **common_args)
    refiner = fetch_pretrained_model(StableDiffusionXLImg2ImgPipeline,
                                     "stabilityai/stable-diffusion-xl-refiner-1.0", **common_args)

    return pipe, refiner


if __name__ == "__main__":
    get_diffusion_pipelines()
