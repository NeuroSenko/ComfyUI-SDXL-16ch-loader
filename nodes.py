from comfy import latent_formats
import folder_paths
import torch

import comfy.sd

def apply_16ch_mod(model):
  m = model.clone()

  m.add_object_patch('concat_keys', ())

  flux_fmt = latent_formats.Flux()

  # add directly to model and as model patch for patch_model()
  m.model.latent_format = flux_fmt
  m.add_object_patch('latent_format', flux_fmt)

  return m


class Sdxl16ChLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        model, clip, vae, clip_vision = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        m = apply_16ch_mod(model)

        return [m, clip, vae]


class Sdxl16ChLoaderWithPrecision:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
                "model_precision": (["auto", "fp32", "fp16", "bf16"], {"default": "auto", "tooltip": "Precision for the diffusion model (UNet)."}),
                "clip_precision": (["auto", "fp32", "fp16", "bf16"], {"default": "auto", "tooltip": "Precision for the CLIP text encoder."}),
                "vae_precision": (["auto", "fp32", "fp16", "bf16"], {"default": "auto", "tooltip": "Precision for the VAE encoder/decoder."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint with individual precision control for MODEL, CLIP, and VAE components."

    def _get_dtype(self, precision):
        """Convert precision string to torch dtype"""
        if precision == "fp32":
            return torch.float32
        elif precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        else:  # auto
            return None

    def load_checkpoint(self, ckpt_name, model_precision="auto", clip_precision="auto", vae_precision="auto"):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)

        # Prepare model options with dtype
        model_options = {}
        te_model_options = {}

        model_dtype = self._get_dtype(model_precision)
        if model_dtype is not None:
            model_options["dtype"] = model_dtype

        clip_dtype = self._get_dtype(clip_precision)
        if clip_dtype is not None:
            te_model_options["dtype"] = clip_dtype

        # Load checkpoint with precision options
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=model_options,
            te_model_options=te_model_options
        )

        model, clip, vae = out[:3]

        # Apply precision to CLIP - convert after loading
        if clip_precision != "auto":
            clip_dtype = self._get_dtype(clip_precision)
            if clip_dtype is not None:
                clip.cond_stage_model.to(clip_dtype)

        # Apply precision to VAE - reload with correct dtype
        if vae_precision != "auto":
            vae_dtype = self._get_dtype(vae_precision)
            if vae_dtype is not None:
                # Reload VAE with proper dtype
                vae_sd = vae.get_sd()
                vae = comfy.sd.VAE(sd=vae_sd, dtype=vae_dtype)

        # Apply 16ch modifications
        m = apply_16ch_mod(model)

        return [m, clip, vae]


class TensorDtypeInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "model": ("MODEL", {"tooltip": "MODEL to check dtype"}),
                "clip": ("CLIP", {"tooltip": "CLIP to check dtype"}),
                "vae": ("VAE", {"tooltip": "VAE to check dtype"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("The dtype of the input (e.g., fp32, fp16, bf16)",)
    FUNCTION = "get_dtype"
    CATEGORY = "utils"
    DESCRIPTION = "Returns the dtype of MODEL, CLIP, or VAE."

    def get_dtype(self, model=None, clip=None, vae=None):
        import torch

        dtype = None

        # Check which input was provided
        if model is not None:
            # For MODEL objects
            dtype = next(model.model.parameters()).dtype
        elif clip is not None:
            # For CLIP objects
            dtype = next(clip.cond_stage_model.parameters()).dtype
        elif vae is not None:
            # For VAE objects
            dtype = next(vae.first_stage_model.parameters()).dtype
        else:
            return ("no input provided",)

        # Convert torch dtype to string
        dtype_map = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
            torch.float64: "fp64",
            torch.int8: "int8",
            torch.int16: "int16",
            torch.int32: "int32",
            torch.int64: "int64",
            torch.uint8: "uint8",
        }

        dtype_str = dtype_map.get(dtype, str(dtype))
        return (dtype_str,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SDXL 16ch Loader": Sdxl16ChLoader,
    "SDXL 16ch Loader (Precision)": Sdxl16ChLoaderWithPrecision,
    "Tensor Dtype Info": TensorDtypeInfo,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXL16ChLoader": "SDXL 16ch Loader",
    "SDXL16ChLoaderWithPrecision": "SDXL 16ch Loader (Precision)",
    "TensorDtypeInfo": "Tensor Dtype Info",
}
