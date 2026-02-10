This node lets you use a 16-channel VAE for modified versions of SDXL-checkpoints which were designed to work with such kind of VAE.

![image](https://github.com/user-attachments/assets/63de6dbe-9646-4a24-b0f6-8f8d7c1feb74)

## CLIP

These models require **both** CLIP encoders simultaneously — CLIP-L (ViT-L/14) and CLIP-G (ViT-bigG/14). Using only one of them will result in degraded quality and poor prompt following.

For GGUF loading use ComfyUI's built-in **DualCLIPLoader** node (type: `sdxl`) to load `clip_l.safetensors` + `clip_g.safetensors` separately, then connect the output to the `clip` input of the GGUF loader node.

## VAE

Use **`ae.safetensors`** (Flux AE) — it is the correct VAE for 16-channel latent space models. The standard SDXL VAE (`sdxl_vae.safetensors`) operates on 4 channels and is **not compatible**.

## GGUF

Requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) extension. Place GGUF files into the `models/unet_gguf` directory.

The GGUF loader outputs only the **MODEL**. Connect CLIP and VAE separately using ComfyUI's built-in **DualCLIPLoader** and **VAELoader** nodes.