from __future__ import annotations
import os
import re
import json
import math
import types
import time
import urllib.request
import urllib.parse
import urllib.error
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,expandable_segments:True")
import gc
import random
import threading
import traceback
import subprocess
import shutil
import sys
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
CHECKPOINTS = MODELS / "Stable-diffusion"
CONTROLNET = MODELS / "controlnet"
MOTION = MODELS / "animatediff"
INSIGHT = MODELS / "insightface"
VOICE_MODELS = MODELS / "voice"
LORA_DIR = MODELS / "loras"
VAE_DIR = MODELS / "vae"
EMBEDDINGS_DIR = MODELS / "embeddings"
IP_ADAPTER_DIR = MODELS / "ip_adapter"
CHARACTERS_DIR = ROOT / "characters"
SCENES_DIR = ROOT / "scenes"
TOOLS_DIR = ROOT / "tools"
OUTPUT = ROOT / "output_media"
OUTPUT.mkdir(parents=True, exist_ok=True)

for p in (CHECKPOINTS, CONTROLNET, MOTION, INSIGHT, VOICE_MODELS,
          LORA_DIR, VAE_DIR, EMBEDDINGS_DIR, IP_ADAPTER_DIR,
          CHARACTERS_DIR, SCENES_DIR, TOOLS_DIR):
    p.mkdir(parents=True, exist_ok=True)

try:
    import torch
    from PIL import Image, ImageDraw, ImageFilter, ImageTk
    import numpy as np
except Exception as e:
    raise SystemExit(
        "Не удалось импортировать базовые библиотеки.\n"
        "Установи зависимости из инструкции и запусти снова.\n\n" + str(e)
    )

try:
    import cv2
except Exception:
    cv2 = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

PIPE = None
PIPE_KEY = None
BUSY = False
FACE_ANALYSER = None
FACE_SWAPPER = None
FFMPEG_EXE = None
PREFERRED_CHECKPOINT = "realisticvisionv60b1_v51hypervae_pruned_fp16"


def set_progress(value, message=None):
    root_obj = globals().get("root")
    bar = globals().get("progress_widget")
    if root_obj is not None and bar is not None:
        try:
            root_obj.after(0, lambda v=max(0.0, min(1.0, float(value))): bar.configure(value=v*100))
        except Exception:
            pass
    if message:
        log(message)


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)
    ui_root = globals().get("root")
    ui_status = globals().get("status_var")
    if ui_root is not None and ui_status is not None:
        try:
            ui_root.after(0, lambda m=msg: ui_status.set(m))
        except Exception:
            pass


def cleanup(drop_pipeline=True):
    """Освобождает память; при drop_pipeline=False сохраняет загруженную SD-модель."""
    global PIPE, PIPE_KEY
    if drop_pipeline:
        PIPE = None
        PIPE_KEY = None
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def get_ffmpeg():
    """Находит FFmpeg в PATH или в бинарниках imageio-ffmpeg."""
    global FFMPEG_EXE
    if FFMPEG_EXE and Path(FFMPEG_EXE).is_file():
        return FFMPEG_EXE
    candidate = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if candidate:
        FFMPEG_EXE = candidate
        return candidate
    try:
        import imageio_ffmpeg
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if candidate and Path(candidate).is_file():
            FFMPEG_EXE = candidate
            return candidate
    except Exception:
        pass
    raise FileNotFoundError("FFmpeg не найден. Установи imageio-ffmpeg или добавь ffmpeg.exe в PATH.")


def scan_files(folder, suffixes=(".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx")):
    try:
        return sorted((p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in suffixes), key=lambda p: p.name.lower())
    except OSError as exc:
        log(f"Не удалось просканировать {folder}: {exc}")
        return []


def apply_optional_components(pipe, lora_rel=None, lora_scale=1.0, vae_rel=None, embedding_rel=None, ip_adapter_rel=None):
    """Подключает локальные LoRA/VAE/embedding/IP-Adapter, если они выбраны."""
    if vae_rel:
        vae_path = (ROOT / vae_rel).resolve()
        if not vae_path.exists():
            raise FileNotFoundError(f"VAE не найден: {vae_path}")
        try:
            from diffusers import AutoencoderKL
            if vae_path.is_dir():
                pipe.vae = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=DTYPE, local_files_only=True)
            else:
                pipe.vae = AutoencoderKL.from_single_file(str(vae_path), torch_dtype=DTYPE, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(f"Не удалось подключить VAE {vae_path.name}: {exc}") from exc
    if embedding_rel:
        emb_path = (ROOT / embedding_rel).resolve()
        if not emb_path.is_file():
            raise FileNotFoundError(f"Embedding не найден: {emb_path}")
        try:
            pipe.load_textual_inversion(str(emb_path), token=emb_path.stem)
        except Exception as exc:
            raise RuntimeError(f"Не удалось загрузить embedding {emb_path.name}: {exc}") from exc
    if lora_rel:
        lora_path = (ROOT / lora_rel).resolve()
        if not lora_path.is_file():
            raise FileNotFoundError(f"LoRA не найдена: {lora_path}")
        try:
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name, adapter_name="aether_selected")
            try:
                pipe.set_adapters(["aether_selected"], adapter_weights=[float(lora_scale)])
            except Exception:
                pipe.fuse_lora(lora_scale=float(lora_scale), adapter_names=["aether_selected"])
        except Exception as exc:
            raise RuntimeError(f"Не удалось подключить LoRA {lora_path.name}: {exc}") from exc
    if ip_adapter_rel:
        ip_path = (ROOT / ip_adapter_rel).resolve()
        if not ip_path.exists():
            raise FileNotFoundError(f"IP-Adapter не найден: {ip_path}")
        try:
            pipe.load_ip_adapter(str(ip_path.parent), subfolder="", weight_name=ip_path.name)
        except Exception as exc:
            raise RuntimeError(f"Не удалось подключить IP-Adapter {ip_path.name}: {exc}") from exc
    return pipe


def _is_complete_diffusers_model(model_dir: Path) -> bool:
    """Проверяет, что это корень модели Diffusers, а не случайный model_index.json."""
    index = model_dir / "model_index.json"
    if not index.is_file():
        return False

    # Минимальная структура базовой Stable Diffusion / SDXL Diffusers-модели.
    required_dirs = ("scheduler", "unet", "vae", "text_encoder", "tokenizer")
    if not all((model_dir / name).is_dir() for name in required_dirs):
        return False

    try:
        import json
        data = json.loads(index.read_text(encoding="utf-8"))
        class_name = str(data.get("_class_name", ""))
        if not class_name:
            return False
        return "StableDiffusion" in class_name or "Stable Diffusion" in class_name
    except Exception:
        return False


def list_checkpoints():
    """Находит одиночные checkpoint-файлы и полные Diffusers-модели.

    Важно: не считает CHECKPOINTS целой Diffusers-моделью только из-за
    случайного model_index.json. Веса .safetensors/.ckpt скрываются как
    компоненты только внутри подтверждённой Diffusers-папки.
    """
    found = []
    valid_diffusers_dirs = set()

    try:
        candidates = [p.parent for p in CHECKPOINTS.rglob("model_index.json") if p.is_file()]
    except OSError as e:
        log(f"Не удалось просканировать папки Diffusers: {e}")
        candidates = []

    for model_dir in candidates:
        try:
            if _is_complete_diffusers_model(model_dir):
                resolved = model_dir.resolve()
                valid_diffusers_dirs.add(resolved)
                found.append(str(model_dir.relative_to(ROOT)))
        except (OSError, ValueError) as e:
            log(f"Пропущена папка модели {model_dir}: {e}")

    try:
        for item in CHECKPOINTS.rglob("*"):
            if not item.is_file() or item.suffix.lower() not in (".safetensors", ".ckpt", ".gguf"):
                continue

            # Исключаем веса компонентов только если файл действительно
            # находится внутри подтверждённой Diffusers-модели.
            resolved_parents = {parent.resolve() for parent in item.parents}
            # Файл в корне CHECKPOINTS/модели может быть отдельным checkpoint;
            # скрываем только веса внутри компонентных подпапок (unet, vae и т.п.).
            component_model_dirs = {
                model_dir for model_dir in valid_diffusers_dirs
                if model_dir in resolved_parents and item.parent.resolve() != model_dir
            }
            if component_model_dirs:
                continue

            try:
                found.append(str(item.relative_to(ROOT)))
            except ValueError:
                continue
    except OSError as e:
        log(f"Ошибка при поиске checkpoint-файлов: {e}")

    # Убираем возможные дубликаты, сохраняя предпочтение Realistic Vision.
    found = list(dict.fromkeys(found))
    found.sort(
        key=lambda value: (
            PREFERRED_CHECKPOINT not in Path(value).name.lower(),
            value.lower()
        )
    )
    return found


def architecture(path: Path):
    """Определяет SD1.5/SDXL по структуре, метаданным и известным семействам."""
    path = Path(path)
    name = path.name.lower().replace("-", "_")

    if any(x in name for x in ("flux", "wan", "svd", "stable_video")):
        return "unsupported"

    # Explicit family names are more reliable than a generic substring test.
    if any(x in name for x in ("realisticvision", "realistic_vision", "realvis", "v5_1", "v6_0", "cyberrealistic", "v1_5", "sd15")):
        return "sd15"
    if any(x in name for x in ("sdxl", "pony", "animagine", "illustrious", "autismmix_xl", "xl_")):
        return "sdxl"

    if path.is_dir():
        index = path / "model_index.json"
        if index.is_file():
            try:
                data = json.loads(index.read_text(encoding="utf-8"))
                cls = str(data.get("_class_name", "")).lower()
                if "xl" in cls:
                    return "sdxl"
                if "stable" in cls and "diffusion" in cls:
                    return "sd15"
            except Exception:
                pass

    # Read safetensors metadata without loading the checkpoint into RAM/VRAM.
    if path.is_file() and path.suffix.lower() == ".safetensors":
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework="pt", device="cpu") as f:
                meta = f.metadata() or {}
            text = " ".join(str(v) for v in meta.values()).lower()
            if "sdxl" in text or "stable-diffusion-xl" in text or "stable_diffusion_xl" in text:
                return "sdxl"
            if "stable-diffusion-v1" in text or "sd 1.5" in text:
                return "sd15"
        except Exception:
            pass

    # Most single-file checkpoints in this application are SD1.5 unless
    # positively identified as SDXL above.
    return "sd15"


def patch_clip_embedding_device(pipe):
    """Avoid CLIP token-index CPU/CUDA mismatch seen with some Transformers/Accelerate builds."""
    for encoder_name in ("text_encoder", "text_encoder_2"):
        encoder = getattr(pipe, encoder_name, None)
        if encoder is None:
            continue
        try:
            embeddings = encoder.text_model.embeddings
            if getattr(embeddings, "_aether_device_patch", False):
                continue
            original_forward = embeddings.forward
            def safe_forward(this, input_ids=None, position_ids=None, inputs_embeds=None, _orig=original_forward, **kwargs):
                try:
                    target_device = this.token_embedding.weight.device
                    if input_ids is not None:
                        input_ids = input_ids.to(target_device)
                    if position_ids is not None:
                        position_ids = position_ids.to(target_device)
                    if inputs_embeds is not None:
                        inputs_embeds = inputs_embeds.to(target_device)
                except Exception:
                    pass
                return _orig(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, **kwargs)
            embeddings.forward = types.MethodType(safe_forward, embeddings)
            embeddings._aether_device_patch = True
        except Exception as exc:
            log(f"CLIP device guard skipped for {encoder_name}: {exc}")
    return pipe


def refresh_pipeline_offload(pipe):
    """Rebuild Accelerate offload hooks after replacing VAE/IP-Adapter components."""
    if DEVICE != "cuda":
        return pipe
    try:
        from accelerate.hooks import remove_hook_from_module
        for component in getattr(pipe, "components", {}).values():
            if isinstance(component, torch.nn.Module):
                remove_hook_from_module(component, recurse=True)
    except Exception:
        try:
            pipe.remove_all_hooks()
        except Exception:
            pass
    return _apply_memory_settings(pipe)


def _apply_memory_settings(pipe):
    if DEVICE == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception as first_error:
            try:
                pipe.enable_sequential_cpu_offload()
                log(f"Используется sequential CPU offload: {first_error}")
            except Exception:
                log(f"CPU offload недоступен; пробую загрузку на GPU: {first_error}")
                pipe.to("cuda")

        for method in (
            "enable_attention_slicing",
            "enable_vae_slicing",
            "enable_vae_tiling"
        ):
            try:
                getattr(pipe, method)()
            except Exception:
                pass
    else:
        pipe.to("cpu")

    return pipe


def load_pipeline(model_rel: str, task: str):
    """Загрузка single-file checkpoint или полной локальной Diffusers-папки."""
    global PIPE, PIPE_KEY

    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        DPMSolverMultistepScheduler
    )

    path = (ROOT / model_rel).resolve()

    if ROOT not in path.parents or not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}")

    is_diffusers = path.is_dir() and (path / "model_index.json").is_file()

    if path.is_dir() and not is_diffusers:
        raise RuntimeError(
            "Это не полная Diffusers-модель: в папке отсутствует model_index.json."
        )

    if not path.is_dir() and path.suffix.lower() not in (".safetensors", ".ckpt"):
        raise RuntimeError(
            "Поддерживаются .safetensors/.ckpt и полные Diffusers-папки."
        )

    arch = architecture(path)

    if arch == "unsupported":
        raise RuntimeError(
            "Это FLUX/Wan/SVD-модель. Для неё нужен отдельный backend, "
            "а не SDXL loader."
        )

    key = (str(path), arch, task, is_diffusers)

    if PIPE is not None and PIPE_KEY == key:
        return PIPE

    cleanup()

    common = {
        "torch_dtype": DTYPE,
        "local_files_only": True
    }

    if arch == "sdxl":
        classes = {
            "txt2img": StableDiffusionXLPipeline,
            "img2img": StableDiffusionXLImg2ImgPipeline,
            "inpaint": StableDiffusionXLInpaintPipeline,
        }
    else:
        classes = {
            "txt2img": StableDiffusionPipeline,
            "img2img": StableDiffusionImg2ImgPipeline,
            "inpaint": StableDiffusionInpaintPipeline,
        }

    cls = classes[task]
    pipe = None
    fallback_inpaint = False

    try:
        if is_diffusers:
            pipe = cls.from_pretrained(str(path), **common)
        elif arch == "sdxl":
            pipe = cls.from_single_file(
                str(path),
                config="stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=DTYPE,
                local_files_only=False,
            )
        else:
            # Single-file SD 1.5 checkpoints need the Diffusers component
            # configuration. Fetch it from the official SD 1.5 repo once,
            # then Hugging Face keeps it in its local cache.
            pipe = cls.from_single_file(
                str(path),
                config="stable-diffusion-v1-5/stable-diffusion-v1-5",
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                torch_dtype=DTYPE,
                local_files_only=False,
            )

    except Exception as first_error:
        if task != "inpaint":
            raise RuntimeError(
                f"Не удалось загрузить модель {path.name}: {first_error}"
            ) from first_error

        # Обычные SDXL/SD1.5 модели часто имеют UNet на 4 канала,
        # а специализированный Inpaint UNet — на 9.
        # Для обычной модели используем Img2Img и затем точно накладываем маску.
        img_cls = (
            StableDiffusionXLImg2ImgPipeline
            if arch == "sdxl"
            else StableDiffusionImg2ImgPipeline
        )

        try:
            if is_diffusers:
                pipe = img_cls.from_pretrained(str(path), **common)
            elif arch == "sdxl":
                pipe = img_cls.from_single_file(
                    str(path),
                    config="stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=DTYPE,
                    local_files_only=False,
                )
            else:
                pipe = img_cls.from_single_file(
                    str(path),
                    config="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                    torch_dtype=DTYPE,
                    local_files_only=False,
                )

            fallback_inpaint = True
            log(
                "Модель без специального inpaint-UNet: "
                "будет применён Img2Img + точное наложение маски."
            )

        except Exception as second_error:
            raise RuntimeError(
                f"Не удалось загрузить модель для Inpaint. "
                f"Основная ошибка: {first_error}; "
                f"резервная загрузка Img2Img: {second_error}"
            ) from second_error

    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )
    except Exception:
        pass

    pipe = patch_clip_embedding_device(pipe)
    pipe = _apply_memory_settings(pipe)
    pipe._ai_exotic_inpaint_fallback = fallback_inpaint

    PIPE, PIPE_KEY = pipe, key

    log(
        f"Модель загружена: {path.name} "
        f"({arch}, {'Diffusers' if is_diffusers else 'checkpoint'}, {task})"
    )

    return pipe


def dimensions(image: Image.Image, max_side: int, multiple=8):
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    w = max(multiple, int(w * scale) // multiple * multiple)
    h = max(multiple, int(h * scale) // multiple * multiple)
    return w, h


def seeded_generator(seed: int):
    seed = random.randrange(0, 2**31 - 1) if seed < 0 else seed
    return torch.Generator(device=DEVICE).manual_seed(seed), seed


def save_image(img: Image.Image, prefix="image"):
    name = (
        f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_"
        f"{random.randint(100,999)}.png"
    )
    path = OUTPUT / name
    img.save(path)
    return path


def run_generation(
    task, model, prompt, negative, source, mask, steps, cfg,
    strength, width, height, seed, callback,
    lora_rel=None, lora_scale=1.0, vae_rel=None, embedding_rel=None, ip_adapter_rel=None, ip_image_path=None
):
    """Генерация; Inpaint сохраняет все пиксели вне белой маски."""
    has_optional = any((lora_rel, vae_rel, embedding_rel, ip_adapter_rel))
    if has_optional:
        cleanup()
    pipe = load_pipeline(model, task)
    if has_optional:
        pipe = apply_optional_components(pipe, lora_rel, lora_scale, vae_rel, embedding_rel, ip_adapter_rel)
        pipe = patch_clip_embedding_device(pipe)
        pipe = refresh_pipeline_offload(pipe)
        if embedding_rel:
            embedding_token = Path(embedding_rel).stem
            if embedding_token.lower() not in prompt.lower():
                prompt = f"{embedding_token} {prompt}".strip()
        if ip_adapter_rel:
            if not ip_image_path or not Path(ip_image_path).is_file():
                raise RuntimeError("Для IP-Adapter выбери reference image.")
            kwargs_ip_image = Image.open(ip_image_path).convert("RGB")
            kwargs_ip_image.thumbnail((768, 768), Image.Resampling.LANCZOS)
    gen, actual_seed = seeded_generator(seed)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=int(steps),
        generator=gen
    )

    if task == "txt2img":
        kwargs.update(
            width=width,
            height=height,
            guidance_scale=float(cfg)
        )

    elif task == "img2img":
        if source is None:
            raise RuntimeError("Для Img2Img не передано исходное изображение.")

        init_image = source.convert("RGB")
        w, h = dimensions(init_image, 768)
        init_image = init_image.resize(
            (w, h),
            Image.Resampling.LANCZOS
        )

        strength = max(0.05, min(float(strength), 0.95))
        effective = max(1, int(round(int(steps) * strength)))

        log(
            f"Img2Img: {steps} шагов × strength {strength:.2f} "
            f"≈ {effective} фактических шагов"
        )

        kwargs.update(
            image=init_image,
            strength=strength,
            guidance_scale=float(cfg)
        )

    else:
        if source is None or mask is None:
            raise RuntimeError(
                "Для Inpaint не переданы исходное изображение или маска."
            )

        init_image = source.convert("RGB")
        mask_image = mask.convert("L")

        if mask_image.size != init_image.size:
            mask_image = mask_image.resize(
                init_image.size,
                Image.Resampling.NEAREST
            )

        if max(init_image.size) > 1024:
            w, h = dimensions(init_image, 1024)
            init_image = init_image.resize(
                (w, h),
                Image.Resampling.LANCZOS
            )
            mask_image = mask_image.resize(
                (w, h),
                Image.Resampling.NEAREST
            )

        strength = max(0.05, min(float(strength), 0.99))
        effective = max(1, int(round(int(steps) * strength)))

        log(
            f"Inpaint: {steps} шагов × strength {strength:.2f} "
            f"≈ {effective} фактических шагов"
        )

        kwargs.update(
            image=init_image,
            strength=strength,
            guidance_scale=float(cfg)
        )

        if not getattr(pipe, "_ai_exotic_inpaint_fallback", False):
            kwargs["mask_image"] = mask_image

    if ip_adapter_rel:
        kwargs["ip_adapter_image"] = kwargs_ip_image
    try:
        import inspect
        if "callback_on_step_end" in inspect.signature(pipe.__call__).parameters:
            def _step_progress(_pipe, step_index, timestep, callback_kwargs):
                set_progress((step_index + 1) / max(1, int(steps)), f"Diffusion step {step_index+1}/{steps}")
                return callback_kwargs
            kwargs["callback_on_step_end"] = _step_progress
    except Exception:
        pass
    with torch.inference_mode():
        generated = pipe(**kwargs).images[0].convert("RGB")

    if task == "inpaint":
        if generated.size != init_image.size:
            generated = generated.resize(
                init_image.size,
                Image.Resampling.LANCZOS
            )

        # Белая маска — новые пиксели; чёрная — исходные без изменений.
        result = Image.composite(generated, init_image, mask_image)
    else:
        result = generated

    path = save_image(result, task)
    callback(
        result,
        f"Готово: {path}\nSeed: {actual_seed}"
    )
    if has_optional:
        cleanup()


def upscale_image(
    image: Image.Image,
    callback,
    scale=4,
    model_name="realesrgan-x4plus"
):
    """
    Portable Real-ESRGAN NCNN Vulkan.
    ?? ????????? ?????????? RTX 3060 Laptop GPU (GPU ID 1).
    """

    import subprocess

    scale = int(scale)

    allowed = {
        "realesrgan-x4plus": (4,),
        "realesrgan-x4plus-anime": (4,),
        "realesr-animevideov3": (2, 3, 4),
    }

    if model_name not in allowed:
        raise ValueError(f"??????????? ?????? Real-ESRGAN: {model_name}")

    if scale not in allowed[model_name]:
        raise ValueError(
            f"?????? {model_name} ?? ???????????? ??????? x{scale}"
        )

    exe = (
        ROOT
        / "external"
        / "realesrgan"
        / "portable"
        / "realesrgan-ncnn-vulkan.exe"
    )

    models_dir = exe.parent / "models"

    if not exe.is_file():
        raise FileNotFoundError(f"Real-ESRGAN EXE ?? ??????: {exe}")

    model_suffix = f"-x{scale}" if model_name == "realesr-animevideov3" else ""

    for extension in (".bin", ".param"):
        model_file = models_dir / f"{model_name}{model_suffix}{extension}"
        if not model_file.is_file():
            raise FileNotFoundError(f"???? ?????? ?? ??????: {model_file}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    unique = f"{time.time_ns()}"
    input_path = OUTPUT / f"_realesrgan_input_{unique}.png"
    output_path = OUTPUT / f"realesrgan_{unique}_x{scale}.png"

    try:
        image.convert("RGB").save(input_path, format="PNG")

        command = [
            str(exe),
            "-i", str(input_path),
            "-o", str(output_path),
            "-n", model_name,
            "-s", str(scale),
            "-t", "256",
            "-m", str(models_dir),
            "-g", "1",
        ]

        process = subprocess.run(
            command,
            cwd=str(exe.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1200,
            check=False,
        )

        if process.returncode != 0 or not output_path.is_file():
            details = (
                (process.stderr or "") + "\n" +
                (process.stdout or "")
            ).strip()

            raise RuntimeError(
                "Real-ESRGAN ?????????? ? ???????.\n"
                f"??? ??????????: {process.returncode}\n"
                f"????? ?????????:\n{details[-5000:]}"
            )

        with Image.open(output_path) as result:
            result_image = result.convert("RGB").copy()

        message = (
            f"Real-ESRGAN NCNN Vulkan: ???????\n"
            f"??????: {model_name}\n"
            f"???????: x{scale}\n"
            f"GPU: NVIDIA GeForce RTX 3060 Laptop GPU (ID 1)\n"
            f"??????: {result_image.width} x {result_image.height}\n"
            f"????: {output_path}"
        )

        callback(result_image, message)

    finally:
        try:
            if input_path.exists():
                input_path.unlink()
        except OSError:
            pass


def restore_portrait(image_path, callback):
    """Восстановление деталей лица через GFPGAN."""
    weights = ROOT / "GFPGANv1.4.pth"

    if not weights.exists():
        raise FileNotFoundError(
            f"Не найден файл {weights}. "
            "Положи GFPGANv1.4.pth в корень E:\\AIExoticLocal."
        )

    try:
        from gfpgan import GFPGANer
    except Exception as e:
        raise RuntimeError(
            "Для восстановления лица установи совместимые gfpgan и basicsr. "
            "На Python 3.13 установка может быть ограничена совместимостью."
        ) from e

    if cv2 is None:
        raise RuntimeError("Для восстановления лица нужен opencv-python.")

    restorer = GFPGANer(
        model_path=str(weights),
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None
    )

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError("Не удалось открыть фотографию.")

    _, _, restored = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    if restored is None:
        raise RuntimeError(
            "GFPGAN не смог восстановить лицо на этом изображении."
        )

    result = Image.fromarray(
        cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    )

    out = save_image(result, "portrait_restore")
    callback(result, f"Восстановленное фото сохранено: {out}")


def submit_comfyui_workflow(server_url, workflow_path, callback):
    """Submit a ComfyUI API-format workflow to a local ComfyUI server and fetch outputs."""
    from urllib.parse import urlparse, urlencode
    base = server_url.strip().rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Укажи корректный адрес ComfyUI, например http://127.0.0.1:8188.")
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Для приватной локальной работы разрешён только localhost/127.0.0.1.")
    wf_path = Path(workflow_path)
    if not wf_path.is_file():
        raise FileNotFoundError("Выбери API workflow JSON, экспортированный из ComfyUI.")
    workflow = json.loads(wf_path.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("Workflow JSON пустой или имеет неверный формат.")
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    request = urllib.request.Request(base + "/prompt", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            submitted = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Не удалось отправить workflow в ComfyUI ({base}). Проверь, что сервер запущен: {exc}") from exc
    prompt_id = submitted.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI не вернул prompt_id: {submitted}")
    callback(None, f"ComfyUI: задача отправлена ({prompt_id}). Ожидаю результат…")
    history_url = base + "/history/" + urllib.parse.quote(str(prompt_id))
    started = time.time()
    history_entry = None
    while time.time() - started < 7200:
        try:
            with urllib.request.urlopen(history_url, timeout=30) as response:
                history = json.loads(response.read().decode("utf-8"))
            history_entry = history.get(str(prompt_id))
            if history_entry:
                status = history_entry.get("status", {})
                status_str = str(status.get("status_str", "")).lower()
                if status_str in ("error", "failed") or status.get("completed") is False and status.get("messages"):
                    raise RuntimeError("ComfyUI workflow завершился с ошибкой: " + json.dumps(status, ensure_ascii=False)[:3000])
                if status.get("completed") or status_str in ("success", "completed"):
                    break
        except urllib.error.HTTPError as exc:
            if exc.code not in (404,):
                raise
        elapsed = int(time.time() - started)
        set_progress(min(0.95, elapsed / 7200), f"ComfyUI выполняет workflow… {elapsed} сек")
        time.sleep(2)
    if not history_entry or not history_entry.get("outputs"):
        raise TimeoutError("ComfyUI не вернул готовые outputs за 2 часа.")

    result_dir = OUTPUT / f"comfyui_{datetime.now():%Y%m%d_%H%M%S}"
    result_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for node_id, node_output in history_entry.get("outputs", {}).items():
        for output_type in ("images", "gifs", "videos", "audio"):
            for item in node_output.get(output_type, []) or []:
                filename = item.get("filename")
                if not filename:
                    continue
                query = urlencode({"filename": filename, "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")})
                view_url = base + "/view?" + query
                suffix = Path(filename).suffix or (".mp4" if output_type in ("gifs", "videos") else ".png")
                dest = result_dir / f"node_{node_id}_{len(saved)+1:03d}{suffix}"
                try:
                    with urllib.request.urlopen(view_url, timeout=120) as response, dest.open("wb") as out_file:
                        shutil.copyfileobj(response, out_file)
                    saved.append(dest)
                except Exception as exc:
                    log(f"Не удалось скачать output {filename}: {exc}")
    if not saved:
        raise RuntimeError("ComfyUI завершил workflow, но в outputs не обнаружены файлы images/gifs/videos/audio.")
    callback(None, "ComfyUI workflow готов. Файлы сохранены:\n" + "\n".join(str(x) for x in saved))


def make_animatediff(model_rel, motion_rel, prompt, negative, steps, cfg, frames, fps, init_image, callback, motion_lora=None):
    """AnimateDiff SD1.5: локальный MotionAdapter и optional camera-motion LoRA."""
    if DEVICE != "cuda":
        raise RuntimeError("AnimateDiff требует NVIDIA CUDA.")
    from diffusers import AnimateDiffPipeline, MotionAdapter, DPMSolverMultistepScheduler
    model_path, motion_path = ROOT / model_rel, ROOT / motion_rel
    if not model_path.exists() or not motion_path.exists():
        raise FileNotFoundError("Не найдена базовая SD1.5 модель или motion adapter.")
    if architecture(model_path) != "sd15":
        raise RuntimeError("AnimateDiff совместим с SD1.5; выбери SD1.5 checkpoint.")
    cleanup()
    # Motion adapter — Diffusers directory, не single-file checkpoint.
    if motion_path.is_dir():
        adapter = MotionAdapter.from_pretrained(str(motion_path), torch_dtype=DTYPE, local_files_only=True)
    else:
        adapter = MotionAdapter.from_single_file(str(motion_path), torch_dtype=DTYPE)
    from diffusers import StableDiffusionPipeline
    if model_path.is_dir():
        base = StableDiffusionPipeline.from_pretrained(str(model_path), torch_dtype=DTYPE, local_files_only=True)
    else:
        base = StableDiffusionPipeline.from_single_file(
            str(model_path), config="stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=DTYPE, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False, local_files_only=False
        )
    video_pipe = AnimateDiffPipeline.from_pipe(base, motion_adapter=adapter)
    try:
        video_pipe.enable_model_cpu_offload()
    except Exception:
        video_pipe.to("cuda")
    for method in ("enable_vae_slicing", "enable_vae_tiling", "enable_attention_slicing"):
        try: getattr(video_pipe, method)()
        except Exception: pass
    video_pipe.scheduler = DPMSolverMultistepScheduler.from_config(video_pipe.scheduler.config, use_karras_sigmas=True)
    if motion_lora:
        lp = ROOT / motion_lora
        if lp.is_file():
            video_pipe.load_lora_weights(str(lp.parent), weight_name=lp.name, adapter_name="camera_motion")
            try: video_pipe.set_adapters(["camera_motion"], adapter_weights=[0.8])
            except Exception: pass
    seed = random.randrange(0, 2**31-1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = dict(prompt=prompt, negative_prompt=negative, num_frames=int(frames),
                  num_inference_steps=int(steps), guidance_scale=float(cfg), generator=gen)
    if init_image is not None:
        raise RuntimeError("AnimateDiff T2V в этой сборке не принимает стартовое изображение. Для I2V используй SVD-XT.")
    with torch.inference_mode():
        result = video_pipe(**kwargs)
    frame_list = result.frames[0]
    out = OUTPUT / f"animatediff_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    try:
        from diffusers.utils import export_to_video
        export_to_video(frame_list, str(out), fps=int(fps))
    except Exception:
        import imageio.v2 as imageio
        imageio.mimsave(str(out), [np.asarray(f) for f in frame_list], fps=int(fps))
    del video_pipe, adapter, base
    cleanup()
    callback(None, f"AnimateDiff создано: {out}\nSeed: {seed}")


def make_svd_video(image_path, motion_notes, negative, duration_seconds, fps, motion_bucket,
                   noise_strength, steps, decode_chunk, callback, seed=-1):
    """SVD-XT image-to-video with configurable duration via chained 25-frame segments.

    SVD-XT is image-conditioned: motion_notes are saved as metadata and are not
    interpreted as text conditioning. Long clips are generated by feeding each
    segment's final frame into the next segment; identity drift can accumulate.
    """
    if DEVICE != "cuda":
        raise RuntimeError("SVD-XT требует NVIDIA CUDA для приемлемой скорости.")
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video

    model_dir = MODELS / "svd" / "stable-video-diffusion-img2vid-xt"
    if not (model_dir / "model_index.json").is_file():
        raise FileNotFoundError(r"Не найдена полная модель models\svd\stable-video-diffusion-img2vid-xt.")
    if not image_path or not Path(image_path).is_file():
        raise FileNotFoundError("Выбери исходное изображение.")

    duration_seconds = float(duration_seconds)
    fps = int(fps)
    total_frames = int(round(duration_seconds * fps))
    if not 0.5 <= duration_seconds <= 180:
        raise ValueError("Длительность SVD-XT: от 0.5 до 180 секунд. Для длинных роликов лучше ComfyUI/video backend.")
    if not 3 <= fps <= 15:
        raise ValueError("Для SVD-XT рекомендуется 3–15 FPS.")
    if not 0 <= int(motion_bucket) <= 255:
        raise ValueError("Motion bucket должен быть от 0 до 255.")
    if not 0.0 <= float(noise_strength) <= 0.5:
        raise ValueError("Noise strength должен быть от 0 до 0.5.")
    if not 10 <= int(steps) <= 60:
        raise ValueError("SVD steps: 10–60.")
    if not 1 <= int(decode_chunk) <= 8:
        raise ValueError("Decode chunk: 1–8.")

    cleanup()
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, variant="fp16", local_files_only=True
    )
    pipe.enable_model_cpu_offload()
    try:
        pipe.enable_vae_slicing()
        pipe.unet.enable_forward_chunking()
    except Exception:
        pass

    source = Image.open(image_path).convert("RGB").resize((1024, 576), Image.Resampling.LANCZOS)
    actual_seed = random.randrange(0, 2**31 - 1) if int(seed) < 0 else int(seed)
    generator = torch.Generator(device="cpu").manual_seed(actual_seed)
    frames_out = []
    current_image = source
    remaining = total_frames
    segment_index = 0
    try:
        while remaining > 0:
            # 25 frames is the native SVD-XT segment length. For subsequent
            # segments, one frame is used as the overlap and removed on join.
            segment_frames = min(25, remaining if segment_index == 0 else remaining + 1)
            segment_frames = max(2, segment_frames) if remaining > 1 else 2
            segment_frames = min(25, segment_frames)
            log(f"SVD-XT segment {segment_index + 1}: {segment_frames} frames; remaining target={remaining}")
            with torch.inference_mode():
                result = pipe(
                    image=current_image,
                    num_frames=segment_frames,
                    num_inference_steps=int(steps),
                    decode_chunk_size=int(decode_chunk),
                    motion_bucket_id=int(motion_bucket),
                    noise_aug_strength=float(noise_strength),
                    fps=fps,
                    generator=generator,
                )
            segment = result.frames[0]
            if not segment:
                raise RuntimeError("SVD-XT returned an empty segment.")
            if segment_index == 0:
                frames_out.extend(segment)
            else:
                frames_out.extend(segment[1:])
            current_image = segment[-1].convert("RGB") if hasattr(segment[-1], "convert") else Image.fromarray(np.asarray(segment[-1])).convert("RGB")
            remaining = total_frames - len(frames_out)
            segment_index += 1
            set_progress(min(0.95, len(frames_out) / max(1, total_frames)), f"SVD-XT: {len(frames_out)}/{total_frames} frames")
            if segment_index > 200:
                raise RuntimeError("Превышен безопасный предел количества сегментов.")

        # Trim any rounding/overlap excess to the exact requested frame count.
        frames_out = frames_out[:total_frames]
        out = OUTPUT / f"svd_i2v_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        export_to_video(frames_out, str(out), fps=fps)
        metadata = {
            "motion_notes": motion_notes, "negative_prompt_notes": negative,
            "duration_seconds_requested": duration_seconds, "duration_seconds_actual": len(frames_out) / fps,
            "fps": fps, "frames": len(frames_out), "motion_bucket_id": int(motion_bucket),
            "noise_aug_strength": float(noise_strength), "steps": int(steps),
            "decode_chunk_size": int(decode_chunk), "seed": actual_seed,
            "segments": segment_index,
            "model_note": "Standard SVD-XT is image-conditioned; text notes are not used as prompt conditioning."
        }
        out.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        del pipe
        cleanup()
    callback(None, f"SVD-XT видео создано: {out}\nДлительность: {len(frames_out)/fps:.2f} сек | FPS: {fps} | Кадров: {len(frames_out)} | Сегментов: {segment_index}\nSeed: {actual_seed}\nВажно: при длинном последовательном продолжении возможно постепенное изменение лица/деталей.")



def list_controlnet_models():
    return sorted(
        str(p.relative_to(ROOT))
        for p in CONTROLNET.rglob("*")
        if p.is_file() and p.suffix.lower() in (".safetensors", ".ckpt")
    )


def run_controlnet_mode(mode, model_rel, control_rel, image_path, prompt, negative, steps, cfg, strength, callback):
    """ControlNet SD1.5: Canny/Tile locally; OpenPose/Depth use optional controlnet_aux preprocessors."""
    mode=str(mode).lower()
    if mode=="canny":
        if "canny" not in Path(control_rel).name.lower():
            raise RuntimeError("Для режима Canny выбери control_v11p_sd15_canny.safetensors или controlnet_canny.safetensors.")
        return run_controlnet_canny(model_rel,control_rel,image_path,prompt,negative,steps,cfg,strength,callback)
    expected={"openpose":"openpose","depth":"depth","tile":"tile"}.get(mode)
    if expected and expected not in Path(control_rel).name.lower():
        raise RuntimeError(f"Для режима {mode.title()} выбери соответствующий ControlNet файл с '{expected}' в имени.")
    if not image_path or not Path(image_path).is_file(): raise FileNotFoundError("Выбери исходное изображение.")
    if architecture(ROOT/model_rel)!="sd15": raise RuntimeError("ControlNet weights in this project are SD1.5; select SD1.5 checkpoint.")
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
    if cv2 is None: raise RuntimeError("Для ControlNet требуется opencv-python.")
    control_path=ROOT/control_rel
    controlnet=ControlNetModel.from_single_file(str(control_path),config="lllyasviel/ControlNet",torch_dtype=DTYPE,local_files_only=False)
    pipe=StableDiffusionControlNetPipeline.from_single_file(str(ROOT/model_rel),config="stable-diffusion-v1-5/stable-diffusion-v1-5",controlnet=controlnet,torch_dtype=DTYPE,safety_checker=None,feature_extractor=None,requires_safety_checker=False,local_files_only=False)
    try: pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
    except Exception: pass
    if DEVICE=="cuda":
        pipe.enable_model_cpu_offload()
        for m in ("enable_vae_slicing","enable_vae_tiling","enable_attention_slicing"):
            try:getattr(pipe,m)()
            except Exception:pass
    src=Image.open(image_path).convert("RGB"); w,h=dimensions(src,768); src=src.resize((w,h),Image.Resampling.LANCZOS)
    if mode=="tile": control_image=src
    else:
        try:
            from controlnet_aux import OpenposeDetector, MidasDetector
            if mode=="openpose":
                detector=OpenposeDetector.from_pretrained("lllyasviel/Annotators", local_files_only=True); control_image=detector(src)
            elif mode=="depth":
                detector=MidasDetector.from_pretrained("lllyasviel/Annotators", local_files_only=True); control_image=detector(src)
            else: raise RuntimeError("Unknown ControlNet mode.")
        except Exception as exc:
            raise RuntimeError(f"Для {mode.title()} preprocessor нужен пакет controlnet-aux и его локальные веса. Исходные ControlNet weights уже найдены. Детали: {exc}") from exc
    gen,seed=seeded_generator(-1)
    with torch.inference_mode():
        result=pipe(prompt=prompt,negative_prompt=negative,image=control_image,width=w,height=h,num_inference_steps=int(steps),guidance_scale=float(cfg),controlnet_conditioning_scale=float(strength),generator=gen).images[0]
    out=save_image(result,"controlnet_"+mode); del pipe,controlnet; cleanup()
    callback(result,f"ControlNet {mode.title()} готово: {out}\nSeed: {seed}")


def run_controlnet_canny(
    model_rel, control_rel, image_path, prompt, negative,
    steps, cfg, strength, callback
):
    """SD1.5 ControlNet Canny."""
    if not image_path or not Path(image_path).is_file():
        raise FileNotFoundError(
            "Выбери исходное изображение для ControlNet."
        )

    if not control_rel:
        raise FileNotFoundError(
            "В models\\controlnet нет файла ControlNet."
        )

    if architecture(ROOT / model_rel) != "sd15":
        raise RuntimeError(
            "Этот режим ControlNet Canny рассчитан на SD 1.5 checkpoint."
        )

    if cv2 is None:
        raise RuntimeError("Для Canny нужен пакет opencv-python.")

    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        DPMSolverMultistepScheduler
    )

    cleanup()

    control_path = ROOT / control_rel
    checkpoint_path = ROOT / model_rel

    controlnet = ControlNetModel.from_single_file(
        str(control_path), config="lllyasviel/ControlNet",
        torch_dtype=DTYPE, local_files_only=False
    )

    pipe = StableDiffusionControlNetPipeline.from_single_file(
        str(checkpoint_path),
        config="stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=False
    )

    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )
    except Exception:
        pass

    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        try:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        except Exception:
            pass
    else:
        pipe.to("cpu")

    src = Image.open(image_path).convert("RGB")
    w, h = dimensions(src, 768)
    src = src.resize((w, h), Image.Resampling.LANCZOS)

    arr = cv2.cvtColor(
        np.array(src),
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(arr, 100, 200)

    control_image = Image.fromarray(
        cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    )

    gen, seed = seeded_generator(-1)

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=control_image,
            width=w,
            height=h,
            num_inference_steps=steps,
            guidance_scale=cfg,
            controlnet_conditioning_scale=strength,
            generator=gen
        ).images[0]

    out = save_image(result, "controlnet_canny")

    del pipe, controlnet
    cleanup()

    callback(
        result,
        f"ControlNet Canny готово: {out}\nSeed: {seed}"
    )


def _initialize_insightface_models():
    global FACE_ANALYSER, FACE_SWAPPER
    model_file=INSIGHT/"inswapper_128.onnx"
    if not model_file.is_file(): raise FileNotFoundError(f"Не найден {model_file}")
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    if FACE_ANALYSER is None:
        try:
            FACE_ANALYSER=FaceAnalysis(name="buffalo_l",root=str(MODELS/"insightface"),providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            FACE_ANALYSER.prepare(ctx_id=0 if torch.cuda.is_available() else -1,det_size=(640,640))
        except Exception as exc:
            log(f"InsightFace fallback to CPU: {exc}")
            FACE_ANALYSER=FaceAnalysis(name="buffalo_l",root=str(MODELS/"insightface"),providers=["CPUExecutionProvider"])
            FACE_ANALYSER.prepare(ctx_id=-1,det_size=(640,640))
    if FACE_SWAPPER is None:
        FACE_SWAPPER=get_model(str(model_file),download=False,download_zip=False)


def run_face_swap(source_face_path, target_image_path, callback):
    """Face Swap через локальные buffalo_l и inswapper_128.onnx."""
    global FACE_ANALYSER, FACE_SWAPPER

    if cv2 is None:
        raise RuntimeError("Для Face Swap установи opencv-python.")

    model_file = INSIGHT / "inswapper_128.onnx"

    if not model_file.is_file():
        raise FileNotFoundError(f"Не найден {model_file}")

    if not source_face_path or not Path(source_face_path).is_file():
        raise FileNotFoundError("Выбери фото лица-источника.")

    if not target_image_path or not Path(target_image_path).is_file():
        raise FileNotFoundError("Выбери целевое фото.")

    try:
        _initialize_insightface_models()
    except Exception as e:
        raise RuntimeError(f"Не удалось инициализировать InsightFace: {e}") from e

    target_bgr = cv2.imread(
        str(target_image_path),
        cv2.IMREAD_COLOR
    )

    source_bgr = cv2.imread(
        str(source_face_path),
        cv2.IMREAD_COLOR
    )

    if target_bgr is None or source_bgr is None:
        raise RuntimeError(
            "Не удалось прочитать одно из изображений."
        )

    target_faces = FACE_ANALYSER.get(target_bgr)
    source_faces = FACE_ANALYSER.get(source_bgr)

    if not target_faces:
        raise RuntimeError("На целевом фото не найдено лицо.")

    if not source_faces:
        raise RuntimeError("На фото-источнике не найдено лицо.")

    source_face = max(
        source_faces,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )
    )

    target_face = max(
        target_faces,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )
    )

    swapped = FACE_SWAPPER.get(
        target_bgr,
        target_face,
        source_face,
        paste_back=True
    )

    result = Image.fromarray(
        cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)
    )

    out = save_image(result, "face_swap")
    callback(result, f"Face Swap сохранён: {out}")


def run_video_face_swap(source_face_path, target_video_path, callback, source_face_index=1, target_face_index=1):
    """Frame-by-frame FaceSwap for user-provided/authorized source and target media."""
    global FACE_ANALYSER, FACE_SWAPPER
    if cv2 is None:
        raise RuntimeError("Для FaceSwap video требуется opencv-python.")
    source_path, video_path = Path(source_face_path), Path(target_video_path)
    if not source_path.is_file() or not video_path.is_file():
        raise FileNotFoundError("Укажи существующее фото-источник и видео-цель.")
    # Reuse the exact same local InsightFace models as photo FaceSwap.
    if FACE_ANALYSER is None or FACE_SWAPPER is None:
        _initialize_insightface_models()
    source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    source_faces = FACE_ANALYSER.get(source_bgr)
    if not source_faces:
        raise RuntimeError("На фото-источнике не найдено лицо.")
    source_faces = sorted(source_faces, key=lambda f: float(f.bbox[0]))
    source_face_index = int(source_face_index) - 1
    target_face_index = int(target_face_index) - 1
    if not 0 <= source_face_index < len(source_faces):
        raise RuntimeError(f"На фото-источнике найдено {len(source_faces)} лиц; выбран индекс {source_face_index+1}.")
    source_face = source_faces[source_face_index]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width < 2 or height < 2:
        cap.release(); raise RuntimeError("Некорректное разрешение исходного видео.")
    temp_video = OUTPUT / f"_faceswap_video_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release(); raise RuntimeError("Не удалось создать временное видео.")
    index = 0
    last_target_center = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            faces = FACE_ANALYSER.get(frame)
            if faces:
                if last_target_center is None:
                    ordered = sorted(faces, key=lambda f: float(f.bbox[0]))
                    target_face = ordered[target_face_index] if 0 <= target_face_index < len(ordered) else None
                else:
                    def center_distance(face):
                        cx=(float(face.bbox[0])+float(face.bbox[2]))/2
                        cy=(float(face.bbox[1])+float(face.bbox[3]))/2
                        return (cx-last_target_center[0])**2+(cy-last_target_center[1])**2
                    target_face=min(faces,key=center_distance)
                if target_face is not None:
                    last_target_center=((float(target_face.bbox[0])+float(target_face.bbox[2]))/2,
                                        (float(target_face.bbox[1])+float(target_face.bbox[3]))/2)
                    frame = FACE_SWAPPER.get(frame, target_face, source_face, paste_back=True)
            writer.write(frame)
            index += 1
            if total:
                set_progress(index / total, f"FaceSwap video: {index}/{total} frames")
    finally:
        cap.release(); writer.release()
    out = OUTPUT / f"faceswap_video_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    ffmpeg = get_ffmpeg()
    # Copy original audio when present; if audio copy fails, keep silent processed video.
    cmd = [ffmpeg, "-y", "-i", str(temp_video), "-i", str(video_path), "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-shortest", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        shutil.copy2(temp_video, out)
        log("FFmpeg не смог перенести аудио; сохранено видео без звука: " + (proc.stderr[-600:] if proc.stderr else ""))
    try: temp_video.unlink(missing_ok=True)
    except Exception: pass
    callback(None, f"FaceSwap video сохранён: {out}\nОбработано кадров: {index}")


def export_timeline(media_items, output_path, audio_path=None, fps=30):
    """Simple sequential timeline export with normalized 1280x720 clips."""
    if not media_items:
        raise RuntimeError("Добавь хотя бы один файл на Timeline.")
    ffmpeg=get_ffmpeg(); output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    work=OUTPUT/f"timeline_work_{datetime.now():%Y%m%d_%H%M%S}"; work.mkdir(parents=True,exist_ok=True)
    clips=[]
    try:
        encoders=subprocess.run([ffmpeg,"-hide_banner","-encoders"],capture_output=True,text=True).stdout
        video_encoder="h264_nvenc" if DEVICE=="cuda" and "h264_nvenc" in encoders else "libx264"
        for idx,item in enumerate(media_items):
            path=Path(item["path"]); duration=item.get("duration")
            if not path.is_file(): raise FileNotFoundError(f"Медиафайл не найден: {path}")
            clip=work/f"clip_{idx:04d}.mp4"
            vf=f"scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps={int(fps)},format=yuv420p"
            if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                if duration is None or float(duration)<=0: raise ValueError(f"Для фото {path.name} укажи длительность больше 0 секунд.")
                cmd=[ffmpeg,"-y","-loop","1","-i",str(path),"-t",str(float(duration)),"-vf",vf,"-an","-c:v",video_encoder,"-preset","p4" if video_encoder=="h264_nvenc" else "veryfast","-pix_fmt","yuv420p",str(clip)]
            else:
                cmd=[ffmpeg,"-y","-i",str(path)]
                if duration is not None and float(duration)>0: cmd += ["-t",str(float(duration))]
                cmd += ["-vf",vf,"-an","-c:v",video_encoder,"-preset","p4" if video_encoder=="h264_nvenc" else "veryfast","-pix_fmt","yuv420p",str(clip)]
            proc=subprocess.run(cmd,capture_output=True,text=True)
            if proc.returncode: raise RuntimeError(f"FFmpeg clip {path.name}: {proc.stderr[-1200:]}")
            clips.append(clip)
            set_progress((idx+1)/max(1,len(media_items))*0.8,f"Timeline: подготовка клипа {idx+1}/{len(media_items)}")
        manifest=work/"concat.txt"
        manifest.write_text("\n".join("file '"+str(p).replace("\\","/")+"'" for p in clips),encoding="utf-8")
        silent=work/"joined.mp4"
        proc=subprocess.run([ffmpeg,"-y","-f","concat","-safe","0","-i",str(manifest),"-c","copy",str(silent)],capture_output=True,text=True)
        if proc.returncode: raise RuntimeError("FFmpeg concat: "+proc.stderr[-1200:])
        cmd=[ffmpeg,"-y","-i",str(silent)]
        if audio_path and Path(audio_path).is_file():
            cmd += ["-i",str(audio_path),"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-shortest"]
        else: cmd += ["-c","copy"]
        cmd += [str(output_path)]
        proc=subprocess.run(cmd,capture_output=True,text=True)
        if proc.returncode: raise RuntimeError("FFmpeg export: "+proc.stderr[-1200:])
    finally:
        shutil.rmtree(work,ignore_errors=True)
    return output_path



def create_lipsync_video(face_video, audio_file, callback):
    """Wav2Lip bridge; requires a local Wav2Lip checkout and checkpoint."""
    wav2lip_root=TOOLS_DIR/"Wav2Lip"
    checkpoint_candidates=[wav2lip_root/"checkpoints"/"wav2lip_gan.pth", wav2lip_root/"checkpoints"/"wav2lip.pth"]
    checkpoint=next((p for p in checkpoint_candidates if p.is_file()),None)
    script=wav2lip_root/"inference.py"
    if not script.is_file() or checkpoint is None:
        raise RuntimeError("Lip-sync backend не установлен. Помести Wav2Lip в tools/Wav2Lip и checkpoint в tools/Wav2Lip/checkpoints/. UI и интеграция готовы, но модельных весов в текущем наборе нет.")
    if not Path(face_video).is_file() or not Path(audio_file).is_file():
        raise FileNotFoundError("Выбери исходное видео и WAV/MP3 аудио.")
    out=OUTPUT/f"lipsync_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    cmd=[sys.executable,str(script),"--checkpoint_path",str(checkpoint),"--face",str(face_video),"--audio",str(audio_file),"--outfile",str(out)]
    proc=subprocess.run(cmd,cwd=str(wav2lip_root),capture_output=True,text=True)
    if proc.returncode or not out.is_file(): raise RuntimeError("Wav2Lip error: "+(proc.stderr or proc.stdout)[-1500:])
    callback(None,f"Lip-sync готов: {out}")


def run_rvc_conversion(audio_file, model_file, index_file, callback):
    """RVC bridge to a local RVC-WebUI installation; no cloud upload."""
    rvc_root=TOOLS_DIR/"RVC"
    infer=rvc_root/"infer-web.py"
    cli=rvc_root/"infer_cli.py"
    if not Path(audio_file).is_file() or not Path(model_file).is_file():
        raise FileNotFoundError("Выбери входное аудио и RVC .pth модель.")
    if not cli.is_file():
        raise RuntimeError("RVC CLI wrapper не найден. Помести совместимый infer_cli.py в tools/RVC (аргументы: --input, --model, --output, optional --index). Стандартный infer-web.py запускает Gradio UI и не имеет стабильного CLI API.")
    out=OUTPUT/f"rvc_{datetime.now():%Y%m%d_%H%M%S}.wav"
    cmd=[sys.executable,str(cli),"--input",str(audio_file),"--model",str(model_file),"--output",str(out)]
    if index_file and Path(index_file).is_file(): cmd += ["--index",str(index_file)]
    proc=subprocess.run(cmd,cwd=str(rvc_root),capture_output=True,text=True)
    if proc.returncode or not out.is_file(): raise RuntimeError("RVC failed: "+(proc.stderr or proc.stdout)[-1500:])
    callback(None,f"RVC converted audio: {out}")


def run_facefusion(source_path,target_path,callback):
    """Optional local FaceFusion CLI bridge; never uploads media."""
    root=TOOLS_DIR/"facefusion"
    script=root/"facefusion.py"
    if not script.is_file(): raise RuntimeError("FaceFusion не найден. Установи его локально в tools/facefusion; встроенный InsightFace FaceSwap уже доступен отдельно.")
    if not Path(source_path).is_file() or not Path(target_path).is_file(): raise FileNotFoundError("Выбери фото-источник и целевое фото/видео.")
    out=OUTPUT/f"facefusion_{datetime.now():%Y%m%d_%H%M%S}{Path(target_path).suffix or '.mp4'}"
    cmd=[sys.executable,str(script),"headless-run","--source-path",str(source_path),"--target-path",str(target_path),"--output-path",str(out),"--processors","face_swapper"]
    proc=subprocess.run(cmd,cwd=str(root),capture_output=True,text=True)
    if proc.returncode or not out.is_file(): raise RuntimeError("FaceFusion failed: "+(proc.stderr or proc.stdout)[-1500:])
    callback(None,f"FaceFusion output: {out}")


def synthesize_voice_piper(text_value, model_path, callback):
    """Локальный синтез речи через Piper."""
    if not text_value.strip():
        raise RuntimeError("Введи текст для озвучивания.")

    if not model_path or not Path(model_path).is_file():
        raise FileNotFoundError(
            "Выбери Piper voice .onnx из models\\voice."
        )

    exe = shutil.which("piper")
    local_exe = ROOT / "tools" / "piper" / "piper.exe"

    if not exe and local_exe.is_file():
        exe = str(local_exe)

    out = OUTPUT / f"voice_{datetime.now():%Y%m%d_%H%M%S}.wav"

    if exe:
        command = [
            exe,
            "--model",
            str(model_path),
            "--output_file",
            str(out)
        ]
        input_text = text_value
    else:
        voice_file = Path(model_path)
        command = [
            sys.executable,
            "-m",
            "piper",
            "-m",
            voice_file.stem,
            "--data-dir",
            str(voice_file.parent),
            "-f",
            str(out),
            "--",
            text_value
        ]
        input_text = None

    proc = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=str(ROOT)
    )

    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(
            "Piper завершился с ошибкой:\n" +
            (proc.stderr or proc.stdout or "неизвестная ошибка")
        )

    callback(None, f"Голос сохранён: {out}")


class Studio:
    def __init__(self, master):
        self.master = master
        self.master.title("AETHER AI ULTIMATE STUDIO — Local")
        self.master.geometry("1420x900")
        self.master.minsize(1100, 720)
        self.master.configure(bg="#11131a")

        self.source = None
        self.source_path = None
        self.mask = None
        self.tk_preview = None
        self.preview_image = None
        self.brush = tk.IntVar(value=35)
        self.mode = tk.StringVar(value="txt2img")

        self._build()

    def _build(self):
        style = ttk.Style()

        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "TNotebook",
            background="#11131a",
            borderwidth=0
        )
        style.configure(
            "TNotebook.Tab",
            padding=(14, 9)
        )
        style.configure(
            "TButton",
            padding=7
        )

        header = tk.Frame(
            self.master,
            bg="#171b27",
            height=54
        )
        header.pack(fill="x")

        tk.Label(
            header,
            text="AETHER AI ULTIMATE STUDIO",
            bg="#171b27",
            fg="#e8edff",
            font=("Segoe UI", 17, "bold")
        ).pack(side="left", padx=18, pady=10)

        self.status = tk.StringVar(
            value=(
                f"Устройство: {DEVICE.upper()} | "
                f"Локальные модели: {len(list_checkpoints())}"
            )
        )

        global status_var
        status_var = self.status

        tk.Label(
            header,
            textvariable=self.status,
            bg="#171b27",
            fg="#9aa8c7"
        ).pack(side="right", padx=18)

        self.tabs = ttk.Notebook(self.master)
        self.tabs.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        self.tab_gen = tk.Frame(self.tabs, bg="#11131a")
        self.tab_edit = tk.Frame(self.tabs, bg="#11131a")
        self.tab_video = tk.Frame(self.tabs, bg="#11131a")
        self.tab_face = tk.Frame(self.tabs, bg="#11131a")
        self.tab_tools = tk.Frame(self.tabs, bg="#11131a")
        self.tab_control = tk.Frame(self.tabs, bg="#11131a")
        self.tab_voice = tk.Frame(self.tabs, bg="#11131a")
        self.tab_character = tk.Frame(self.tabs, bg="#11131a")
        self.tab_scene = tk.Frame(self.tabs, bg="#11131a")
        self.tab_timeline = tk.Frame(self.tabs, bg="#11131a")
        self.tab_comfy = tk.Frame(self.tabs, bg="#11131a")
        self.tab_system = tk.Frame(self.tabs, bg="#11131a")

        self.tabs.add(self.tab_gen, text="Image Studio")
        self.tabs.add(self.tab_edit, text="AI Canvas / Edit")
        self.tabs.add(self.tab_character, text="Character Studio")
        self.tabs.add(self.tab_scene, text="Scene Studio")
        self.tabs.add(self.tab_video, text="Motion / Video")
        self.tabs.add(self.tab_timeline, text="Timeline / Export")
        self.tabs.add(self.tab_comfy, text="ComfyUI Bridge")
        self.tabs.add(self.tab_face, text="Face Studio")
        self.tabs.add(self.tab_tools, text="Enhancement")
        self.tabs.add(self.tab_control, text="ControlNet")
        self.tabs.add(self.tab_voice, text="Voice Studio")
        self.tabs.add(self.tab_system, text="Model / System")

        self.progress = ttk.Progressbar(self.master, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=12, pady=(0, 8))
        global progress_widget
        progress_widget = self.progress
        self.timeline_items = []
        self._build_gen()
        self._build_edit()
        self._build_character()
        self._build_scene()
        self._build_video()
        self._build_timeline()
        self._build_comfy()
        self._build_face()
        self._build_tools()
        self._build_controlnet()
        self._build_voice()
        self._build_system()

    def model_box(self, parent):
        vals = list_checkpoints()
        cb = ttk.Combobox(
            parent,
            values=vals,
            state="readonly",
            width=55
        )

        if vals:
            cb.current(0)

        return cb

    def _field(self, parent, label, default="", height=3):
        tk.Label(
            parent,
            text=label,
            bg="#11131a",
            fg="#cbd5e1",
            anchor="w"
        ).pack(
            fill="x",
            padx=12,
            pady=(8, 2)
        )

        box = tk.Text(
            parent,
            height=height,
            bg="#202637",
            fg="#f8fafc",
            insertbackground="white",
            relief="flat",
            wrap="word"
        )

        box.pack(fill="x", padx=12)
        box.insert("1.0", default)

        return box

    def _build_gen(self):
        left = tk.Frame(
            self.tab_gen,
            bg="#171b27",
            width=510
        )
        left.pack(
            side="left",
            fill="y",
            padx=8,
            pady=8
        )
        left.pack_propagate(False)

        right = tk.Frame(
            self.tab_gen,
            bg="#171b27"
        )
        right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=8,
            pady=8
        )

        tk.Label(
            left,
            text="Локальная модель",
            bg="#171b27",
            fg="white"
        ).pack(
            anchor="w",
            padx=12,
            pady=8
        )

        self.gen_model = self.model_box(left)
        self.gen_model.pack(fill="x", padx=12)

        self.gen_style = ttk.Combobox(
            left,
            state="readonly",
            values=[
                "Фотореализм",
                "Аниме",
                "Нейтральный"
            ]
        )
        self.gen_style.set("Фотореализм")
        self.gen_style.pack(fill="x", padx=12, pady=4)

        tk.Label(left, text="LoRA (optional)", bg="#171b27", fg="#cbd5e1").pack(anchor="w", padx=12, pady=(8, 2))
        self.gen_lora = ttk.Combobox(left, state="readonly", values=[""] + [str(p.relative_to(ROOT)) for p in scan_files(LORA_DIR, (".safetensors", ".ckpt", ".pt"))])
        self.gen_lora.pack(fill="x", padx=12)
        self.gen_lora_weight = self._entry(left, "LoRA weight", "0.75")
        tk.Label(left, text="VAE / Embedding / IP-Adapter (optional)", bg="#171b27", fg="#cbd5e1").pack(anchor="w", padx=12, pady=(8, 2))
        self.gen_vae = ttk.Combobox(left, state="readonly", values=[""] + [str(p.relative_to(ROOT)) for p in scan_files(VAE_DIR, (".safetensors", ".ckpt"))])
        self.gen_vae.pack(fill="x", padx=12)
        self.gen_embedding = ttk.Combobox(left, state="readonly", values=[""] + [str(p.relative_to(ROOT)) for p in scan_files(EMBEDDINGS_DIR, (".safetensors", ".pt", ".bin"))])
        self.gen_embedding.pack(fill="x", padx=12, pady=3)
        self.gen_ip_adapter = ttk.Combobox(left, state="readonly", values=[""] + [str(p.relative_to(ROOT)) for p in scan_files(IP_ADAPTER_DIR, (".safetensors", ".bin"))])
        self.gen_ip_adapter.pack(fill="x", padx=12, pady=3)
        self.gen_ip_image = tk.StringVar()
        ip_row=tk.Frame(left,bg="#171b27"); ip_row.pack(fill="x",padx=12,pady=3)
        tk.Entry(ip_row,textvariable=self.gen_ip_image,bg="#202637",fg="white").pack(side="left",fill="x",expand=True)
        ttk.Button(ip_row,text="IP reference…",command=lambda:self.pick_image_preview(self.gen_ip_image,self.gen_ip_preview)).pack(side="right")
        self.gen_ip_preview=tk.Label(left,text="Reference preview: no image selected",bg="#171b27",fg="#9aa8c7")
        self.gen_ip_preview.pack(anchor="w",padx=12,pady=2)

        self.gen_prompt = self._field(
            left,
            "Prompt",
            "cinematic photorealistic portrait, natural skin texture, detailed lighting",
            5
        )

        self.gen_neg = self._field(
            left,
            "Negative prompt",
            "blurry, low quality, distorted anatomy, watermark",
            3
        )

        self.gen_steps = self._entry(left, "Steps", "25")
        self.gen_cfg = self._entry(left, "CFG", "6.5")
        self.gen_w = self._entry(left, "Width", "768")
        self.gen_h = self._entry(left, "Height", "768")
        self.gen_seed = self._entry(left, "Seed (-1 = random)", "-1")

        ttk.Button(
            left,
            text="Сгенерировать изображение",
            command=self.generate
        ).pack(
            fill="x",
            padx=12,
            pady=14
        )

        self.preview = tk.Label(
            right,
            text="Предпросмотр результата",
            bg="#171b27",
            fg="#9aa8c7"
        )
        self.preview.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        self.output_text = tk.Text(
            right,
            height=4,
            bg="#202637",
            fg="#cbd5e1",
            relief="flat"
        )
        self.output_text.pack(
            fill="x",
            padx=10,
            pady=10
        )

    def _entry(self, parent, label, default):
        row = tk.Frame(parent, bg="#171b27")
        row.pack(fill="x", padx=12, pady=3)

        tk.Label(
            row,
            text=label,
            width=23,
            anchor="w",
            bg="#171b27",
            fg="#cbd5e1"
        ).pack(side="left")

        e = tk.Entry(
            row,
            bg="#202637",
            fg="white",
            insertbackground="white",
            relief="flat"
        )
        e.pack(side="right", fill="x", expand=True)
        e.insert(0, default)

        return e

    def _build_edit(self):
        left = tk.Frame(
            self.tab_edit,
            bg="#171b27",
            width=430
        )
        left.pack(
            side="left",
            fill="y",
            padx=8,
            pady=8
        )
        left.pack_propagate(False)

        right = tk.Frame(
            self.tab_edit,
            bg="#171b27"
        )
        right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=8,
            pady=8
        )

        self.edit_model = self.model_box(left)
        self.edit_model.pack(fill="x", padx=12, pady=8)

        ttk.Button(
            left,
            text="Открыть фото",
            command=self.open_source
        ).pack(fill="x", padx=12, pady=4)

        self.edit_prompt = self._field(
            left,
            "Что изменить / дорисовать",
            "continue the scene naturally, matching lighting, perspective and texture",
            4
        )

        preset_row = tk.Frame(left, bg="#171b27")
        preset_row.pack(fill="x", padx=12, pady=5)

        tk.Label(
            preset_row,
            text="Быстрые пресеты",
            bg="#171b27",
            fg="#cbd5e1"
        ).pack(anchor="w")

        self.edit_preset = ttk.Combobox(
            preset_row,
            state="readonly",
            values=[
                "Одежда (повседневная)",
                "Одежда (деловой стиль)",
                "Одежда (спорт)",
                "Причёска",
                "Фон / окружение",
                "Освещение",
                "Поза / положение тела",
                "Фотореализм",
                "Кинематографичный стиль",
                "Аниме / иллюстрация"
            ]
        )

        self.edit_preset.pack(fill="x", pady=3)
        self.edit_preset.bind(
            "<<ComboboxSelected>>",
            self.apply_edit_preset
        )

        self.edit_neg = self._field(
            left,
            "Negative prompt",
            "blurry, artifacts, extra fingers, text, deformed anatomy",
            2
        )

        self.edit_steps = self._entry(left, "Steps", "28")
        self.edit_cfg = self._entry(left, "CFG", "6.5")
        self.edit_strength = self._entry(
            left,
            "Strength (28 × 0.25 ≈ 7 steps)",
            "0.25"
        )

        tk.Label(
            left,
            text="Размер кисти маски",
            bg="#171b27",
            fg="white"
        ).pack(anchor="w", padx=12)

        tk.Scale(
            left,
            from_=5,
            to=120,
            orient="horizontal",
            variable=self.brush,
            bg="#171b27",
            fg="white",
            highlightthickness=0
        ).pack(fill="x", padx=12)

        ttk.Button(
            left,
            text="Сбросить маску",
            command=self.clear_mask
        ).pack(fill="x", padx=12, pady=4)

        ttk.Button(
            left,
            text="Inpaint — изменить закрашенную область",
            command=self.inpaint
        ).pack(fill="x", padx=12, pady=4)

        ttk.Button(
            left,
            text="Outpaint — расширить холст",
            command=self.outpaint
        ).pack(fill="x", padx=12, pady=4)

        ttk.Button(
            left,
            text="Image-to-image — изменить всё фото",
            command=self.img2img
        ).pack(fill="x", padx=12, pady=4)

        self.canvas = tk.Canvas(
            right,
            bg="#0d1018",
            highlightthickness=0
        )
        self.canvas.pack(
            fill="both",
            expand=True,
            padx=8,
            pady=8
        )

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.edit_info = tk.Label(
            right,
            text="Открой изображение. Рисуй маску мышью по области для Inpaint.",
            bg="#171b27",
            fg="#9aa8c7"
        )
        self.edit_info.pack(fill="x", padx=8, pady=8)

    def apply_edit_preset(self, event=None):
        presets = {
            "Одежда (повседневная)": (
                "replace only the masked clothing with stylish modern casual clothing, "
                "realistic fabric folds, natural fit, preserve the character, face, "
                "body proportions and pose"
            ),
            "Одежда (деловой стиль)": (
                "replace only the masked clothing with elegant business attire, "
                "tailored blazer and shirt, realistic premium fabric, preserve "
                "the character and body proportions"
            ),
            "Одежда (спорт)": (
                "replace only the masked clothing with modern athletic sportswear, "
                "realistic technical fabric, natural fit, preserve the character "
                "and body proportions"
            ),
            "Причёска": (
                "change only the masked hairstyle to a stylish realistic haircut, "
                "detailed individual hair strands, preserve face identity and facial features"
            ),
            "Фон / окружение": (
                "replace the masked background with a coherent detailed environment, "
                "matching camera perspective, depth and natural shadows"
            ),
            "Освещение": (
                "modify the masked lighting and shadows to create natural cinematic "
                "studio lighting, preserve objects and anatomy"
            ),
            "Поза / положение тела": (
                "adjust the masked body posture to a natural anatomically plausible pose, "
                "realistic joints and proportions, preserve the character design"
            ),
            "Фотореализм": (
                "convert the masked area to highly photorealistic detail, natural skin "
                "and material texture, realistic light and shadows"
            ),
            "Кинематографичный стиль": (
                "cinematic color grading, filmic contrast, atmospheric practical lighting, "
                "subtle depth of field"
            ),
            "Аниме / иллюстрация": (
                "transform the masked area into a polished anime illustration, "
                "clean linework, expressive but consistent design"
            )
        }

        choice = self.edit_preset.get()

        if choice in presets:
            self.edit_prompt.delete("1.0", "end")
            self.edit_prompt.insert("1.0", presets[choice])

    def pick(self, variable=None):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("All files", "*.*")
            ]
        )

        if path and variable is not None:
            variable.set(path)

        return path

    def pick_video(self, variable=None):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm"), ("All files", "*.*")])
        if path and variable is not None:
            variable.set(path)
        return path

    def pick_json(self, variable=None):
        path=filedialog.askopenfilename(filetypes=[("ComfyUI API workflow", "*.json"), ("JSON files", "*.json"), ("All files", "*.*")])
        if path and variable is not None: variable.set(path)
        return path

    def pick_image_preview(self, variable, preview_label):
        path=self.pick(variable)
        if path:
            try:
                im=Image.open(path).convert("RGB"); im.thumbnail((180,120),Image.Resampling.LANCZOS)
                photo=ImageTk.PhotoImage(im); preview_label.configure(image=photo,text="")
                preview_label.image=photo
            except Exception as exc:
                preview_label.configure(text=f"Не удалось показать фото: {exc}")
        return path

    def open_source(self):
        p = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp")
            ]
        )

        if not p:
            return

        self.source_path = Path(p)
        self.source = Image.open(p).convert("RGB")
        self.mask = Image.new("L", self.source.size, 0)
        self.show_canvas()

    def show_canvas(self):
        if self.source is None:
            return

        self.canvas.delete("all")

        w, h = self.source.size
        scale = min(900 / w, 700 / h, 1)

        self.display_size = (
            int(w * scale),
            int(h * scale)
        )
        self.display_scale = scale

        im = self.source.resize(
            self.display_size,
            Image.Resampling.LANCZOS
        )

        self.canvas_img = ImageTk.PhotoImage(im)
        self.canvas.config(
            width=self.display_size[0],
            height=self.display_size[1]
        )

        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.canvas_img
        )

        self.edit_info.config(
            text=f"{w}×{h} | Кисть: {self.brush.get()} px на исходном фото"
        )

    def paint(self, event):
        if self.mask is None:
            return

        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)
        r = max(2, int(self.brush.get() / self.display_scale))

        d = ImageDraw.Draw(self.mask)
        d.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=255
        )

        self.show_canvas()

        overlay = Image.new(
            "RGBA",
            self.display_size,
            (255, 0, 80, 0)
        )

        m = self.mask.resize(self.display_size)
        overlay.putalpha(
            m.point(lambda v: int(v * 0.35))
        )

        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.canvas_img
        )

        self.overlay_tk = ImageTk.PhotoImage(overlay)

        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.overlay_tk
        )

    def clear_mask(self):
        if self.source:
            self.mask = Image.new("L", self.source.size, 0)
            self.show_canvas()

    def _edit_args(self):
        if self.source is None:
            raise RuntimeError("Сначала открой исходное фото.")

        model = self.edit_model.get()

        if not model:
            raise RuntimeError(
                "В папке models\\Stable-diffusion нет доступных моделей."
            )

        return (
            model,
            self.edit_prompt.get("1.0", "end").strip(),
            self.edit_neg.get("1.0", "end").strip(),
            int(self.edit_steps.get()),
            float(self.edit_cfg.get()),
            float(self.edit_strength.get())
        )

    def inpaint(self):
        try:
            model, p, n, s, c, st = self._edit_args()

            if not p:
                raise RuntimeError("Введи prompt для Inpaint.")

            if self.mask.getbbox() is None:
                raise RuntimeError(
                    "Нарисуй маску кистью на области для Inpaint."
                )

            source_copy = self.source.copy()
            mask_copy = self.mask.copy()

            self._start(
                lambda: run_generation(
                    "inpaint",
                    model,
                    p,
                    n,
                    source_copy,
                    mask_copy,
                    s,
                    c,
                    st,
                    0,
                    0,
                    -1,
                    self.done_edit
                )
            )

        except Exception as e:
            messagebox.showerror("Параметры", str(e))

    def img2img(self):
        try:
            model, p, n, s, c, st = self._edit_args()

            if not p:
                raise RuntimeError(
                    "Опиши, что именно нужно изменить."
                )

            source_copy = self.source.copy()
            st = max(0.05, min(st, 0.95))

            self._start(
                lambda: run_generation(
                    "img2img",
                    model,
                    p,
                    n,
                    source_copy,
                    None,
                    s,
                    c,
                    st,
                    0,
                    0,
                    -1,
                    self.done_edit
                )
            )

        except Exception as e:
            messagebox.showerror("Параметры", str(e))

    def done_edit(self, image, text):
        """Обновляет исходник редактора после завершения обработки."""
        def ui():
            if image is not None:
                self.source = image.copy()
                self.mask = Image.new("L", self.source.size, 0)
                self.show_canvas()
                self.preview_image = image.copy()

            self.edit_info.config(text=text)
            self.status.set(text.splitlines()[0][:160])
            self.progress.configure(value=100)
            log(text)

        self.master.after(0, ui)

    def outpaint(self):
        """Только расширяет холст. Генерация запускается отдельно через Inpaint."""
        if self.source is None:
            messagebox.showerror(
                "Outpaint",
                "Сначала открой исходное фото."
            )
            return

        original = self.source.copy().convert("RGB")
        w, h = original.size
        pad = 128

        new_w = ((w + 2 * pad + 7) // 8) * 8
        new_h = ((h + 2 * pad + 7) // 8) * 8

        left = (new_w - w) // 2
        top = (new_h - h) // 2

        canvas = Image.new(
            "RGB",
            (new_w, new_h),
            (127, 127, 127)
        )

        canvas.paste(original, (left, top))

        self.source = canvas
        self.mask = Image.new("L", (new_w, new_h), 0)
        self.source_path = None

        self.show_canvas()

        self.edit_info.config(
            text=(
                f"Холст расширен: {w}×{h} → {new_w}×{new_h}. "
                "Ничего не сгенерировано. Закрась белым новые области и нажми Inpaint."
            )
        )

    def _build_character(self):
        frame=tk.Frame(self.tab_character,bg="#171b27"); frame.pack(fill="both",expand=True,padx=18,pady=18)
        tk.Label(frame,text="CHARACTER STUDIO — локальная библиотека персонажей",bg="#171b27",fg="white",font=("Segoe UI",14,"bold")).pack(anchor="w",pady=8)
        self.character_name=self._entry(frame,"Имя персонажа","Character_01")
        self.character_prompt=self._field(frame,"Внешность / постоянный prompt","fictional adult character, consistent facial features, hairstyle, eye color, body proportions, clothing style",5)
        self.character_negative=self._field(frame,"Постоянный negative prompt","different face, inconsistent identity, distorted anatomy, extra fingers",2)
        self.character_reference=tk.StringVar()
        row=tk.Frame(frame,bg="#171b27"); row.pack(fill="x",pady=4)
        tk.Entry(row,textvariable=self.character_reference,bg="#202637",fg="white").pack(side="left",fill="x",expand=True)
        ttk.Button(row,text="Reference image…",command=lambda:self.pick(self.character_reference)).pack(side="right")
        buttons=tk.Frame(frame,bg="#171b27"); buttons.pack(anchor="w",pady=8)
        ttk.Button(buttons,text="Save character profile",command=self.save_character).pack(side="left",padx=4)
        ttk.Button(buttons,text="Load profile",command=self.load_character).pack(side="left",padx=4)
        ttk.Button(buttons,text="Use prompt in Image Studio",command=self.character_to_prompt).pack(side="left",padx=4)
        self.character_status=tk.Label(frame,text="Profiles stored locally in characters/ as JSON.",bg="#171b27",fg="#9aa8c7"); self.character_status.pack(anchor="w",pady=8)

    def save_character(self):
        import json
        name=re.sub(r"[^A-Za-z0-9_-]+","_",self.character_name.get().strip())
        if not name: raise RuntimeError("Укажи имя персонажа.")
        data={"name":name,"prompt":self.character_prompt.get("1.0","end").strip(),"negative_prompt":self.character_negative.get("1.0","end").strip(),"reference_image":self.character_reference.get().strip(),"created_at":datetime.now().isoformat(timespec="seconds")}
        path=CHARACTERS_DIR/f"{name}.json"; path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
        self.character_status.config(text=f"Saved: {path}")

    def load_character(self):
        import json
        path=filedialog.askopenfilename(initialdir=CHARACTERS_DIR,filetypes=[("Character profiles","*.json")])
        if not path:return
        data=json.loads(Path(path).read_text(encoding="utf-8"))
        self.character_name.delete(0,"end"); self.character_name.insert(0,data.get("name",""))
        self.character_prompt.delete("1.0","end"); self.character_prompt.insert("1.0",data.get("prompt",""))
        self.character_negative.delete("1.0","end"); self.character_negative.insert("1.0",data.get("negative_prompt",""))
        self.character_reference.set(data.get("reference_image","")); self.character_status.config(text=f"Loaded: {path}")

    def character_to_prompt(self):
        current=self.gen_prompt.get("1.0","end").strip(); profile=self.character_prompt.get("1.0","end").strip()
        self.gen_prompt.delete("1.0","end"); self.gen_prompt.insert("1.0",(profile+", "+current).strip(", "))
        neg=self.character_negative.get("1.0","end").strip()
        if neg:self.gen_neg.insert("end","\n"+neg)
        self.tabs.select(self.tab_gen)

    def _build_scene(self):
        frame=tk.Frame(self.tab_scene,bg="#171b27"); frame.pack(fill="both",expand=True,padx=18,pady=18)
        tk.Label(frame,text="SCENE STUDIO — сцены и storyboard",bg="#171b27",fg="white",font=("Segoe UI",14,"bold")).pack(anchor="w",pady=8)
        self.scene_project=self._entry(frame,"Project name","My_Scene_Project")
        self.scene_title=self._entry(frame,"Scene title","Scene 01")
        self.scene_prompt=self._field(frame,"Scene description / prompt","wide cinematic establishing shot, coherent environment, consistent character design",4)
        self.scene_duration=self._entry(frame,"Duration (seconds)","4")
        tk.Label(frame,text="Checkpoint for scene frames",bg="#171b27",fg="#cbd5e1").pack(anchor="w")
        self.scene_model=self.model_box(frame); self.scene_model.pack(fill="x",pady=3)
        self.scene_steps=self._entry(frame,"Generation steps","20")
        self.scene_list=tk.Listbox(frame,height=10,bg="#202637",fg="white",selectmode="extended")
        self.scene_list.pack(fill="both",expand=True,pady=8)
        row=tk.Frame(frame,bg="#171b27"); row.pack(anchor="w")
        ttk.Button(row,text="Add scene",command=self.add_scene).pack(side="left",padx=3)
        ttk.Button(row,text="Remove selected",command=self.remove_scene).pack(side="left",padx=3)
        ttk.Button(row,text="Save storyboard",command=self.save_storyboard).pack(side="left",padx=3)
        ttk.Button(row,text="Load storyboard",command=self.load_storyboard).pack(side="left",padx=3)
        ttk.Button(row,text="Generate storyboard images (not video)",command=self.generate_scene_frames).pack(side="left",padx=3)
        self.scene_data=[]

    def add_scene(self):
        title=self.scene_title.get().strip() or f"Scene {len(self.scene_data)+1:02d}"
        prompt=self.scene_prompt.get("1.0","end").strip(); duration=float(self.scene_duration.get())
        if duration<=0 or duration>600: raise RuntimeError("Длительность сцены должна быть от 0 до 600 секунд.")
        self.scene_data.append({"title":title,"prompt":prompt,"duration":duration})
        self.scene_list.insert("end",f"{len(self.scene_data):02d}. {title} — {duration:g}s | {prompt[:100]}")

    def remove_scene(self):
        selected=list(self.scene_list.curselection())
        for i in reversed(selected): self.scene_list.delete(i); self.scene_data.pop(i)
        self.scene_list.delete(0,"end")
        for i,s in enumerate(self.scene_data): self.scene_list.insert("end",f"{i+1:02d}. {s['title']} — {s['duration']:g}s | {s['prompt'][:100]}")

    def save_storyboard(self):
        import json
        name=re.sub(r"[^A-Za-z0-9_-]+","_",self.scene_project.get().strip()) or "storyboard"
        path=SCENES_DIR/f"{name}.json"; path.write_text(json.dumps({"project":name,"scenes":self.scene_data},ensure_ascii=False,indent=2),encoding="utf-8")
        self.status.set(f"Storyboard saved: {path}")

    def generate_scene_frames(self):
        if not self.scene_data: raise RuntimeError("Добавь хотя бы одну сцену в storyboard.")
        model=self.scene_model.get(); steps=int(self.scene_steps.get())
        if not model: raise RuntimeError("Выбери checkpoint.")
        scenes=[dict(x) for x in self.scene_data]
        def worker():
            made=[]
            for idx,scene in enumerate(scenes):
                prompt=scene.get("prompt","")
                if not prompt: continue
                pipe=load_pipeline(model,"txt2img")
                gen,seed=seeded_generator(-1)
                with torch.inference_mode():
                    image=pipe(prompt=prompt,negative_prompt="blurry, low quality, inconsistent scene",width=768,height=512,num_inference_steps=steps,guidance_scale=6.5,generator=gen).images[0]
                out=save_image(image,f"scene_{idx+1:02d}"); made.append(str(out))
                self.master.after(0,lambda i=idx+1,n=len(scenes):set_progress(i/max(1,n),f"Scene frame {i}/{n}"))
            cleanup()
            self.master.after(0,lambda:self.done(None,"Scene frames created:\n"+"\n".join(made)))
        self._start(worker)

    def load_storyboard(self):
        import json
        path=filedialog.askopenfilename(initialdir=SCENES_DIR,filetypes=[("Storyboards","*.json")])
        if not path:return
        data=json.loads(Path(path).read_text(encoding="utf-8")); self.scene_data=data.get("scenes",[])
        self.scene_project.delete(0,"end"); self.scene_project.insert(0,data.get("project",""))
        self.scene_list.delete(0,"end")
        for i,s in enumerate(self.scene_data): self.scene_list.insert("end",f"{i+1:02d}. {s.get('title','Scene')} — {float(s.get('duration',4)):g}s | {s.get('prompt','')[:100]}")

    def _build_timeline(self):
        frame=tk.Frame(self.tab_timeline,bg="#171b27"); frame.pack(fill="both",expand=True,padx=18,pady=18)
        tk.Label(frame,text="TIMELINE / EDITOR / EXPORT",bg="#171b27",fg="white",font=("Segoe UI",14,"bold")).pack(anchor="w",pady=8)
        tk.Label(frame,text="1) Добавь фото и видео. 2) Меняй порядок стрелками. 3) Для фото укажи длительность; видео по умолчанию берётся целиком. 4) Добавь отдельный аудиофайл при необходимости. 5) Нажми Export MP4.",bg="#171b27",fg="#cbd5e1",wraplength=1000,justify="left").pack(anchor="w",pady=4)
        self.timeline_list=tk.Listbox(frame,height=14,bg="#202637",fg="white",selectmode="extended"); self.timeline_list.pack(fill="both",expand=True,pady=8)
        row=tk.Frame(frame,bg="#171b27"); row.pack(anchor="w")
        ttk.Button(row,text="Add media",command=self.timeline_add).pack(side="left",padx=3)
        ttk.Button(row,text="Remove selected",command=self.timeline_remove).pack(side="left",padx=3)
        ttk.Button(row,text="Move up",command=lambda:self.timeline_move(-1)).pack(side="left",padx=3)
        ttk.Button(row,text="Move down",command=lambda:self.timeline_move(1)).pack(side="left",padx=3)
        ttk.Button(row,text="Set selected image duration",command=self.timeline_set_duration).pack(side="left",padx=3)
        self.timeline_duration=self._entry(frame,"Длительность добавляемых фото (сек.)","4")
        self.timeline_audio=tk.StringVar()
        audio=tk.Frame(frame,bg="#171b27"); audio.pack(fill="x",pady=4)
        tk.Entry(audio,textvariable=self.timeline_audio,bg="#202637",fg="white").pack(side="left",fill="x",expand=True)
        ttk.Button(audio,text="Audio track…",command=lambda:self.pick(self.timeline_audio)).pack(side="right")
        ttk.Button(frame,text="Export MP4 (FFmpeg / NVENC if available)",command=self.timeline_export).pack(anchor="w",pady=10)

    def timeline_add(self):
        paths=filedialog.askopenfilenames(filetypes=[("Media","*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.mov *.mkv *.avi *.webm")])
        try: still_duration=float(self.timeline_duration.get())
        except Exception: messagebox.showerror("Timeline","Длительность фото должна быть числом."); return
        if still_duration <= 0: messagebox.showerror("Timeline","Длительность фото должна быть больше нуля."); return
        for path in paths:
            is_image=Path(path).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
            item={"path":path,"duration":still_duration if is_image else None}
            self.timeline_items.append(item)
            label=f"{Path(path).name} — {still_duration:g}s" if is_image else f"{Path(path).name} — full video"
            self.timeline_list.insert("end",label)

    def timeline_set_duration(self):
        selected=self.timeline_list.curselection()
        if len(selected)!=1: messagebox.showinfo("Timeline","Выбери один клип."); return
        idx=selected[0]; item=self.timeline_items[idx]
        if Path(item["path"]).suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            messagebox.showinfo("Timeline","Для видео сохраняется исходная длительность. Редактирование обрезки добавим отдельным инструментом."); return
        try: duration=float(self.timeline_duration.get())
        except Exception: messagebox.showerror("Timeline","Введи число секунд."); return
        if duration<=0: messagebox.showerror("Timeline","Длительность должна быть больше нуля."); return
        item["duration"]=duration
        self.timeline_list.delete(idx); self.timeline_list.insert(idx,f"{Path(item['path']).name} — {duration:g}s"); self.timeline_list.selection_set(idx)

    def timeline_remove(self):
        for i in reversed(self.timeline_list.curselection()): self.timeline_list.delete(i); self.timeline_items.pop(i)

    def timeline_move(self,delta):
        sel=self.timeline_list.curselection()
        if len(sel)!=1:return
        i=sel[0]; j=i+delta
        if not 0<=j<len(self.timeline_items):return
        self.timeline_items[i],self.timeline_items[j]=self.timeline_items[j],self.timeline_items[i]
        vals=[(f"{Path(x['path']).name} — {x['duration']:g}s" if x.get("duration") is not None else f"{Path(x['path']).name} — full video") for x in self.timeline_items]
        self.timeline_list.delete(0,"end")
        for v in vals:self.timeline_list.insert("end",v)
        self.timeline_list.selection_set(j)

    def timeline_export(self):
        if not self.timeline_items: raise RuntimeError("Timeline пуст. Добавь изображения или видео.")
        path=filedialog.asksaveasfilename(initialdir=OUTPUT,defaultextension=".mp4",filetypes=[("MP4 video","*.mp4")])
        if not path:return
        items=[dict(x) for x in self.timeline_items]; audio=self.timeline_audio.get().strip() or None
        self._start(lambda: self.done(None,f"Timeline export: {export_timeline(items,Path(path),audio)}"))

    def _build_comfy(self):
        frame=tk.Frame(self.tab_comfy,bg="#171b27"); frame.pack(fill="both",expand=True,padx=18,pady=18)
        tk.Label(frame,text="COMFYUI LOCAL WORKFLOW BRIDGE",bg="#171b27",fg="white",font=("Segoe UI",14,"bold")).pack(anchor="w",pady=8)
        tk.Label(frame,text="AETHER остаётся главным интерфейсом, а ComfyUI выполняет сложные графы: ControlNet, IP-Adapter, LoRA, видео-workflows и другие custom nodes. Запусти ComfyUI отдельно на этом компьютере.",bg="#171b27",fg="#cbd5e1",wraplength=1000,justify="left").pack(anchor="w",pady=5)
        self.comfy_url=self._entry(frame,"ComfyUI local server URL","http://127.0.0.1:8188")
        self.comfy_workflow=tk.StringVar()
        row=tk.Frame(frame,bg="#171b27"); row.pack(fill="x",pady=6)
        tk.Entry(row,textvariable=self.comfy_workflow,bg="#202637",fg="white").pack(side="left",fill="x",expand=True)
        ttk.Button(row,text="Выбрать API workflow JSON…",command=lambda:self.pick_json(self.comfy_workflow)).pack(side="right")
        buttons=tk.Frame(frame,bg="#171b27"); buttons.pack(anchor="w",pady=8)
        ttk.Button(buttons,text="Проверить ComfyUI",command=self.comfy_check).pack(side="left",padx=4)
        ttk.Button(buttons,text="Запустить workflow",command=self.comfy_run).pack(side="left",padx=4)
        tk.Label(frame,text="Важно: нужен именно API-format JSON из ComfyUI (Save / Export API format), а не обычный workflow JSON. Выводы будут скопированы в output_media. На RTX 3060 Laptop 6 GB выбирай low-VRAM workflows; тяжёлые Wan 14B и большие LTX-модели не подходят.",bg="#171b27",fg="#fbbf24",wraplength=1000,justify="left").pack(anchor="w",pady=8)
        self.comfy_status=tk.Label(frame,text="Ожидание локального ComfyUI…",bg="#171b27",fg="#9aa8c7",wraplength=1000,justify="left"); self.comfy_status.pack(anchor="w",pady=8)

    def comfy_check(self):
        base=self.comfy_url.get().strip().rstrip("/")
        try:
            parsed=urllib.parse.urlparse(base)
            if parsed.hostname not in ("127.0.0.1","localhost","::1"): raise RuntimeError("Разрешён только локальный ComfyUI на localhost.")
            with urllib.request.urlopen(base+"/system_stats",timeout=5) as response:
                data=json.loads(response.read().decode("utf-8"))
            self.comfy_status.config(text="ComfyUI подключён. System stats: "+json.dumps(data,ensure_ascii=False)[:1200])
        except Exception as exc:
            self.comfy_status.config(text=f"ComfyUI не отвечает: {exc}")
            messagebox.showerror("ComfyUI",str(exc))

    def comfy_run(self):
        workflow=self.comfy_workflow.get().strip()
        if not workflow: messagebox.showerror("ComfyUI","Выбери API workflow JSON."); return
        self._start(lambda:submit_comfyui_workflow(self.comfy_url.get(),workflow,self.done))

    def _build_system(self):
        frame=tk.Frame(self.tab_system,bg="#171b27"); frame.pack(fill="both",expand=True,padx=18,pady=18)
        tk.Label(frame,text="MODEL MANAGER / SYSTEM CONTROL",bg="#171b27",fg="white",font=("Segoe UI",14,"bold")).pack(anchor="w",pady=8)
        self.system_text=tk.Text(frame,bg="#202637",fg="#dbeafe",height=22,wrap="word"); self.system_text.pack(fill="both",expand=True,pady=8)
        row=tk.Frame(frame,bg="#171b27"); row.pack(anchor="w")
        ttk.Button(row,text="Refresh diagnostics",command=self.refresh_system).pack(side="left",padx=3)
        ttk.Button(row,text="Rescan models",command=self.refresh_models).pack(side="left",padx=3)
        ttk.Button(row,text="Clear GPU cache",command=lambda:cleanup(drop_pipeline=False)).pack(side="left",padx=3)
        ttk.Button(row,text="Open output folder",command=self.open_output_folder).pack(side="left",padx=3)
        ttk.Button(row,text="Open models folder",command=self.open_models_folder).pack(side="left",padx=3)
        self.refresh_system()

    def open_models_folder(self):
        try:
            if hasattr(os,"startfile"): os.startfile(str(MODELS))
            else: subprocess.Popen(["explorer",str(MODELS)])
        except Exception as exc: messagebox.showerror("Models folder",str(exc))

    def open_output_folder(self):
        try:
            if hasattr(os,"startfile"): os.startfile(str(OUTPUT))
            else: subprocess.Popen(["explorer",str(OUTPUT)])
        except Exception as exc: messagebox.showerror("Output folder",str(exc))

    def refresh_system(self):
        try: ffmpeg_status=get_ffmpeg()
        except Exception: ffmpeg_status="not detected"
        lines=[f"Project: {ROOT}",f"Python: {sys.version.split()[0]}",f"PyTorch: {torch.__version__}",f"CUDA available: {torch.cuda.is_available()}",f"Device: {DEVICE}",f"Checkpoints: {len(list_checkpoints())}",f"LoRA: {len(scan_files(LORA_DIR))}",f"VAE: {len(scan_files(VAE_DIR))}",f"Embeddings: {len(scan_files(EMBEDDINGS_DIR))}",f"IP-Adapter weights: {len(scan_files(IP_ADAPTER_DIR))}",f"ControlNet: {len(list_controlnet_models())}",f"Motion adapters/LoRAs: {len(scan_files(MOTION))}",f"Voice models: {len(scan_files(VOICE_MODELS,('.onnx',)))}",f"FFmpeg: {ffmpeg_status}",f"FaceSwap: InsightFace + inswapper_128 (local)",f"FaceFusion: {'installed' if (TOOLS_DIR/'facefusion'/'facefusion.py').is_file() else 'optional / not installed'}",f"Lip-sync: {'Wav2Lip installed' if (TOOLS_DIR/'Wav2Lip'/'inference.py').is_file() else 'optional / weights missing'}",f"RVC: {'CLI installed' if (TOOLS_DIR/'RVC'/'infer_cli.py').is_file() else 'optional / CLI wrapper missing'}"]
        if torch.cuda.is_available():
            try: lines += [f"GPU: {torch.cuda.get_device_name(0)}",f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB",f"VRAM allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB",f"VRAM reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB"]
            except Exception: pass
        self.system_text.delete("1.0","end"); self.system_text.insert("1.0","\n".join(lines))

    def refresh_models(self):
        vals=list_checkpoints()
        for cb_name in ("gen_model","edit_model","video_model","cn_model"):
            cb=getattr(self,cb_name,None)
            if cb is not None:
                current=cb.get(); cb.configure(values=vals)
                if current in vals: cb.set(current)
                elif vals: cb.current(0)
        self.refresh_system(); self.status.set(f"Models rescanned: {len(vals)}")

    def _build_video(self):
        left = tk.Frame(
            self.tab_video,
            bg="#171b27",
            width=510
        )
        left.pack(
            side="left",
            fill="y",
            padx=8,
            pady=8
        )
        left.pack_propagate(False)

        right = tk.Frame(
            self.tab_video,
            bg="#171b27"
        )
        right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=8,
            pady=8
        )

        tk.Label(
            left,
            text="AnimateDiff: только SD 1.5 + motion adapter",
            bg="#171b27",
            fg="#fbbf24"
        ).pack(anchor="w", padx=12, pady=8)

        self.video_model = self.model_box(left)
        self.video_model.pack(fill="x", padx=12)

        motion_dirs=sorted({str(p.parent.relative_to(ROOT)) for p in MOTION.rglob("config.json")})
        self.motion_box = ttk.Combobox(left, values=motion_dirs, state="readonly")
        if motion_dirs: self.motion_box.set(motion_dirs[0])
        self.motion_box.pack(fill="x", padx=12, pady=8)
        tk.Label(left,text="Camera-motion LoRA (optional)",bg="#171b27",fg="#cbd5e1").pack(anchor="w",padx=12)
        self.motion_lora_box=ttk.Combobox(left,values=[""]+[str(p.relative_to(ROOT)) for p in scan_files(MOTION/"motion-lora",(".safetensors",))],state="readonly")
        self.motion_lora_box.pack(fill="x",padx=12,pady=4)

        self.video_prompt = self._field(
            left,
            "Prompt",
            "cinematic camera movement, natural motion, detailed scene",
            4
        )

        self.video_neg = self._field(
            left,
            "Negative prompt",
            "flicker, jitter, distorted anatomy, blurry",
            2
        )

        self.video_steps = self._entry(left, "Steps", "20")
        self.video_cfg = self._entry(left, "CFG", "7")
        self.video_frames = self._entry(left, "Frames (6 GB VRAM: 8–16)", "12")
        self.video_fps = self._entry(left, "FPS", "8")

        ttk.Button(
            left,
            text="Создать видео (AnimateDiff T2V)",
            command=self.generate_video
        ).pack(fill="x", padx=12, pady=8)

        ttk.Separator(left).pack(fill="x", padx=12, pady=8)

        tk.Label(
            left,
            text="SVD-XT: изображение → видео",
            bg="#171b27",
            fg="#7dd3fc",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=12, pady=4)

        self.svd_path = tk.StringVar()

        svd_row = tk.Frame(left, bg="#171b27")
        svd_row.pack(fill="x", padx=12, pady=4)

        tk.Entry(
            svd_row,
            textvariable=self.svd_path,
            bg="#202637",
            fg="white"
        ).pack(side="left", fill="x", expand=True)

        ttk.Button(
            svd_row,
            text="Фото…",
            command=lambda: self.pick(self.svd_path)
        ).pack(side="right")

        self.svd_prompt = self._field(
            left,
            "SVD motion notes (SVD-XT is image-conditioned; notes are saved as metadata)",
            "A fictional adult character gently turns toward the camera, smiles naturally, hair and clothes move slightly in a soft breeze, smooth realistic motion, stable face, consistent anatomy",
            4
        )
        self.svd_negative = self._field(
            left,
            "SVD Negative prompt",
            "flicker, jitter, sudden camera cuts, face distortion, extra limbs, deformed hands, unstable identity, blurry frames",
            2
        )
        self.svd_duration = self._entry(left, "Длительность видео (секунды, 0.5–180)", "4")
        self.svd_fps = self._entry(left, "FPS видео (3–15; стандартно 7)", "7")
        self.svd_motion = self._entry(left, "Сила движения (Motion bucket 0–255)", "100")
        self.svd_noise = self._entry(left, "Изменение исходного кадра (noise 0.00–0.50; ниже = стабильнее лицо)", "0.03")
        self.svd_steps = self._entry(left, "Шаги SVD-XT (10–60)", "25")
        self.svd_decode = self._entry(left, "Decode chunk (1–8; 1 = меньше VRAM)", "2")
        self.svd_seed = self._entry(left, "Seed (-1 = случайный)", "-1")

        ttk.Button(
            left,
            text="СОЗДАТЬ ВИДЕО SVD-XT",
            command=self.generate_svd
        ).pack(fill="x", padx=12, pady=8)
        tk.Label(left,text="Длинное видео собирается из последовательных сегментов до 25 кадров. Возможен постепенный дрейф лица. Текстовые заметки SVD-XT не понимает как prompt — это ограничение самой модели.",bg="#171b27",fg="#fbbf24",wraplength=470,justify="left").pack(anchor="w",padx=12,pady=4)

        tk.Label(
            right,
            text=(
                "Результаты сохраняются в output_media. "
                "SVD-XT требует полную локальную Diffusers-модель."
            ),
            bg="#171b27",
            fg="#cbd5e1",
            wraplength=600
        ).pack(expand=True, padx=30)

    def generate_video(self):
        model = self.video_model.get()
        motion = self.motion_box.get()

        if not model or not motion:
            messagebox.showerror(
                "Видео",
                "Нужны SD 1.5 checkpoint и motion adapter."
            )
            return

        p = self.video_prompt.get("1.0", "end").strip()
        n = self.video_neg.get("1.0", "end").strip()
        s = int(self.video_steps.get())
        c = float(self.video_cfg.get())
        f = int(self.video_frames.get())
        fps = int(self.video_fps.get())

        if not 4 <= f <= 24:
            messagebox.showerror("Видео", "Число кадров: 4–24."); return
        if not 1 <= fps <= 30 or s < 1 or s > 60 or not 1 <= c <= 20:
            messagebox.showerror("Видео", "Проверь Steps (1–60), CFG (1–20), FPS (1–30)."); return

        motion_lora=self.motion_lora_box.get().strip() or None
        self._start(
            lambda: make_animatediff(model,motion,p,n,s,c,f,fps,None,self.done,motion_lora)
        )

    def generate_svd(self):
        try:
            path = self.svd_path.get().strip()
            prompt = self.svd_prompt.get("1.0", "end").strip()
            negative = self.svd_negative.get("1.0", "end").strip()
            duration = float(self.svd_duration.get())
            fps = int(self.svd_fps.get())
            motion = int(self.svd_motion.get())
            noise = float(self.svd_noise.get())
            steps = int(self.svd_steps.get())
            decode_chunk = int(self.svd_decode.get())
            seed = int(self.svd_seed.get())
            if not path or not Path(path).is_file():
                raise RuntimeError("Сначала выбери исходное изображение.")
            if not 0.5 <= duration <= 180:
                raise RuntimeError("Длительность должна быть от 0.5 до 180 секунд.")
            if not 3 <= fps <= 15 or not 0 <= motion <= 255:
                raise RuntimeError("FPS: 3–15; motion bucket: 0–255.")
            if not 0 <= noise <= 0.5 or not 10 <= steps <= 60 or not 1 <= decode_chunk <= 8:
                raise RuntimeError("Проверь noise 0–0.5, steps 10–60 и decode chunk 1–8.")
            self._start(lambda: make_svd_video(path,prompt,negative,duration,fps,motion,noise,steps,decode_chunk,self.done,seed))
        except Exception as e:
            messagebox.showerror("SVD-XT", str(e))

    def _build_face(self):
        left = tk.Frame(
            self.tab_face,
            bg="#171b27",
            width=560
        )
        left.pack(
            side="left",
            fill="y",
            padx=8,
            pady=8
        )
        left.pack_propagate(False)

        right = tk.Frame(
            self.tab_face,
            bg="#171b27"
        )
        right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=8,
            pady=8
        )

        tk.Label(
            left,
            text="PORTRAIT LAB — восстановление и улучшение",
            bg="#171b27",
            fg="white",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=12, pady=12)

        ttk.Separator(left).pack(fill="x", padx=12, pady=8)

        tk.Label(
            left,
            text="FACE SWAP — источник → целевое фото",
            bg="#171b27",
            fg="#7dd3fc",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=12, pady=4)

        self.swap_source = tk.StringVar()
        self.swap_target = tk.StringVar()

        for label, var in (
            ("Фото лица-источника", self.swap_source),
            ("Целевое фото", self.swap_target)
        ):
            row_swap = tk.Frame(left, bg="#171b27")
            row_swap.pack(fill="x", padx=12, pady=3)

            tk.Entry(
                row_swap,
                textvariable=var,
                bg="#202637",
                fg="white"
            ).pack(side="left", fill="x", expand=True)

            ttk.Button(
                row_swap,
                text="Выбрать…",
                command=lambda v=var: self.pick(v)
            ).pack(side="right")

        self.swap_consent = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            left,
            text="У меня есть разрешение использовать лицо-источник и целевое фото",
            variable=self.swap_consent
        ).pack(anchor="w", padx=12, pady=4)

        ttk.Button(
            left,
            text="Выполнить Face Swap",
            command=self.face_swap
        ).pack(fill="x", padx=12, pady=6)

        ttk.Separator(left).pack(fill="x", padx=12, pady=8)
        tk.Label(left,text="VIDEO FACE SWAP — source photo → target video",bg="#171b27",fg="#7dd3fc").pack(anchor="w",padx=12)
        self.swap_video_path=tk.StringVar()
        video_row=tk.Frame(left,bg="#171b27"); video_row.pack(fill="x",padx=12,pady=4)
        tk.Entry(video_row,textvariable=self.swap_video_path,bg="#202637",fg="white").pack(side="left",fill="x",expand=True)
        ttk.Button(video_row,text="Выбрать видео…",command=lambda:self.pick_video(self.swap_video_path)).pack(side="right")
        self.swap_video_consent=tk.BooleanVar(value=False)
        ttk.Checkbutton(left,text="Есть разрешение на использование лица-источника и целевого видео",variable=self.swap_video_consent).pack(anchor="w",padx=12,pady=3)
        self.video_source_face_index=self._entry(left,"Номер лица на фото-источнике (слева направо, 1 = первое)","1")
        self.video_target_face_index=self._entry(left,"Номер лица в каждом кадре видео (слева направо, 1 = первое)","1")
        ttk.Button(left,text="Заменить выбранное лицо в видео (сохранить звук)",command=self.face_swap_video).pack(fill="x",padx=12,pady=5)
        ttk.Button(left,text="Run optional local FaceFusion",command=self.facefusion_run).pack(fill="x",padx=12,pady=4)

        tk.Label(
            left,
            text="Нужны models\\insightface\\inswapper_128.onnx и buffalo_l.",
            bg="#171b27",
            fg="#9aa8c7",
            wraplength=500,
            justify="left"
        ).pack(anchor="w", padx=12, pady=4)

        ttk.Separator(left).pack(fill="x", padx=12, pady=8)

        self.portrait_path = tk.StringVar()

        row = tk.Frame(left, bg="#171b27")
        row.pack(fill="x", padx=12, pady=6)

        tk.Entry(
            row,
            textvariable=self.portrait_path,
            bg="#202637",
            fg="white"
        ).pack(side="left", fill="x", expand=True)

        ttk.Button(
            row,
            text="Выбрать фото",
            command=lambda: self.pick(self.portrait_path)
        ).pack(side="right")

        ttk.Button(
            left,
            text="Восстановить лицо (GFPGAN)",
            command=self.portrait_restore
        ).pack(fill="x", padx=12, pady=8)

        ttk.Button(
            left,
            text="Открыть фото в редакторе для смены одежды / причёски / фона",
            command=self.portrait_to_editor
        ).pack(fill="x", padx=12, pady=8)

        tk.Label(
            right,
            text=(
                "GFPGAN восстанавливает детали лица, не меняя личность. "
                "Для изменения причёски, одежды, фона и художественного стиля "
                "открой фотографию в редакторе, закрась нужную область маской "
                "и используй Inpainting с подробным prompt."
            ),
            bg="#171b27",
            fg="#cbd5e1",
            wraplength=600,
            justify="left"
        ).pack(expand=True, padx=35)

    def face_swap(self):
        source = self.swap_source.get().strip()
        target = self.swap_target.get().strip()

        if not self.swap_consent.get():
            messagebox.showwarning(
                "Face Swap",
                "Подтверди, что у тебя есть разрешение на использование этих изображений."
            )
            return

        self._start(
            lambda: run_face_swap(
                source,
                target,
                self.done
            )
        )

    def facefusion_run(self):
        source=self.swap_source.get().strip(); target=self.swap_video_path.get().strip() or self.swap_target.get().strip()
        if not self.swap_consent.get() and not self.swap_video_consent.get():
            messagebox.showwarning("FaceFusion","Подтверди разрешение на использование исходного лица и целевого медиа."); return
        self._start(lambda:run_facefusion(source,target,self.done))

    def face_swap_video(self):
        source=self.swap_source.get().strip(); target=self.swap_video_path.get().strip()
        if not self.swap_video_consent.get():
            messagebox.showwarning("FaceSwap video","Подтверди разрешение на использование лица и видео."); return
        try:
            source_idx=int(self.video_source_face_index.get())
            target_idx=int(self.video_target_face_index.get())
            if source_idx < 1 or target_idx < 1: raise ValueError("Номер лица начинается с 1.")
        except Exception as exc:
            messagebox.showerror("FaceSwap video",str(exc)); return
        self._start(lambda:run_video_face_swap(source,target,self.done,source_idx,target_idx))

    def portrait_restore(self):
        p = self.portrait_path.get()

        if not p or not Path(p).is_file():
            messagebox.showerror(
                "Portrait Lab",
                "Сначала выбери фотографию."
            )
            return

        self._start(
            lambda: restore_portrait(p, self.done)
        )

    def portrait_to_editor(self):
        p = self.portrait_path.get()

        if not p or not Path(p).is_file():
            messagebox.showerror(
                "Portrait Lab",
                "Сначала выбери фотографию."
            )
            return

        self.source_path = Path(p)
        self.source = Image.open(p).convert("RGB")
        self.mask = Image.new("L", self.source.size, 0)
        self.show_canvas()
        self.tabs.select(self.tab_edit)

    def _build_tools(self):
        frame = tk.Frame(
            self.tab_tools,
            bg="#171b27"
        )
        frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

        tk.Label(
            frame,
            text="Улучшение изображения",
            bg="#171b27",
            fg="white",
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=10)

        ttk.Button(
            frame,
            text="Открыть изображение и увеличить ×2 / RealESRGAN ×4",
            command=self.upscale
        ).pack(anchor="w", pady=8)

        info = (
            f"Рабочая папка: {ROOT}\n"
            f"Устройство: {DEVICE} | PyTorch: {torch.__version__}\n"
            f"CUDA: {torch.cuda.is_available()}\n\n"
            "Модели: models/Stable-diffusion/ — .safetensors/.ckpt "
            "и полные Diffusers-папки с model_index.json.\n"
            "ControlNet: Canny/Tile; OpenPose/Depth требуют controlnet-aux + annotator weights.\n"
            "LoRA/VAE/Embeddings/IP-Adapter: optional local folders models/loras, vae, embeddings, ip_adapter.\n"
            "Portrait Lab: GFPGANv1.4.pth в корне (восстановление лица)\n"
            "AnimateDiff: models/animatediff/ (совместимый SD1.5 motion adapter)\n"
            "Piper TTS: pip install piper-tts; voice pair in models/voice/\n"
            "Результаты: output_media/\n\n"
            "RTX 3060 Laptop 6 GB: начни с 512–768 px, 20–30 steps; "
            "видео 8–12 кадров.\n"
            "CPU offload экономит VRAM, но может заметно замедлять генерацию.\n"
            "FLUX GGUF и Wan требуют отдельных специализированных backend. "
            "SVD-XT image-to-video доступен при полной локальной Diffusers-модели.\n"
            "Wav2Lip требует tools/Wav2Lip + checkpoint; RVC требует локальный tools/RVC/infer_cli.py.\n"
            "FaceFusion подключается при наличии tools/facefusion/facefusion.py.\n"
            "FLUX GGUF отображается в каталоге, но требует отдельного GGUF backend."
        )

        tk.Label(
            frame,
            text=info,
            bg="#171b27",
            fg="#cbd5e1",
            justify="left",
            anchor="nw",
            font=("Consolas", 10)
        ).pack(fill="both", expand=True, pady=10)

    def _build_controlnet(self):
        left = tk.Frame(
            self.tab_control,
            bg="#171b27",
            width=510
        )
        left.pack(
            side="left",
            fill="y",
            padx=8,
            pady=8
        )
        left.pack_propagate(False)

        right = tk.Frame(
            self.tab_control,
            bg="#171b27"
        )
        right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=8,
            pady=8
        )

        tk.Label(
            left,
            text="ControlNet Canny — SD 1.5",
            bg="#171b27",
            fg="white",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", padx=12, pady=10)

        self.cn_model = self.model_box(left)
        self.cn_model.pack(fill="x", padx=12, pady=4)

        self.cn_mode = ttk.Combobox(left,values=["Canny","OpenPose","Depth","Tile"],state="readonly")
        self.cn_mode.set("Canny"); self.cn_mode.pack(fill="x",padx=12,pady=3)
        self.cn_mode.bind("<<ComboboxSelected>>",self._select_controlnet_for_mode)
        self.cn_control = ttk.Combobox(
            left,
            values=list_controlnet_models(),
            state="readonly",
            width=55
        )

        cn_values=list(self.cn_control["values"])
        canny_default=next((i for i,v in enumerate(cn_values) if "canny" in v.lower()),0)
        if cn_values:
            self.cn_control.current(canny_default)

        self.cn_control.pack(fill="x", padx=12, pady=4)

        self.cn_image = tk.StringVar()

        row = tk.Frame(self.tab_control, bg="#171b27")
        row.pack(fill="x", padx=12, pady=5)

        tk.Entry(
            row,
            textvariable=self.cn_image,
            bg="#202637",
            fg="white"
        ).pack(side="left", fill="x", expand=True)

        ttk.Button(
            row,
            text="Фото…",
            command=lambda: self.pick(self.cn_image)
        ).pack(side="right")

        self.cn_prompt = self._field(
            left,
            "Prompt",
            "high quality, detailed, preserve composition",
            4
        )

        self.cn_neg = self._field(
            left,
            "Negative prompt",
            "blurry, low quality, distorted anatomy",
            2
        )

        self.cn_steps = self._entry(left, "Steps", "20")
        self.cn_cfg = self._entry(left, "CFG", "6.5")
        self.cn_strength = self._entry(left, "Control strength", "0.8")

        ttk.Button(
            left,
            text="Generate with ControlNet Canny",
            command=self.generate_controlnet
        ).pack(fill="x", padx=12, pady=12)

        tk.Label(
            right,
            text=(
                "Canny — контуры; Tile — исходная карта; OpenPose — скелет/поза; Depth — карта глубины.\n"
                "OpenPose/Depth запускаются при наличии локальных controlnet-aux annotator weights."
            ),
            bg="#171b27",
            fg="#cbd5e1",
            justify="left"
        ).pack(expand=True, padx=30)

    def _select_controlnet_for_mode(self,event=None):
        mode=self.cn_mode.get().lower()
        token={"canny":"canny","openpose":"openpose","depth":"depth","tile":"tile"}.get(mode,"canny")
        vals=list_controlnet_models()
        match=next((v for v in vals if token in Path(v).name.lower()),None)
        if match:
            self.cn_control.set(match)

    def generate_controlnet(self):
        try:
            model = self.cn_model.get()
            control = self.cn_control.get()
            image = self.cn_image.get().strip()

            if not model or not control or not image:
                raise RuntimeError(
                    "Выбери базовый checkpoint SD 1.5 (например, Realistic Vision), совместимый ControlNet Canny и исходное фото."
                )

            prompt = self.cn_prompt.get("1.0", "end").strip()
            negative = self.cn_neg.get("1.0", "end").strip()

            steps = int(self.cn_steps.get())
            cfg = float(self.cn_cfg.get())
            strength = float(self.cn_strength.get())

            if not prompt:
                raise RuntimeError("Введи prompt.")

            mode = self.cn_mode.get()
            self._start(
                lambda: run_controlnet_mode(
                    mode, model, control, image, prompt, negative,
                    steps, cfg, strength, self.done
                )
            )

        except Exception as e:
            messagebox.showerror("ControlNet", str(e))

    def _build_voice(self):
        frame = tk.Frame(
            self.tab_voice,
            bg="#171b27"
        )
        frame.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=18
        )

        tk.Label(
            frame,
            text="Piper — локальный синтез речи",
            bg="#171b27",
            fg="white",
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=8)

        self.voice_text = self._field(
            frame,
            "Текст для озвучивания",
            "Hello! This is a fictional AI character voice.",
            7
        )

        tk.Label(
            frame,
            text="Локальная голосовая модель (.onnx)",
            bg="#171b27",
            fg="#cbd5e1"
        ).pack(anchor="w", pady=(10, 3))

        self.voice_model = ttk.Combobox(
            frame,
            values=[str(p) for p in VOICE_MODELS.glob("*.onnx")],
            state="readonly"
        )

        if self.voice_model["values"]:
            self.voice_model.current(0)

        self.voice_model.pack(fill="x", pady=4)

        ttk.Button(
            frame,
            text="Обновить список моделей",
            command=self.refresh_voice_models
        ).pack(anchor="w", pady=4)

        ttk.Button(
            frame,
            text="Создать WAV",
            command=self.generate_voice
        ).pack(anchor="w", pady=10)

        ttk.Separator(frame).pack(fill="x",pady=12)
        tk.Label(frame,text="LIP-SYNC — видео + готовая речь",bg="#171b27",fg="#7dd3fc",font=("Segoe UI",11,"bold")).pack(anchor="w",pady=4)
        self.lipsync_video=tk.StringVar(); self.lipsync_audio=tk.StringVar()
        for label,var,types in (("Face video",self.lipsync_video,("Video","*.mp4 *.mov *.mkv")),("Audio file",self.lipsync_audio,("Audio","*.wav *.mp3 *.m4a"))):
            rr=tk.Frame(frame,bg="#171b27"); rr.pack(fill="x",pady=3)
            tk.Label(rr,text=label,bg="#171b27",fg="#cbd5e1",width=14,anchor="w").pack(side="left")
            tk.Entry(rr,textvariable=var,bg="#202637",fg="white").pack(side="left",fill="x",expand=True)
            ttk.Button(rr,text="Browse…",command=lambda v=var:self.pick(v)).pack(side="right")
        ttk.Button(frame,text="Create lip-sync video (local Wav2Lip)",command=self.lipsync_run).pack(anchor="w",pady=6)
        ttk.Separator(frame).pack(fill="x",pady=10)
        tk.Label(frame,text="RVC — local voice conversion (optional)",bg="#171b27",fg="#7dd3fc",font=("Segoe UI",11,"bold")).pack(anchor="w")
        self.rvc_audio=tk.StringVar(); self.rvc_model=tk.StringVar(); self.rvc_index=tk.StringVar()
        for label,var,types in (("Input audio",self.rvc_audio,("Audio","*.wav *.mp3 *.flac")),("RVC model",self.rvc_model,("RVC model","*.pth")),("Index (optional)",self.rvc_index,("Index","*.index"))):
            rr=tk.Frame(frame,bg="#171b27"); rr.pack(fill="x",pady=2)
            tk.Label(rr,text=label,bg="#171b27",fg="#cbd5e1",width=14,anchor="w").pack(side="left")
            tk.Entry(rr,textvariable=var,bg="#202637",fg="white").pack(side="left",fill="x",expand=True)
            ttk.Button(rr,text="Browse…",command=lambda v=var:self.pick(v)).pack(side="right")
        ttk.Button(frame,text="Run local RVC conversion",command=self.rvc_run).pack(anchor="w",pady=6)

        tk.Label(
            frame,
            text=(
                "Для Piper нужны piper.exe и пара файлов voice.onnx + "
                "voice.onnx.json.\n"
                "Используй синтетические голоса или голоса, на которые у тебя есть разрешение."
            ),
            bg="#171b27",
            fg="#9aa8c7",
            justify="left"
        ).pack(anchor="w", pady=8)

    def lipsync_run(self):
        video=self.lipsync_video.get().strip(); audio=self.lipsync_audio.get().strip()
        self._start(lambda:create_lipsync_video(video,audio,self.done))

    def rvc_run(self):
        audio=self.rvc_audio.get().strip(); model=self.rvc_model.get().strip(); index=self.rvc_index.get().strip()
        self._start(lambda:run_rvc_conversion(audio,model,index,self.done))

    def refresh_voice_models(self):
        vals = [
            str(p)
            for p in VOICE_MODELS.glob("*.onnx")
        ]

        self.voice_model.configure(values=vals)

        if vals:
            self.voice_model.current(0)

    def generate_voice(self):
        try:
            text_value = self.voice_text.get("1.0", "end").strip()
            model = self.voice_model.get().strip()

            self._start(
                lambda: synthesize_voice_piper(
                    text_value,
                    model,
                    self.done
                )
            )

        except Exception as e:
            messagebox.showerror("Voice Studio", str(e))

    def upscale(self):
        p = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.webp")
            ]
        )

        if not p:
            return

        img = Image.open(p).convert("RGB")

        self._start(
            lambda: upscale_image(img, self.done)
        )

    def generate(self):
        try:
            model = self.gen_model.get()

            if not model:
                raise RuntimeError(
                    "Нет моделей в models\\Stable-diffusion."
                )

            prompt = self.gen_prompt.get("1.0", "end").strip()
            style = self.gen_style.get()

            if style == "Аниме":
                prompt = (
                    "anime illustration, clean lineart, expressive character design, "
                    "high detail, " + prompt
                )
            elif style == "Фотореализм":
                prompt = (
                    "photorealistic, realistic lighting, natural materials "
                    "and skin texture, " + prompt
                )

            negative = self.gen_neg.get("1.0", "end").strip()

            steps = int(self.gen_steps.get())
            cfg = float(self.gen_cfg.get())
            w = int(self.gen_w.get())
            h = int(self.gen_h.get())
            seed = int(self.gen_seed.get())

            if not prompt:
                raise RuntimeError("Введи prompt.")
            if not 1 <= steps <= 100 or not 1.0 <= cfg <= 20.0:
                raise RuntimeError("Steps: 1–100; CFG: 1–20.")
            if seed < -1:
                raise RuntimeError("Seed должен быть -1 (random) или неотрицательным числом.")

            if not (256 <= w <= 1536 and 256 <= h <= 1536):
                raise RuntimeError(
                    "Размеры должны быть 256–1536."
                )

            w = (w // 8) * 8
            h = (h // 8) * 8

            if DEVICE == "cuda" and max(w, h) > 1024:
                raise RuntimeError(
                    "Для 6 GB VRAM используй максимум 1024 px по длинной стороне."
                )

            lora_rel = self.gen_lora.get().strip() or None
            lora_scale = float(self.gen_lora_weight.get())
            vae_rel = self.gen_vae.get().strip() or None
            embedding_rel = self.gen_embedding.get().strip() or None
            ip_adapter_rel = self.gen_ip_adapter.get().strip() or None
            ip_image_path = self.gen_ip_image.get().strip() or None
            self._start(
                lambda: run_generation(
                    "txt2img",
                    model,
                    prompt,
                    negative,
                    None,
                    None,
                    steps,
                    cfg,
                    1.0,
                    w,
                    h,
                    seed,
                    self.done,
                    lora_rel, lora_scale, vae_rel, embedding_rel, ip_adapter_rel, ip_image_path
                )
            )

        except Exception as e:
            messagebox.showerror("Параметры", str(e))

    def _start(self, fn):
        global BUSY

        if BUSY:
            messagebox.showinfo(
                "Задача выполняется",
                "Дождись завершения текущей GPU-задачи."
            )
            return

        BUSY = True
        self.progress.configure(value=0)
        self.status.set("Выполняется…")

        def worker():
            global BUSY
            succeeded=False
            try:
                fn()
                succeeded=True
            except Exception as e:
                cleanup(drop_pipeline=True)
                traceback.print_exc()
                self.master.after(0, lambda err=str(e): messagebox.showerror("Ошибка", err))
                log(f"Ошибка: {e}")
                self.master.after(0, lambda: self.status.set("Ошибка — смотри сообщение и консоль"))
            finally:
                BUSY=False
                if succeeded:
                    set_progress(1.0,"Задача завершена")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def done(self, image, text):
        def ui():
            if image is not None:
                self.preview_image = image.copy()
                im = image.copy()

                im.thumbnail(
                    (850, 720),
                    Image.Resampling.LANCZOS
                )

                self.tk_preview = ImageTk.PhotoImage(im)
                self.preview.configure(
                    image=self.tk_preview,
                    text=""
                )

            if hasattr(self, "output_text"):
                self.output_text.delete("1.0", "end")
                self.output_text.insert("1.0", text)

            self.status.set(text.splitlines()[0][:160])
            self.progress.configure(value=100)
            log(text)

        self.master.after(0, ui)


def main():
    global root, status_var

    root = None
    status_var = None

    if DEVICE == "cuda":
        log(
            f"CUDA обнаружена: {torch.cuda.get_device_name(0)}"
        )
    else:
        log(
            "CUDA не обнаружена. Генерация будет очень медленной на CPU."
        )

    app = tk.Tk()
    root = app

    Studio(app)
    app.mainloop()


if __name__ == "__main__":
    main()