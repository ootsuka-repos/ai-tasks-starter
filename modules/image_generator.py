import os
from typing import Optional
from PIL import Image
from abc import ABC, abstractmethod

try:
    import torch
    from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
    from compel import Compel, ReturnedEmbeddingsType
except ImportError as e:
    raise ImportError("diffusers, torch, pillow, compelがインストールされている必要があります。") from e

class ImageGeneratorInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, output_path: Optional[str] = None) -> Image.Image:
        pass

class StableDiffusionImageGenerator(ImageGeneratorInterface):
    def __init__(self):
        self.pipe = None
        self.compel_proc = None

    def _get_pipeline(self):
        if self.pipe is None:
            try:
                model_name = os.environ.get("IMAGE_GEN_MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0")
                hf_token = os.environ.get("HF_TOKEN")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    token=hf_token
                )
                self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    self.pipe.scheduler.config,
                    algorithm_type="dpmsolver++",
                    solver_order=3,
                    use_karras_sigmas=True,
                    lower_order_final=True,
                )
                if torch.cuda.is_available():
                    self.pipe = self.pipe.to(os.environ.get("PIPELINE_DEVICE_CUDA", "cuda"))
                else:
                    self.pipe = self.pipe.to(os.environ.get("PIPELINE_DEVICE_CPU", "cpu"))
            except Exception as e:
                raise RuntimeError(f"パイプライン初期化中にエラーが発生しました: {e}")
        if self.compel_proc is None:
            try:
                self.compel_proc = Compel(
                    truncate_long_prompts=False,
                    tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                    text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )
            except Exception as e:
                raise RuntimeError(f"Compel初期化中にエラーが発生しました: {e}")
        return self.pipe, self.compel_proc

    def generate(self, prompt: str, output_path: Optional[str] = None) -> Image.Image:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("プロンプトは空でない文字列で指定してください。")

        pipe, compel_proc = self._get_pipeline()
        try:
            negative_prompt = "(speech_bubble:2.0), (thought_bubble:2.0), (sex:2.0), (1boy:2.0), (penis:2.0), lowres, cropped, multiple_girls, 2girls, disembodied_hand, pov_hands, signature, watermark, username, artist_name, (simple_background:1.5), vibrator, (sex_toy:1.3), (uncensored:1.2), recording, text_focus"
            # サンプル通りリストで渡す
            prompt_embeds, pooled_prompt_embeds = compel_proc([prompt])
            negative_embeds, negative_pooled_embeds = compel_proc([negative_prompt])
            [prompt_embeds, negative_embeds] = compel_proc.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_embeds])
            result = pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                negative_pooled_prompt_embeds=negative_pooled_embeds,
                num_inference_steps=30,
                height=1024,
                width=1024,
                guidance_scale=4.5,
            )
            image = result.images[0]
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
            return image
        except Exception as e:
            raise RuntimeError(f"画像生成中にエラーが発生しました: {e}")

class ImageGenerationService:
    def __init__(self, generator: ImageGeneratorInterface):
        self.generator = generator

    def generate_image(self, prompt: str, output_path: Optional[str] = None) -> Image.Image:
        return self.generator.generate(prompt, output_path)

# 後方互換のための関数（既存呼び出し元のため）
_default_service = ImageGenerationService(StableDiffusionImageGenerator())

def generate_image(prompt: str, output_path: Optional[str] = None) -> Image.Image:
    """
    テキストプロンプトから画像を生成します（サービス層経由）。
    """
    return _default_service.generate_image(prompt, output_path)