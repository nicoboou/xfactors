import torch
from io import BytesIO
from PIL import Image
from torchvision.datasets import CelebA
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class CelebADataset(CelebA):
    def __init__(
        self,
        *args,
        degradation_types: list[str] | None = None,
        degradation_levels: list[int] | None = None,
        add_degradation_factor: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.degradation_types = [d.lower() for d in (degradation_types or []) if d]
        self.degradation_levels = degradation_levels or [0]
        self.add_degradation_factor = add_degradation_factor
        self.max_level = max(self.degradation_levels) if self.degradation_levels else 0

    def _check_integrity(self):
        return True

    def _sample_level(self) -> int:
        if not self.degradation_levels:
            return 0
        idx = torch.randint(0, len(self.degradation_levels), (1,)).item()
        return self.degradation_levels[idx]

    def _apply_blur(self, image: torch.Tensor, level_norm: float) -> torch.Tensor:
        kernel = 1 + 2 * int(round(3 * level_norm))
        if kernel <= 1:
            return image
        sigma = 0.1 + 1.9 * level_norm
        return TF.gaussian_blur(image, kernel_size=[kernel, kernel], sigma=[sigma, sigma])

    def _apply_down_up_bilinear(self, image: torch.Tensor, level_norm: float) -> torch.Tensor:
        _, h, w = image.shape
        scale = max(0.2, 1.0 - 0.75 * level_norm)
        low_h = max(8, int(h * scale))
        low_w = max(8, int(w * scale))
        low = TF.resize(
            image,
            [low_h, low_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return TF.resize(low, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)

    def _apply_down_up_bicubic(self, image: torch.Tensor, level_norm: float) -> torch.Tensor:
        _, h, w = image.shape
        scale = max(0.2, 1.0 - 0.75 * level_norm)
        low_h = max(8, int(h * scale))
        low_w = max(8, int(w * scale))
        low = TF.resize(
            image,
            [low_h, low_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return TF.resize(low, [h, w], interpolation=InterpolationMode.BICUBIC, antialias=True)

    def _apply_nearest_neighbor(self, image: torch.Tensor, level_norm: float) -> torch.Tensor:
        _, h, w = image.shape
        scale = max(0.2, 1.0 - 0.75 * level_norm)
        low_h = max(8, int(h * scale))
        low_w = max(8, int(w * scale))
        low = TF.resize(
            image,
            [low_h, low_w],
            interpolation=InterpolationMode.NEAREST,
            antialias=False,
        )
        return TF.resize(low, [h, w], interpolation=InterpolationMode.NEAREST, antialias=False)

    def _apply_noise(self, image: torch.Tensor, level_norm: float) -> torch.Tensor:
        if level_norm <= 0:
            return image
        std = 0.02 + 0.12 * level_norm
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0.0, 1.0)

    def _apply_jpeg(self, image: torch.Tensor, level_norm: float) -> torch.Tensor:
        quality = max(5, int(round(95 - 75 * level_norm)))
        pil_img = TF.to_pil_image(torch.clamp(image, 0.0, 1.0))
        if pil_img.mode not in {"L", "RGB"}:
            pil_img = pil_img.convert("RGB")
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        jpeg_img = Image.open(buffer)
        out = TF.to_tensor(jpeg_img).to(image.dtype)
        return torch.clamp(out, 0.0, 1.0)

    def _apply_inpainting(self, image: torch.Tensor, level_norm: float) -> torch.Tensor:
        size_ratio = 0.8 * level_norm
        if size_ratio <= 0:
            return image
        _, h, w = image.shape
        crop_h = int(h * size_ratio)
        crop_w = int(w * size_ratio)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        image = image.clone()
        image[:, top:top+crop_h, left:left+crop_w] = 0.5
        return image.clamp(0.0, 1.0)

    def _apply_degradation(self, image: torch.Tensor, level: int) -> torch.Tensor:
        if (not self.degradation_types) or (self.degradation_types == ["none"]):
            return image

        level_norm = 0.0 if self.max_level <= 0 else float(level) / float(self.max_level)
        selected_types = list(self.degradation_types)
        allowed_types = {"none", "combo", "bilinear", "bicubic", "nearest_neighbor", "blur", "noise", "jpeg", "inpainting"}
        invalid_types = [d for d in selected_types if d not in allowed_types]
        if invalid_types:
            raise ValueError(f"Unknown degradation type(s): {invalid_types}. Allowed: {sorted(allowed_types)}")

        if "combo" in selected_types:
            selected_types = ["bilinear", "bicubic", "blur", "noise", "jpeg"]

        out = image
        for degradation_type in selected_types:
            if degradation_type == "bilinear":
                out = self._apply_down_up_bilinear(out, level_norm)
            elif degradation_type == "bicubic":
                out = self._apply_down_up_bicubic(out, level_norm)
            elif degradation_type == "nearest_neighbor":
                out = self._apply_nearest_neighbor(out, level_norm)
            elif degradation_type == "blur":
                out = self._apply_blur(out, level_norm)
            elif degradation_type == "noise":
                out = self._apply_noise(out, level_norm)
            elif degradation_type == "jpeg":
                out = self._apply_jpeg(out, level_norm)
            elif degradation_type == "inpainting":
                out = self._apply_inpainting(out, level_norm)
        return out

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)

        level = self._sample_level()
        image = self._apply_degradation(image, level)

        if not self.add_degradation_factor:
            return image, target

        if isinstance(target, tuple):
            target = target[0]

        target_tensor = target if isinstance(target, torch.Tensor) else torch.as_tensor(target)
        target_tensor = target_tensor.reshape(-1).to(torch.int64)
        degradation_tensor = torch.tensor([level], dtype=torch.int64)
        target_with_degradation = torch.cat([target_tensor, degradation_tensor], dim=0)
        return image, target_with_degradation
