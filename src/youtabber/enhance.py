import urllib.request
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from youtabber.source import CACHE

FACTOR = 2
# Unsharp mask: add back the difference between the image and a blurred copy to crisp up edges
SHARPEN_AMOUNT = 1.0
SHARPEN_SIGMA = 1.5
# Levels: anything darker than BLACK becomes black and lighter than WHITE becomes white.
# Grays in between stay gray, which keeps light staff lines and measure numbers visible.
BLACK = 80
WHITE = 230

# Short names for models that can be downloaded on first use
MODELS = {
    # Real-ESRGAN trained on line art, which suits engraved notation better than the photo models
    "realesrgan-anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
}

Upscaler = Callable[[np.ndarray], np.ndarray]


def _levels(image: np.ndarray) -> np.ndarray:
    return np.clip((image.astype(np.float32) - BLACK) * 255 / (WHITE - BLACK), 0, 255).astype(np.uint8)


def classic(image: np.ndarray) -> np.ndarray:
    """Upscale whitened notation and sharpen its edges for printing."""
    up = cv2.resize(image, None, fx=FACTOR, fy=FACTOR, interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    sharp = up + SHARPEN_AMOUNT * (up - cv2.GaussianBlur(up, (0, 0), SHARPEN_SIGMA))
    return _levels(sharp)


def model_upscaler(model: str) -> Upscaler:
    """An upscaler that runs a super-resolution model: a name from MODELS or a path to a model file."""
    try:
        import torch
        from spandrel import ModelLoader
    except ImportError:
        project = Path(__file__).resolve().parents[2]
        source = project if (project / "pyproject.toml").exists() else Path("path/to/youtabber")
        raise SystemExit(f"Running a model needs PyTorch and spandrel. Reinstall with the ai extra:\n"
                         f"  uv tool install --editable '{source}[ai]'")
    path = _model_file(model)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    net = ModelLoader().load_from_file(str(path)).to(device).eval()

    def upscale(image: np.ndarray) -> np.ndarray:
        # The models expect RGB in 0..1, so repeat the gray channel three times
        x = torch.from_numpy(np.repeat(image[None, None].astype(np.float32) / 255, 3, axis=1)).to(device)
        with torch.no_grad():
            y = net(x)
        out = (y[0].mean(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        # Bring every model to the same output size so PDFs don't grow with the model's scale
        size = (image.shape[1] * FACTOR, image.shape[0] * FACTOR)
        return _levels(cv2.resize(out, size, interpolation=cv2.INTER_AREA))

    return upscale


def _model_file(model: str) -> Path:
    if model not in MODELS:
        path = Path(model).expanduser()
        if not path.exists():
            raise SystemExit(f"Model {model!r} is neither a file nor one of: {', '.join(MODELS)}")
        return path
    url = MODELS[model]
    path = CACHE / "models" / url.rsplit("/", 1)[1]
    if not path.exists():
        print(f"Downloading model {model}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Download beside the final name and rename when complete, so a cut-off download is never used
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    return path
