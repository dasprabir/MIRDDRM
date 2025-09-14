import torch
import torch.nn.functional as F

class BicubicDownsampling:
    def __init__(self, scale_factor: int, image_size: int, device='cpu'):
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.device = device
        self.output_size = image_size // scale_factor
        self.svd_ready = True
        self._singulars = None

    def H(self, x: torch.Tensor) -> torch.Tensor:
        # downsample via bicubic
        return F.interpolate(
            x, scale_factor=1/self.scale_factor,
            mode='bicubic', align_corners=False
        )

    def H_pinv(self, y: torch.Tensor) -> torch.Tensor:
        # pseudoinverse via bicubic upsampling
        return F.interpolate(
            y, scale_factor=self.scale_factor,
            mode='bicubic', align_corners=False
        )

    def Ut(self, y: torch.Tensor) -> torch.Tensor:
        # use same as H_pinv for back-projection
        return self.H_pinv(y)

    def singulars(self) -> torch.Tensor:
        # dummy singular values
        if self._singulars is None:
            size = self.output_size * self.output_size
            self._singulars = torch.ones(size, device=self.device)
        return self._singulars

    def __repr__(self):
        return f'BicubicDownsampling(scale={self.scale_factor}, size={self.image_size})'


def get_operator(deg: str, image_size: int, device: str):
    # only bicubic x4 supported
    if deg == 'sr_bicubic_x4':
        return BicubicDownsampling(scale_factor=4, image_size=image_size, device=device)
    raise ValueError(f"Unknown degradation type: {deg}")