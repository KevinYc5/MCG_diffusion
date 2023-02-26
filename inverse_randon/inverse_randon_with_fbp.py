#%%
import torch
from torch import nn
import torch.nn.functional as F
from skimage.transform import radon, rescale
from physics.radon.filters import RampFilter
from physics.radon.utils import PI, SQRT2, deg2rad, affine_grid, grid_sample
import matplotlib.pyplot as plt
import math
import numpy as np
'''source: https://github.com/matteo-ronchetti/torch-radon'''

#%%
class Radon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True, dtype=torch.float):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):
        N, C, W, H = x.shape
        assert (W == H)

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, C, W, len(self.theta), device=x.device, dtype=self.dtype)

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            out[..., i] = rotated.sum(2)

        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, torch.Size([1, 1, grid_size, grid_size])))
        return all_grids


class IRadon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=True,
                 use_filter=RampFilter(), out_size=None, dtype=torch.float):
        super(IRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            self.all_grids = self._create_grids(self.theta, in_size, circle)
        self.filter = use_filter if use_filter is not None else lambda x: x

    def forward(self, x):
        it_size = x.shape[2]
        ch_size = x.shape[1]

        if self.in_size is None:
            self.in_size = int((it_size / SQRT2).floor()) if not self.circle else it_size
        # if None in [self.ygrid, self.xgrid, self.all_grids]:
        if self.ygrid is None or self.xgrid is None or self.all_grids is None :
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.all_grids = self._create_grids(self.theta, self.in_size, self.circle)
        # sinogram
        x = self.filter(x)

        reco = torch.zeros(x.shape[0], ch_size, it_size, it_size, device=x.device, dtype=self.dtype)
        for i_theta in range(len(self.theta)):
            reco += grid_sample(x, self.all_grids[i_theta].repeat(reco.shape[0], 1, 1, 1).to(x.device))

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1]))

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(x.shape[0], ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.

        reco = reco * PI.item() / (2 * len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size) // 2
            reco = F.pad(reco, (pad, pad, pad, pad))

        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2 * in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype)
        return torch.meshgrid(unitrange, unitrange)

    def _XYtoT(self, theta):
        T = self.xgrid * (deg2rad(theta)).cos() - self.ygrid * (deg2rad(theta)).sin()
        return T

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for i_theta in range(len(angles)):
            X = torch.ones(grid_size, dtype=self.dtype).view(-1, 1).repeat(1, grid_size) * i_theta * 2. / (
                        len(angles) - 1) - 1.
            Y = self._XYtoT(angles[i_theta])
            all_grids.append(torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1).unsqueeze(0))
        return all_grids

#%%
# example data
from skimage.data import shepp_logan_phantom
x = shepp_logan_phantom()
x = rescale(x, scale=0.4, mode='reflect')

img_width = max(x.shape)
num_proj = 180
device = 'cuda:0'
radon = Radon(in_size=img_width, theta=torch.arange(num_proj), circle=False).to(device)
iradon = IRadon(in_size=img_width, theta=torch.arange(num_proj), circle=False).to(device)

#%%
img = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device).to(torch.float)
sinogram = radon(img)
b_img = iradon(sinogram)

#%% plt sinogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(x, cmap=plt.cm.Greys_r)


theta = np.linspace(0., 180., num_proj, endpoint=False)
x_sino = sinogram[0][0].cpu().detach().numpy()
dx, dy = 0.5 * 180.0 / num_proj, 0.5 / x_sino.shape[0]
ax2.set_title("Radon transform from FBP code\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(x_sino, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, x_sino.shape[0] + dy),
           aspect='auto')

fig.tight_layout()
plt.show()
#%% plt recon

recon_x = b_img[0][0].cpu().detach().numpy()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(x, cmap=plt.cm.Greys_r)


ax2.set_title("Recon_fbp")
ax2.imshow(recon_x, cmap=plt.cm.Greys_r)

fig.tight_layout()
plt.show()

# %%
