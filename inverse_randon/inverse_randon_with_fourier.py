#%%
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale

def get_r_coords(diameter, num):
    if diameter % 2 == 0:
        radius = diameter / 2 - 0.5
        center = -0.5
        return np.linspace(-radius, radius, num) + center
    else:
        radius = (diameter - 1) / 2
        return np.linspace(-radius, radius, num)


def expand_diameter(diameter, K):
    expanded_diameter = int(diameter * K)
    if expanded_diameter % 2 == 1:
        expanded_diameter += 1
    return expanded_diameter


def get_kspace_radial(diameter, expanded_diameter, n_projections):
  r = get_r_coords(diameter, expanded_diameter)
  a = np.linspace(0, np.pi, n_projections, endpoint=False)
  r_grid, a_grid = np.meshgrid(r, a, indexing='xy')
  x = np.round((r_grid * np.cos(a_grid)) * expanded_diameter / diameter) % expanded_diameter
  y = np.round((-r_grid * np.sin(a_grid)) * expanded_diameter / diameter) % expanded_diameter
  return x.astype(np.int32), y.astype(np.int32)


def pad_image(image):
    diagonal = np.sqrt(2) * max(image.shape[-2:])
    pad = [int(np.ceil(diagonal - s)) for s in image.shape[-2:]]
    new_center = [(s + p) // 2 for s, p in zip(image.shape[-2:], pad)]
    old_center = [s // 2 for s in image.shape[-2:]]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    pad_width = [(0, 0) for i in image.shape[:-2]] + pad_width
    padded_image = np.pad(image, pad_width, mode='constant',
                            constant_values=0)
    return padded_image


def unpad_image(image):
    size = int(np.sqrt(image.shape[-1] ** 2 / 2))
    pad_left = (image.shape[-1] - size) // 2
    return image[..., pad_left:pad_left + size, pad_left:pad_left + size]


def _expand_shapes(*shapes):
  shapes = [list(shape) for shape in shapes]
  max_ndim = max(len(shape) for shape in shapes)
  shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                for shape in shapes]

  return tuple(shapes_exp)


def resize(input, oshape, ishift=None, oshift=None):
    """Resize with zero-padding or cropping.
    Args:
        input (array): Input array.
        oshape (tuple of ints): Output shape.
        ishift (None or tuple of ints): Input shift.
        oshift (None or tuple of ints): Output shift.
    Returns:
        array: Zero-padded or cropped result.
    """

    ishape1, oshape1 = _expand_shapes(input.shape, oshape)

    if ishape1 == oshape1:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

    copy_shape = [min(i - si, o - so)
                    for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = np.zeros(oshape1, dtype=input.dtype)
    input = input.reshape(ishape1)
    output[oslice] = input[islice]

    return output.reshape(oshape)


def fft_radon_to_kspace(image, expansion=6):

    image = pad_image(image)


    diameter = image.shape[-1]
    print(image.shape)
    expanded_diameter = expand_diameter(diameter, expansion)
    print(expanded_diameter)
    oshape = image.shape[:-2] + (expanded_diameter, expanded_diameter)
    print(oshape)
    image = resize(image, oshape)

    kspace = np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), axes=(-2, -1))
    return kspace


def fft_radon_to_image(kspace, size):
    image = np.fft.fftshift(np.fft.ifft2(kspace, axes=(-2, -1)), axes=(-2, -1))
    diagonal = math.ceil(np.sqrt(2) * size)
    oshape = image.shape[:-2] + (diagonal, diagonal)
    image = resize(image, oshape)
    return unpad_image(image.real)



def fft_kspace_to_sino(kspace, n_projections, size, expansion):
    diameter = math.ceil(np.sqrt(2.) * size)
    expanded_diameter = expand_diameter(diameter, expansion)
    x, y = get_kspace_radial(diameter, expanded_diameter, n_projections)
    print(kspace.shape)
    print(x.shape)
    print(y.shape)
    slices = kspace[..., y.astype(np.int32), x.astype(np.int32)]
    sinogram = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(slices, axes=-1), axis=-1), axes=-1)
    # oshape = sinogram.shape[:-1] + (diameter,)
    # return util.resize(sinogram, oshape)
    return sinogram


def fft_sino_to_kspace(sino, n_projections, size, expansion):
    diameter = math.ceil(np.sqrt(2.) * size)
    expanded_diameter = expand_diameter(diameter, expansion)
    x, y = get_kspace_radial(diameter, expanded_diameter, n_projections)

    # oshape = sino.shape[:-2] + (n_projections, expanded_diameter)
    # sino = util.resize(sino, oshape)
    slices = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(sino, axes=-1), axis=-1), axes=-1)
    oshape = sino.shape[:-2] + (expanded_diameter, expanded_diameter)
    kspace = np.zeros(oshape, dtype=np.complex64)
    kspace[..., y.astype(np.int32), x.astype(np.int32)] = slices
    return kspace

#%%
# image to sinogram
# by: image -> kspace -> sinogram

# example data
from skimage.data import shepp_logan_phantom
x = shepp_logan_phantom()
x = rescale(x, scale=0.4, mode='reflect')
# x = torch.from_numpy(x)
#%%
expansion = 3
size = max(x.shape)

to_space = lambda x: fft_radon_to_kspace(x, expansion)[..., None]
from_space = lambda x: fft_radon_to_image(x[..., 0], size)[..., None]
x_space = to_space(x)
x_sino = fft_kspace_to_sino(x_space[..., 0], size, size, expansion)[..., None]
x_sino = np.real(x_sino)


#%% plot sinogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(x, cmap=plt.cm.Greys_r)


sinogram = x_sino[..., 0].T
dx, dy = 0.5 * 180.0 / size, 0.5 / sinogram.shape[0]
ax2.set_title("Radon transform from Fourier code\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
# ax2.set_ylim(350,550)
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')

fig.tight_layout()
plt.show()

#%% change expansion observe sinogram
expansions = [1,2,3,4,5,6]
size = max(x.shape)
fig, axes = plt.subplots(1, 6, figsize=(16, 4))
for i in range(6):
    to_space = lambda x: fft_radon_to_kspace(x, expansions[i])[..., None]
    from_space = lambda x: fft_radon_to_image(x[..., 0], size)[..., None]
    x_space = to_space(x)
    x_sino = fft_kspace_to_sino(x_space[..., 0], size, size, expansions[i])[..., None]
    x_sino = np.real(x_sino)
    sinogram = x_sino[..., 0].T
    dx, dy = 0.5 * 180.0 / size, 0.5 / sinogram.shape[0]
    axes[i].set_title("expansion="+str(i+1))
    axes[i].imshow(sinogram, cmap=plt.cm.Greys_r,
            extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
            aspect='auto')
axes[0].set_xlabel("Projection angle (deg)")
axes[0].set_ylabel("Projection position (pixels)")
axes[0].set_title("Sinogram\nexpansion="+str(i+1))
axes[0].set_ylim(25,200)
axes[1].set_ylim(130,320)
axes[2].set_ylim(250,425)
axes[3].set_ylim(360,550)
axes[4].set_ylim(470,660)
axes[5].set_ylim(590,770)
fig.tight_layout()
plt.show()
#%% reconstruction
# by: sinogram -> kspace --(interpolation)--> x

recon_x_kspace = fft_sino_to_kspace(x_sino[..., 0], size, size, expansion)[..., None]
recon_x = from_space(recon_x_kspace)
# %% plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(x, cmap=plt.cm.Greys_r)


ax2.set_title("Recon_fourier(expansion=4)")
ax2.imshow(recon_x, cmap=plt.cm.Greys_r)

fig.tight_layout()
plt.show()