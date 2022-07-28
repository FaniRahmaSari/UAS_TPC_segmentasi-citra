import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max


# Hasilkan gambar awal dengan dua lingkaran yang tumpang tindih
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 26, 26, 42, 58
r1, r2 = 18, 20
mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
image = np.logical_or(mask_circle1, mask_circle2)

# memisahkan dua objek dalam gambar
# Hasilkan penanda sebagai maxima lokal dari jarak ke latar belakang
distance = ndi.distance_transform_edt(image)
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('objek tumpang tindih')
ax[1].imshow(-distance, cmap=plt.cm.gray)
ax[1].set_title('jarak')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[2].set_title('objek terpisah')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()