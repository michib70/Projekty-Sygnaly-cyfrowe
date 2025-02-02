import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from scipy.signal import convolve2d

# Filtry konwolucyjne
sobel_x = np.array([[1, 0, -1], 
                    [2, 0, -2], 
                    [1, 0, -1]])

sobel_y = np.array([[1, 2, 1], 
                    [0, 0, 0], 
                    [-1, -2, -1]])

rozmycie_gaussa = (1/16) * np.array([[1, 2, 1], 
                                     [2, 4, 2], 
                                     [1, 2, 1]])

sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1], 
                    [0, -1, 0]])

rozmycie_gaussa2 = (1/256) * np.array([[1,  4,  6,  4, 1], 
                                      [4, 16, 24, 16, 4], 
                                      [6, 24, 36, 24, 6], 
                                      [4, 16, 24, 16, 4], 
                                      [1,  4,  6,  4, 1]])

# Interpolacja z interp1d, przy ręcznej implementacji program zajmuje zbyt długo
def interpolacja(x, y, x_interp):
    from scipy.interpolate import interp1d
    interpolator = interp1d(x, y, kind="linear", fill_value="extrapolate", bounds_error=False)
    return interpolator(x_interp)

# Implementacja interpolacji kanałowej
def interpolacja_kanalu(kanal, height, width):
    nowy_kanal = kanal.copy()
    for row in range(height):
        x = np.where(kanal[row, :] > 0)[0]
        if len(x) > 1:
            y = kanal[row, x]
            x_interp = np.arange(width)
            nowy_kanal[row, :] = interpolacja(x, y, x_interp)
    for col in range(width):
        y = nowy_kanal[:, col]
        x = np.where(y > 0)[0]
        if len(x) > 1:
            y_interp = interpolacja(x, y[x], np.arange(height))
            nowy_kanal[:, col] = y_interp
    return nowy_kanal

# Demozaikowanie obrazu Bayera
def demozaikowanie_Bayera(image):
    height, width = image.shape
    R, G, B = np.zeros((height, width)), np.zeros((height, width)), np.zeros((height, width))
    
    R[0::2, 1::2] = image[0::2, 1::2]
    B[1::2, 0::2] = image[1::2, 0::2]
    G[0::2, 0::2] = image[0::2, 0::2]
    G[1::2, 1::2] = image[1::2, 1::2]

    R = interpolacja_kanalu(R, height, width)
    G = interpolacja_kanalu(G, height, width)
    B = interpolacja_kanalu(B, height, width)

    demo_obraz = np.stack((R, G, B), axis=-1)
    return np.clip(demo_obraz, 0, 255).astype(np.uint8)

# Wczytanie obrazu 
bayer_image = np.load('/content/mond.npy')

# Konwersja obrazu na skalę szarości i uint8
if len(bayer_image.shape) == 3 and bayer_image.shape[2] == 3:
    bayer_image = rgb2gray(bayer_image)  
bayer_image = img_as_ubyte(bayer_image)

demozaikowany_obraz = demozaikowanie_Bayera(bayer_image)

# Korekta balansu kolorów
avg_r = np.mean(demozaikowany_obraz[:, :, 0])
avg_g = np.mean(demozaikowany_obraz[:, :, 1])
avg_b = np.mean(demozaikowany_obraz[:, :, 2])

demozaikowany_obraz[:, :, 0] = np.clip(demozaikowany_obraz[:, :, 0] * (avg_g / avg_r), 0, 255)
demozaikowany_obraz[:, :, 2] = np.clip(demozaikowany_obraz[:, :, 2] * (avg_g / avg_b), 0, 255)

# Konwolucja dla każdego kanału
def zast_konw(image, kernel):
    return np.dstack([convolve2d(image[:, :, i], kernel, mode='same', boundary='symm', fillvalue=0) for i in range(3)])

# Wykrywanie krawędzi filtrami Sobela (w skali szarości)
krawedz_x = convolve2d(rgb2gray(demozaikowany_obraz), sobel_x, mode='same', boundary='symm')
krawedz_y = convolve2d(rgb2gray(demozaikowany_obraz), sobel_y, mode='same', boundary='symm')
krawedzie = np.sqrt(krawedz_x**2 + krawedz_y**2)
krawedzie = np.clip((krawedzie - np.min(krawedzie)) / (np.max(krawedzie) - np.min(krawedzie)) * 255, 0, 255).astype(np.uint8)

# Rozmycie i wyostrzenie kolorow
rozmazanie = np.dstack([
    convolve2d(demozaikowany_obraz[:, :, i], rozmycie_gaussa, mode='same', boundary='symm')
    for i in range(3)
])

rozmazanie2 = np.dstack([
    convolve2d(demozaikowany_obraz[:, :, i], rozmycie_gaussa2, mode='same', boundary='symm')
    for i in range(3)
])
rozmazanie = np.clip(rozmazanie, 0, 255).astype(np.uint8)
rozmazanie2 = np.clip(rozmazanie2, 0, 255).astype(np.uint8)

wyostrzenie = zast_konw(demozaikowany_obraz, sharpen)

# Wizualizacja efektów
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(bayer_image, cmap='gray')
axes[0, 0].set_title("Mozaika Bayera")
axes[0, 1].imshow(demozaikowany_obraz)
axes[0, 1].set_title("Demozaikowanie")
axes[0, 2].imshow(krawedzie, cmap='gray')
axes[0, 2].set_title("Wykrywanie krawędzi")
axes[1, 0].imshow(rozmazanie)
axes[1, 0].set_title("Rozmycie")
axes[1, 1].imshow(wyostrzenie)
axes[1, 1].set_title("Wyostrzanie")
axes[1, 2].imshow(rozmazanie2)
axes[1, 2].set_title("Większe rozmycie")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
