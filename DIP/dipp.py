
import cv2
import numpy as np
import matplotlib.pyplot as plt

def print_image_size(image):
    print("Image shape (Height x Width x Channels):", image.shape)

def subset_image(image, size=(15, 15)):
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    start_x, start_y = center_x - size[0] // 2, center_y - size[1] // 2
    subset = image[start_y:start_y + size[1], start_x:start_x + size[0]]
    return subset

def add_salt_pepper_noise(img, prob=0.1):
    noisy = np.copy(img)
    total_pixels = img.shape[0] * img.shape[1]
    num_salt = int(prob * total_pixels)

    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = [255, 255, 255]

    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = [0, 0, 0]
    return noisy

def print_dn_values(subset):
    print("\nDN values (R, G, B) for each pixel in 15x15 subset:")
    for i in range(subset.shape[0]):
        for j in range(subset.shape[1]):
            print(f"Pixel ({i},{j}): {subset[i, j]}")

def notch_filter_denoise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    notch_radius = 3
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - notch_radius:crow + notch_radius, ccol - notch_radius:ccol + notch_radius] = 0
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(img_back)

image = cv2.imread('C:/Users/utpal/OneDrive/Desktop/dip/new1.JPG')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print_image_size(image_rgb)
subset = subset_image(image_rgb, size=(15, 15))
noisy_subset = add_salt_pepper_noise(subset, prob=0.2)
noisy_image = add_salt_pepper_noise(image_rgb, prob=0.2)
print_dn_values(noisy_subset)
restored_subset = notch_filter_denoise(noisy_subset)

plt.figure(figsize=(18, 6))
plt.subplot(1, 5, 1), plt.imshow(image_rgb), plt.title('Original Image')
plt.subplot(1, 5, 2), plt.imshow(noisy_image), plt.title('Noisy Image')
plt.subplot(1, 5, 3), plt.imshow(subset), plt.title('Subset (15x15)')
plt.subplot(1, 5, 4), plt.imshow(noisy_subset), plt.title('Noisy Subset')
plt.subplot(1, 5, 5), plt.imshow(restored_subset), plt.title('Restored Subset')

plt.tight_layout()
plt.show()

