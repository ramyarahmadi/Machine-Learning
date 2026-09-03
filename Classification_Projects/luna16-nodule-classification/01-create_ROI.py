import os
import numpy as np
import cv2

nodule_folder = r"D:\lung\masks\nodule_mask"
slice_folder = r"D:\lung\slices"

# پیدا کردن اولین Positive
positive_file = None
for filename in os.listdir(nodule_folder):
    if not filename.endswith("_nodule_mask.npy"):
        continue
    mask_path = os.path.join(nodule_folder, filename)
    mask = np.load(mask_path)
    if np.any(mask > 0):
        positive_file = filename
        break
if positive_file is None:
    print("No positive sample found.")
else:
    print("Positive nodule mask:")
    print(positive_file)
    # ساخت نام فایل CT
    ct_filename = positive_file.replace("_nodule_mask.npy",".npy")
    ct_path = os.path.join(slice_folder, ct_filename)
    print("\nCorresponding CT:")
    print(ct_filename)
    print("\nCT exists:", os.path.exists(ct_path))

    # Load Positive Sample
ct_path = r"D:\lung\slices\luna16_subset0_1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059_p40.npy"
nodule_mask_path = r"D:\lung\masks\nodule_mask\luna16_subset0_1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059_p40_nodule_mask.npy"
ct = np.load(ct_path)
nodule_mask = np.load(nodule_mask_path)

# Find Nodule Pixels
ys, xs = np.where(nodule_mask > 0)
print("Number of nodule pixels:", len(xs))
    # مرکز نودول
center_x = int(np.mean(xs))
center_y = int(np.mean(ys))
print("Nodule center:")
print("X =", center_x)
print("Y =", center_y)

# Create 64x64 ROI
roi_size = 64
half = roi_size // 2

x1 = center_x - half
x2 = center_x + half
y1 = center_y - half
y2 = center_y + half
# جلوگیری از خارج شدن ROI از تصویر
x1 = max(0, x1)
y1 = max(0, y1)

x2 = min(ct.shape[1], x2)
y2 = min(ct.shape[0], y2)

roi = ct[y1:y2, x1:x2]
print("ROI shape:", roi.shape)



# Convert CT to display image
window_min = -1000
window_max = 400
roi_display = np.clip(roi, window_min, window_max)
roi_display = ((roi_display - window_min)/ (window_max - window_min)* 255).astype(np.uint8)
# Show ROI
cv2.imshow("Nodule ROI", roi_display)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Feature Extraction
mean_value = np.mean(roi)
median_value = np.median(roi)
std_value = np.std(roi)
min_value = np.min(roi)
max_value = np.max(roi)
range_value = max_value - min_value
variance_value = np.var(roi)


# Display Features
print("\n===== Extracted Features =====")
print("Mean:", mean_value)
print("Median:", median_value)
print("Standard Deviation:", std_value)
print("Minimum:", min_value)
print("Maximum:", max_value)
print("Range:", range_value)
print("Variance:", variance_value)