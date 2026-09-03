import os
import numpy as np
import pandas as pd



SLICE_FOLDER = r"D:\lung\slices"
LUNG_MASK_FOLDER = r"D:\lung\masks\lung_mask"
NODULE_MASK_FOLDER = r"D:\lung\masks\nodule_mask"

ROI_SIZE = 64
NUMBER_OF_SAMPLES = 100

# برای تکرارپذیری
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Feature Extraction
def extract_features(roi):
    return {
        "Mean": np.mean(roi),
        "Median": np.median(roi),
        "Std": np.std(roi),
        "Min": np.min(roi),
        "Max": np.max(roi),
        "Range": np.max(roi) - np.min(roi),
        "Variance": np.var(roi)
    }

# Create ROI
def get_roi(image, center_x, center_y, size=64):
    half = size // 2
    x1 = center_x - half
    x2 = center_x + half

    y1 = center_y - half
    y2 = center_y + half

    # اگر ROI از تصویر خارج شود
    if x1 < 0 or y1 < 0:
        return None
    if x2 > image.shape[1] or y2 > image.shape[0]:
        return None
    return image[y1:y2, x1:x2]


# Get all nodule files
nodule_files = [
f for f in os.listdir(NODULE_MASK_FOLDER)
    if f.endswith("_nodule_mask.npy")
]



# POSITIVE SAMPLES
print("Searching for positive samples...")
positive_samples = []
for filename in nodule_files:
    if len(positive_samples) >= NUMBER_OF_SAMPLES:
        break
    nodule_path = os.path.join(NODULE_MASK_FOLDER,filename)
    nodule_mask = np.load(nodule_path)

    # پیدا کردن پیکسل‌های نودول
    ys, xs = np.where(nodule_mask > 0)
    if len(xs) == 0:
        continue
    # CT متناظر
    ct_filename = filename.replace("_nodule_mask.npy",".npy")
    ct_path = os.path.join(SLICE_FOLDER,ct_filename)
    if not os.path.exists(ct_path):
        continue
    ct = np.load(ct_path)
    # مرکز نودول
    center_x = int(np.mean(xs))
    center_y = int(np.mean(ys))

    # ROI
    roi = get_roi(ct,center_x,center_y,ROI_SIZE)

    if roi is None:
        continue
    # Feature extraction
    features = extract_features(roi)
    features["Label"] = 1
    features["Filename"] = ct_filename
    positive_samples.append(features)

print("Positive samples:", len(positive_samples))

# NEGATIVE SAMPLES
print("\nSearching for negative samples...")
negative_samples = []
for filename in nodule_files:
    if len(negative_samples) >= NUMBER_OF_SAMPLES:
        break
    nodule_path = os.path.join(
        NODULE_MASK_FOLDER,
        filename
    )
    nodule_mask = np.load(nodule_path)

    # این فایل باید کاملاً بدون نودول باشد
    if np.any(nodule_mask > 0):
        continue
    # CT
    ct_filename = filename.replace("_nodule_mask.npy",".npy")
    ct_path = os.path.join(SLICE_FOLDER,ct_filename)
    # Lung Mask
    lung_filename = filename.replace("_nodule_mask.npy","_lung_mask.npy")
    lung_path = os.path.join(LUNG_MASK_FOLDER,lung_filename)
    if not os.path.exists(ct_path):
        continue
    if not os.path.exists(lung_path):
        continue
    ct = np.load(ct_path)
    lung_mask = np.load(lung_path)

    # Find possible ROI centers
    ys, xs = np.where(lung_mask > 0)
    if len(xs) == 0:
        continue

    # چند نقطه تصادفی امتحان می‌کنیم
    indices = rng.permutation(len(xs))
    roi_found = False

    for index in indices:
        center_x = int(xs[index])
        center_y = int(ys[index])
        roi = get_roi(ct,center_x,center_y,ROI_SIZE)
        if roi is None:
            continue
        # مطمئن شو ROI داخل ریه باشد
        half = ROI_SIZE // 2

        x1 = center_x - half
        x2 = center_x + half

        y1 = center_y - half
        y2 = center_y + half

        lung_roi = lung_mask[y1:y2, x1:x2]
        if lung_roi.shape != (ROI_SIZE, ROI_SIZE):
            continue
        # حداقل 80 درصد ROI داخل ریه باشد
        lung_ratio = np.mean(lung_roi > 0)
        if lung_ratio < 0.80:
            continue

        # قبول کردن Negative ROI
        features = extract_features(roi)
        features["Label"] = 0
        features["Filename"] = ct_filename
        negative_samples.append(features)
        roi_found = True
        break
    if not roi_found:
        continue

print("Negative samples:", len(negative_samples))

# Combine
dataset = positive_samples + negative_samples
df = pd.DataFrame(dataset)


# Shuffle
df = df.sample(frac=1,random_state=42).reset_index(drop=True)

output_path = r"D:\lung\features.csv"
df.to_csv(output_path,index=False)

print("\n==============================")
print("Dataset created successfully")
print("==============================")
print("Total samples:", len(df))
print("\nClass distribution:")
print(df["Label"].value_counts())
print("\nDataset shape:")
print(df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nSaved to:")
print(output_path)