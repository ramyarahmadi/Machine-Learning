import os
import numpy as np

nodule_folder = r"D:\lung\masks\nodule_mask"

count_total = 0
count_positive = 0
count_negative = 0

for filename in os.listdir(nodule_folder):

    if not filename.endswith(".npy"):
        continue

    path = os.path.join(nodule_folder, filename)

    mask = np.load(path)

    count_total += 1

    if np.any(mask > 0):
        count_positive += 1

        print("Positive:", filename)

        # فقط چند نمونه اول را نمایش بده
        if count_positive >= 5:
            break

    else:
        count_negative += 1

print("\nTotal checked:", count_total)
print("Positive:", count_positive)
print("Negative:", count_negative)