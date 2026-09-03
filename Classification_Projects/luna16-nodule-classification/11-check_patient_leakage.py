import pandas as pd
import re

data_path = r"D:\lung\features.csv"
df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)
print("\nExample filenames:")
for filename in df["Filename"].head(10):
    print(filename)

def extract_scan_id(filename):
    # Find LUNA16 ID
    pattern = r"(1\.3\.6\.1\.4\.1\.14519\.[0-9.]+)"
    match = re.search(pattern, str(filename))
    if match:
        return match.group(1)
    return None


df["Scan_ID"] = df["Filename"].apply(
    extract_scan_id
)
#Check extraction
print("\nExtracted Scan IDs:")
print(df[["Filename", "Scan_ID", "Label"]].head(20).to_string(index=False))

# Missing IDs
missing = df["Scan_ID"].isna().sum()
print("\nMissing Scan IDs:", missing)
# Number of unique scans
unique_scans = df["Scan_ID"].nunique()
print("Unique Scan IDs:", unique_scans)

# Number of samples per scan
scan_counts = (df.groupby("Scan_ID").size().sort_values(ascending=False))
print("\n")
print("==========================================")
print("SAMPLES PER SCAN")
print("==========================================")
print(
    scan_counts.head(20)
)

# Scans with multiple samples
multiple_samples = scan_counts[scan_counts > 1]
print("\nScans containing multiple samples:")
print(multiple_samples)
print("\nNumber of scans with multiple samples:",len(multiple_samples))

# Class distribution per scan
scan_class = (df.groupby(["Scan_ID", "Label"]).size().unstack(fill_value=0))
print("\n")
print("==========================================")
print("CLASS DISTRIBUTION PER SCAN")
print("==========================================")
print(scan_class.head(20))

# Scans containing both classes
if 0 in scan_class.columns and 1 in scan_class.columns:
    both_classes = scan_class[(scan_class[0] > 0) &(scan_class[1] > 0)]
else:
    both_classes = pd.DataFrame()
print("\nScans containing BOTH positive and negative samples:")
print(both_classes)
print("\nNumber of scans containing both classes:",len(both_classes))

# Save analysis
output_path = (r"D:\lung\scan_leakage_analysis.csv")
scan_counts.to_csv(output_path,header=["Sample_Count"])
print("\nAnalysis saved to:")
print(output_path)

#Final interpretation
print("\n")
print("==========================================")
print("INTERPRETATION")
print("==========================================")
if unique_scans == len(df):
    print("Every sample belongs to a unique Scan ID.")
    print("Patient/Scan-level leakage is unlikely.")
else:
    print("Multiple samples belong to the same Scan ID.")
    print("Patient/Scan-level Cross Validation is recommended.")