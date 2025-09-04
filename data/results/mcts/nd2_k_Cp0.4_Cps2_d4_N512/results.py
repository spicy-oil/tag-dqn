import re
import math

aaa_values = []
ratios = []

for i in range(25):
    filename = f"log_{i}.txt"
    with open(filename, "r") as f:
        lines = f.readlines()
        line = lines[-24].strip()

        # Extract using regex
        match = re.search(r"out of (\d+) and (\d+) correct IDs", line)
        if match:
            zzz = int(match.group(1))
            aaa = int(match.group(2))
            aaa_values.append(aaa)
            ratios.append(aaa / zzz)
        else:
            print(f"Pattern not found in {filename}")

# Helper function for standard deviation
def std(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

# Compute averages and 2×STD
mean_aaa = sum(aaa_values) / len(aaa_values)
std_aaa = std(aaa_values)

mean_ratio = sum(ratios) / len(ratios)
std_ratio = std(ratios)

print(f"Average aaa: {mean_aaa:.2f} ± {2 * std_aaa:.2f}")
print(f"Average aaa/zzz: {mean_ratio:.4f} ± {2 * std_ratio:.4f}")

