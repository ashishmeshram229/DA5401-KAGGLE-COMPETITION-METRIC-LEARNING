import numpy as np
import json

# Load .npy file
data = np.load("metric_name_embeddings.npy", allow_pickle=True)

# Convert numpy array to list (JSON cannot store numpy types)
data_list = data.tolist()

# Save as JSON
with open("output.json", "w") as f:
    json.dump(data_list, f, indent=4)

print("Converted npy → json successfully!")
