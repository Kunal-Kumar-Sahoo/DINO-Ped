import json
import pandas as pd
import matplotlib.pyplot as plt

# Read and parse the log file
data = []
with open("log.txt", "r") as file:
    for line in file:
        data.append(json.loads(line.strip()))

# Convert parsed data to a DataFrame
df = pd.DataFrame(data)

# Extract only training and validation loss columns
loss_columns = ["train_loss", "test_loss"] 
print(df["train_loss"].mean(), df["test_loss"].mean())

# Plot training and validation losses
plt.figure(figsize=(10, 6))
for col in loss_columns:
    plt.plot(df["epoch"], df[col], label=col)
    
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
