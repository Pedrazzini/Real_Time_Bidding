import pickle
import matplotlib.pyplot as plt
import numpy as np

# Carica la training history
with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

print(history.keys())

for epoch, (train_loss, val_loss) in enumerate(
    zip(history["train_loss"], history["val_loss"]), start=1
):
    print(
        f"Epoch {epoch:3d} | "
        f"Train loss: {train_loss:.6f} | "
        f"Val loss: {val_loss:.6f}"
    )


best_epoch = np.argmin(history["val_loss"]) + 1
best_val_loss = min(history["val_loss"])

print()
print("Miglior epoca:", best_epoch)
print("Miglior validation loss:", best_val_loss)

plt.figure(figsize=(10, 6))

plt.plot(history["train_loss"], label="Train loss")
plt.plot(history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Training history - p_theta")
plt.legend()
plt.grid(True)

plt.show()