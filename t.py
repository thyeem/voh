# Create dummy datasets

from voh.voh import _dataloader, _dataset

# Create dummy datasets
train_data = [(f"train_{i}", i) for i in range(20)]
val_data = [(f"val_{i}", i) for i in range(10)]

train_dataset = _dataset("data-train.db")
val_dataset = _dataset("data-val.db")

# Initialize the dataloader
dataloader = _dataloader(
    datasets=(train_dataset, val_dataset), size_buffer=20, num_workers=2
)

# Test train mode
print("Testing train mode:")
dataloader.train()
for i, item in enumerate(dataloader):
    print(f"Train item {i}: {item}")

    if i % 10 == 0:
        print("Evaluation-mode")
        dataloader.eval()
        for i, item in enumerate(dataloader):
            print(f"Eval item {i}: {item}")
            if i >= 3:  # Stop after 10 items
                break


print("\nTest completed.")
# Assuming the _dataloader class is already defined
