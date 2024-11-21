# Create dummy datasets

from foc import *

from voh.voh import _dataloader, _dataset

train_dataset = _dataset("data-train.db")
val_dataset = _dataset("data-val.db")

# Initialize the dataloader
tloader = _dataloader(train_dataset, size_buffer=5, num_workers=4)
vloader = _dataloader(val_dataset, size_buffer=5, num_workers=4)

# Test train mode
print("Testing train mode:")
for i, (item, a, b) in enumerate(take(100)(tloader)):
    print(f"Train item {i}: {item}")

    if i % 10 == 0:
        print("Evaluation-mode")
        for i, item in enumerate(take(5)(vloader)):
            print(f"Eval item {i}: {item}")
            # if i >= 3:  # Stop after 10 items
            # break


print("\nTest completed.")
# Assuming the _dataloader class is already defined
