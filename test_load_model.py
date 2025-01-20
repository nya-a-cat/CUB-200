import torch
import torchvision.models as models

# Define the same ResNet50 architecture as in the saving code
def create_resnet50():
    model = models.resnet50(weights='IMAGENET1K_V2')
    model.fc = torch.nn.Linear(model.fc.in_features, 200)
    return model

# Path to the saved checkpoint
checkpoint_path = 'model_checkpoints/best_model.pth'

# --- Test 1: Loading only the model_state_dict ---
print("--- Test 1: Loading only the model_state_dict ---")
model_test1 = create_resnet50()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_test1.to(device)

try:
    checkpoint = torch.load(checkpoint_path)
    model_test1.load_state_dict(checkpoint['model_state_dict'])
    print("Successfully loaded model_state_dict.")

    # Optional: Verify functionality by doing a dummy forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model_test1(dummy_input)
    print("Dummy forward pass successful. Output shape:", output.shape)

except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {checkpoint_path}")
except RuntimeError as e:
    print(f"Error loading model_state_dict: {e}")

print("\n")

# --- Test 2: Loading the entire checkpoint (model, optimizer, scheduler) ---
print("--- Test 2: Loading the entire checkpoint ---")
model_test2 = create_resnet50()
optimizer_test2 = torch.optim.AdamW([
    {'params': model_test2.layer4.parameters(), 'lr': 1e-4},
    {'params': model_test2.fc.parameters(), 'lr': 1e-3}
], lr=1e-5, weight_decay=0.01)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler_test2 = CosineAnnealingLR(optimizer_test2, T_max=100, eta_min=1e-6)
model_test2.to(device)
optimizer_test2.to(device) # Ensure optimizer internal states are on the correct device

try:
    checkpoint = torch.load(checkpoint_path)
    model_test2.load_state_dict(checkpoint['model_state_dict'])
    optimizer_test2.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_test2.load_state_dict(checkpoint['scheduler_state_dict'])
    print("Successfully loaded model_state_dict, optimizer_state_dict, and scheduler_state_dict.")

    # Optional: Verify optimizer and scheduler state
    print("Optimizer learning rate after loading:", optimizer_test2.param_groups[0]['lr'])
    print("Scheduler last epoch:", scheduler_test2.last_epoch)

except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {checkpoint_path}")
except KeyError as e:
    print(f"Error loading checkpoint: Missing key {e}")
except RuntimeError as e:
    print(f"Error loading state dict: {e}")

print("\n")

# --- Test 3: Loading model_state_dict into a model with a slightly different architecture (Simulated - same arch for now) ---
print("--- Test 3: Loading model_state_dict into a model with a slightly different architecture (Simulated) ---")
model_test3 = create_resnet50() # Keeping the architecture the same for this test
model_test3.to(device)

try:
    checkpoint = torch.load(checkpoint_path)
    model_test3.load_state_dict(checkpoint['model_state_dict'])
    print("Successfully loaded model_state_dict (even with simulated 'different' architecture - in this case it's the same).")
except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {checkpoint_path}")
except RuntimeError as e:
    print(f"Error loading model_state_dict: {e}")

print("\n")

# --- Test 4: Loading model_state_dict with strict=False ---
print("--- Test 4: Loading model_state_dict with strict=False ---")
model_test4 = create_resnet50()
model_test4.to(device)

try:
    checkpoint = torch.load(checkpoint_path)
    model_test4.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Successfully loaded model_state_dict with strict=False.")

    # Note: With strict=False, missing or unexpected keys are ignored.
    # It's important to understand the implications if your architectures truly differ.

except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {checkpoint_path}")
except RuntimeError as e:
    print(f"Error loading model_state_dict: {e}")

print("\n")

print("All tests completed. Check the output for any errors.")