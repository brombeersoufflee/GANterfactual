from swin_transformer import SwinTransformer
import torch
import monai
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dataloader import DataLoader


# Data loading and preprocessing
# Justification?
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale images
])

train_dataset = torchvision.datasets.ImageFolder(root="data3d/train", transform=transform)
val_dataset = torchvision.datasets.ImageFolder(root="data3d/val",transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))
classes = ["Negative", "Positive"]

# Sanity check: Display some images from the training set
# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(train_dataloader)
images, labels = dataiter.next()

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print('  '.join(classes[labels[j]] for j in range(4)))


# Swin Transformer model definition
patch_size = monai.utils.ensure_tuple_rep(2, 3)
window_size = monai.utils.ensure_tuple_rep(7, 3)
SwinViT = SwinTransformer(
            in_chans=1,
            embed_dim=96, # args.feature_size 48 # embed is patchsize* channel_size
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 18, 2], # if 18 then swin-s and if 6 then swin-t
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            spatial_dims=3,
            )

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
dummy_outputs = torch.rand(4, 2)
# Represents the correct class among the 10 being tested
dummy_labels = torch.tensor([1, 0, 1, 0])
    
print(dummy_outputs)
print(dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print('Total loss for this batch: {}'.format(loss.item()))

# Optimizer
optimizer = torch.optim.SGD(SwinViT.parameters(), lr=0.001, momentum=0.9)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        # Make predictions for this batch
        outputs = SwinViT(inputs)
        
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        # change the magic number 1000 to the number of batches in dataset
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            
    return last_loss

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = torch.utils.tensorboard.SummaryWriter('runs/Swin_trainer_{}'.format(timestamp))
epoch_number = 0


EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    
    # Make sure gradient tracking is on, and do a pass over the data
    SwinViT.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    
    # We don't need gradients on to do reporting
    SwinViT.train(False)
    
    running_vloss = 0.0
    for i, vdata in enumerate(val_dataloader):
        vinputs, vlabels = vdata
        voutputs = SwinViT(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss
    
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    
    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()
    
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'SwinViT_{}_{}'.format(timestamp, epoch_number)
        torch.save(SwinViT.state_dict(), model_path)
    
    epoch_number += 1


# Dummy data
inputs = torch.randn(1, 1, 10, 10, 10)  # 10 samples, each with 10 features
labels = torch.randint(0, 2, (10,))  # Random integer labels (0 or 1) for 10 samples

# Training loop
SwinViT.train()  # Set the model to training mode
optimizer.zero_grad()  # Zero the gradients
outputs = SwinViT(inputs)  # Forward pass
print("Model Outputs (before softmax):\n", outputs)

loss = criterion(outputs, labels)  # Compute the loss
print("Loss:", loss.item())

loss.backward()  # Backward pass (compute gradients)
optimizer.step()  # Update model parameters

# Updated model parameters for fc1 layer (optional)
print("\nUpdated fc1 weights:\n", SwinViT.fc1.weight)
print("Updated fc1 biases:\n", SwinViT.fc1.bias)