import torch
import torch.nn as nn
import torch.nn.init as init

# weight initialization helps models converge better and faster, It helps prevent vanishing an exploding gradients
# it also helps to break symmetry, prevents all neurons from learning the same thing.

# you use different initializations based on the model architecture
# xavier (glorot) initialization -> for tanh and sigmoid activations
# kaiming (he) initialization for relu and relu variants (leaky relu...)
# constant intialization is often used for biases to ensure a neurtal starting point

# reproducibility
torch.manual_seed(42)

dummy_model = nn.Sequential(
    nn.Conv1d(1, 12, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(12, 12)
)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias) # make bias 0, bias can be zero becasuse they act as an offset for the activatio function, they dont necessarily help 'break symmetry'
    elif isinstance(m, nn.Conv1d):
        init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity='relu')
        # fan_out -> preserves weight magnitudes in the backward pass
        # fan_in -> preserves weight magitudes in the forward pass


# compare before and after weight init
print("Before weight initialization: \n###############################")
for child in dummy_model.children():
    print(child.weight[0][0]) # view first set of weights
    break
print("############################ \n")

# to initialize model weights
dummy_model.apply(initialize_weights)

print("After weight initialization: \n###############################")
for child in dummy_model.children():
    print(child.weight[0][0])
    break
print("############################")