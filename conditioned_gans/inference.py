from models import Generator
import torch
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on", device)

LATENT_DIM=100
latenet_space = torch.randn(1, LATENT_DIM, device=device)

generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth"))

print("loaded generator successfully")

pred = generator(latenet_space).detach().cpu().squeeze(0)
plt.imshow(pred.permute(1,2,0), cmap="gray")
plt.show()