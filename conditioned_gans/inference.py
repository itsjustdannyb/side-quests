from models import CondGenerator
import torch
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on", device)



def run_inference(target_label, num_preds:int=5, latent_dim:int=100):
    latent_space = torch.randn(num_preds, latent_dim, device=device)

    generator = CondGenerator().to(device)
    generator.load_state_dict(torch.load("generator.pth", weights_only=True))

    print("loaded generator successfully")

    # The class we want to generate
    target_labels = torch.full((num_preds,), target_label, dtype=torch.long, device=device)

    preds = generator(latent_space, target_labels).detach().cpu()

    fig, axs = plt.subplots(nrows=1, ncols=num_preds, figsize=(10, 5))
    if num_preds == 1:
        axs = [axs]

    for i in range(num_preds):
        img = preds[i].squeeze(0) # shape (28, 28)
        axs[i].imshow(img, cmap="gray")
        axs[i].set_title(f"Label: {target_label}")
        axs[i].axis("off")

    plt.suptitle("GANS are hard to train...")
    plt.savefig(f"generated_images_for_{target_label}.png")
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, metavar="", default=5, help="number of samples to plot")
parser.add_argument("--latent_dim", type=int, metavar="", default=100, help="dimension of latent space for generation")
parser.add_argument("--target_label", type=int, metavar="", required=True, help="number(0-9) to generate")

args = parser.parse_args()

if __name__ == "__main__":
    run_inference(num_preds=args.num_samples, latent_dim=args.latent_dim, target_label=args.target_label)