import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def load_image(image_path, max_size=512, device=None):
    """Load an image from disk, resize if needed, and convert to tensor."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_path).convert('RGB')

    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = tuple([int(dim * scale) for dim in image.size])
    else:
        new_size = image.size

    transform = transforms.Compose([
        transforms.Resize(new_size[::-1]),  # PIL is (W,H), PyTorch expects (H,W)
        transforms.ToTensor(),
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)


def tensor_to_img(tensor):
    """Convert a tensor back to a PIL Image."""
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    return transforms.ToPILImage()(image)


def preprocess_tensor(tensor):
    """Apply VGG19 normalization to a tensor."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def show_images(content_tensor, style_tensor):
    """Display content and style images side by side."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_img(content_tensor))
    plt.title('Content Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_img(style_tensor))
    plt.title('Style Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)


def show_output(output_tensor, title='Output Image'):
    """Display the final stylized output image."""
    plt.figure(figsize=(6, 6))
    plt.imshow(tensor_to_img(output_tensor))
    plt.title(title)
    plt.axis('off')
    plt.show()


def show_intermediates(intermediate_outputs):
    """Display intermediate outputs saved during optimization."""
    num_images = len(intermediate_outputs)
    if num_images == 0:
        return

    fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axs = [axs]

    for ax, (step, img) in zip(axs, intermediate_outputs):
        ax.imshow(img)
        ax.set_title(f'Step {step}')
        ax.axis('off')

    plt.tight_layout()
    plt.show(block=False)


def save_image(tensor, path):
    """Save output tensor as an image file."""
    img = tensor_to_img(tensor)
    img.save(path)
    print(f"Image saved to {path}")