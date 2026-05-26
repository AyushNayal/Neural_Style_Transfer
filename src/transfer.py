import torch
import torch.optim as optim
from src.utils import tensor_to_img


def run_style_transfer(
    model,
    content_losses,
    style_losses,
    tv_loss,
    input_img,
    num_steps=2000,
    style_weight=1e6,
    content_weight=2.0,
    tv_weight=1e-5,
    save_every=500,
):
    """
    Core optimization loop for neural style transfer.

    Instead of training a network, we optimize the pixel values
    of the input image directly to minimize the combined loss.

    Args:
        model          : built VGG model with loss probes inserted
        content_losses : list of ContentLoss modules
        style_losses   : list of StyleLoss modules
        tv_loss        : TotalVariationLoss module
        input_img      : starting image tensor (clone of content image)
        num_steps      : number of optimization iterations
        style_weight   : how strongly to enforce style
        content_weight : how strongly to preserve content structure
        tv_weight      : how strongly to enforce smoothness
        save_every     : save intermediate output every N steps

    Returns:
        input_img            : final optimized image tensor
        intermediate_outputs : list of (step, PIL Image) tuples
    """

    # The input image is the only thing being optimized
    input_img.requires_grad_(True)

    optimizer = optim.Adam([input_img], lr=0.005)

    intermediate_outputs = []

    print("Starting optimization...")
    print(f"Steps: {num_steps} | Style weight: {style_weight} | "
          f"Content weight: {content_weight} | TV weight: {tv_weight}\n")

    for step in range(1, num_steps + 1):

        optimizer.zero_grad()

        # Forward pass through the model — loss probes record losses internally
        model(input_img)

        # Collect and scale losses
        style_score   = sum(sl.loss for sl in style_losses) * style_weight
        content_score = sum(cl.loss for cl in content_losses) * content_weight

        # TV loss operates on the raw image, not VGG features
        tv_loss(input_img)
        tv_score = tv_loss.loss

        total_loss = style_score + content_score + tv_score
        total_loss.backward()

        optimizer.step()

        # Keep pixel values valid after each step
        with torch.no_grad():
            input_img.clamp_(0, 1)

        if step % 100 == 0 or step == num_steps:
            print(
                f"Step {step:>5}/{num_steps} | "
                f"Total: {total_loss.item():.4f} | "
                f"Style: {style_score.item():.4f} | "
                f"Content: {content_score.item():.4f} | "
                f"TV: {tv_score.item():.4f}"
            )

        if step % save_every == 0:
            with torch.no_grad():
                intermediate_outputs.append(
                    (step, tensor_to_img(input_img.cpu().clone()))
                )

    print("\nOptimization complete.")
    return input_img, intermediate_outputs