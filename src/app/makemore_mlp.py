import argparse
import typing

import torch
import torch.nn.functional as F

import wandb


MLP = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


S_TO_I = {c: i for i, c in enumerate(".abcdefghijklmnopqrstuvwxyz")}

I_TO_S = {i: c for c, i in S_TO_I.items()}


def get_lines(path: str) -> list[str]:
    lines: list[str] = []
    with open(path) as fd:
        for line in fd:
            lines.append(line.strip())
    return lines


def build_datasets(
    words: list[str], block_size: int, device: str, generator: torch.Generator
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    xs = []
    ys = []

    # xs contains context windows of size `block_size` of character indexes
    # ys contains character indexes of immediately following characters
    for w in words:
        context = [0] * block_size
        for c in w + ".":
            xs.append(context)
            ix = S_TO_I[c]
            ys.append(ix)
            context = context[1:] + [ix]

    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)

    distribution = torch.multinomial(
        torch.tensor([0.8, 0.1, 0.1], device=device),
        ys.shape[0],
        replacement=True,
        generator=generator,
    )

    xs_train = xs[distribution == 0]
    ys_train = ys[distribution == 0]
    xs_dev = xs[distribution == 1]
    ys_dev = ys[distribution == 1]
    xs_test = xs[distribution == 2]
    ys_test = ys[distribution == 2]

    return xs_train, ys_train, xs_dev, ys_dev, xs_test, ys_test


def build_model(
    context_length: int,
    embedding_dim: int,
    hidden_layer_neurons: int,
    device: str,
    generator: torch.Generator,
) -> MLP:
    lookup = torch.randn(
        (27, embedding_dim), generator=generator, device=device, requires_grad=True
    )
    w1 = torch.randn(
        (context_length * embedding_dim, hidden_layer_neurons),
        device=device,
        generator=generator,
        requires_grad=True,
    )
    b1 = torch.randn(
        hidden_layer_neurons, generator=generator, device=device, requires_grad=True
    )

    w2 = torch.randn(
        (hidden_layer_neurons, 27),
        generator=generator,
        device=device,
        requires_grad=True,
    )
    b2 = torch.randn(27, device=device, generator=generator, requires_grad=True)

    return (lookup, w1, b1, w2, b2)


def forward_pass(
    model: MLP,
    xs: torch.Tensor,
    ys: torch.Tensor,
    regularization_loss_factor: torch.Tensor,
) -> torch.Tensor:
    lookup_table, w1, b1, w2, b2 = model

    emb = lookup_table[xs].view(xs.shape[0], -1)

    h = (emb @ w1 + b1).tanh()

    logits = h @ w2 + b2

    loss = F.cross_entropy(logits, ys) + regularization_loss_factor * (
        (w1**2).mean() + (w2**2).mean()
    )

    return loss


def fit_model(
    model: MLP,
    xs: torch.Tensor,
    ys: torch.Tensor,
    batch_size: int,
    epochs: int,
    learning_rate: torch.Tensor,
    regularization_loss_factor: torch.Tensor,
    device: str,
    enable_wandb: bool,
):
    powers = torch.arange(epochs, device=device) // 100000
    lr_decays = torch.pow(10.0, -powers)

    for epoch in range(epochs):
        batch_idx = torch.randint(
            0, xs.shape[0], (batch_size,), generator=generator, device=device
        )
        training_loss = forward_pass(
            model, xs[batch_idx], ys[batch_idx], regularization_loss_factor
        )
        if epoch % 1000 == 0:
            if enable_wandb:
                wandb.log({"training_loss": training_loss})
            print(f"{epoch=} {(lr_decays[epoch] * learning_rate)=} {training_loss=}")

        for p in model:
            p.grad = None

        training_loss.backward()

        for p in model:
            p.data -= (
                lr_decays[epoch] * learning_rate * typing.cast(torch.Tensor, p.grad)
            )


def sample_model(
    model: MLP,
    context_length: int,
    generator: torch.Generator,
    prefix: typing.Optional[str],
):
    sampled_chars = []
    with torch.no_grad():
        lookup_table, w1, b1, w2, b2 = model

        context = [0] * context_length
        if prefix is not None:
            prefix_context = prefix[-context_length:]
            for i, c in enumerate(reversed(prefix_context)):
                context[(context_length - 1 - i)] = S_TO_I[c]

        while True:
            emb = (lookup_table[context]).view(-1)

            h = (emb @ w1 + b1).tanh()

            logits = h @ w2 + b2

            probs = logits.softmax(dim=0)

            next_i = typing.cast(
                int,
                torch.multinomial(
                    probs, num_samples=1, replacement=True, generator=generator
                ).item(),
            )

            if next_i == 0:
                break

            sampled_chars.append(I_TO_S[next_i])
            context = context[1:] + [next_i]

    return (prefix or "") + "".join(sampled_chars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple bigram model.")
    parser.add_argument(
        "--path",
        default="data/names.txt",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--manual-seed",
        type=int,
        default=2147483647,
        help="Manual seed",
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda:0)")
    parser.add_argument("--training-sample-size", type=int)
    parser.add_argument("--epochs", type=int, default=10, help="Epoch count")
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "--regularization-loss-factor",
        type=float,
        default=0.01,
        help="Regularization loss factor",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2,
        help="Dimension of character embedding space",
    )
    parser.add_argument(
        "--context-length", type=int, default=3, help="Training context window length"
    )
    parser.add_argument(
        "--hidden-layer-neurons", type=int, default=100, help="Neurons in hidden layer"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size during SGD"
    )
    parser.add_argument(
        "--print-model-samples", type=int, help="Number of output samples to print"
    )
    parser.add_argument("--sample-prefix", help="Optional sample prefix")
    parser.add_argument(
        "--enable-wandb", action="store_true", help="Enable Weights and Biases"
    )
    parser.add_argument(
        "--wandb-project",
        default="makemore-mlp-dev",
        help="Weights and Biases project to use",
    )

    args = parser.parse_args()

    enable_wandb = args.enable_wandb
    wandb_project = args.wandb_project

    # Script params
    path = args.path
    device = args.device
    print_model_samples = args.print_model_samples or 0
    training_sample_size = args.training_sample_size
    manual_seed = args.manual_seed
    sample_prefix = args.sample_prefix

    print(
        (
            f"SCRIPT PARAMS:\n"
            f"{path=}\n{device=}\n{print_model_samples=}\n"
            f"{training_sample_size=}\n{manual_seed=}\n{sample_prefix=}\n"
        )
    )

    # Model hyperparams
    epochs: int = args.epochs
    learning_rate: torch.Tensor = torch.tensor(args.learning_rate, device=device)
    regularization_loss_factor: torch.Tensor = torch.tensor(
        args.regularization_loss_factor, device=device
    )
    embedding_dim: int = args.embedding_dim
    hidden_layer_neurons: int = args.hidden_layer_neurons
    context_length: int = args.context_length
    batch_size: int = args.batch_size

    if enable_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_project,
            # track hyperparameters and run metadata
            config={
                "training_sample_size": training_sample_size,
                "manual_seed": manual_seed,
                "learning_rate": learning_rate,
                "regularization_loss_factor": regularization_loss_factor,
                "embedding_dim": embedding_dim,
                "hidden_layer_neurons": hidden_layer_neurons,
                "context_length": context_length,
                "batch_size": batch_size,
                "epochs": epochs,
            },
        )

    print(
        (
            f"MODEL PARAMS:\n"
            f"{epochs=}\n{learning_rate=}\n{regularization_loss_factor=}\n"
            f"{embedding_dim=}\n{hidden_layer_neurons=}\n{context_length=}\n"
            f"{batch_size=}\n"
        )
    )

    generator = torch.Generator(device=device).manual_seed(args.manual_seed)

    names = get_lines(path)
    if training_sample_size is not None:
        names = names[:training_sample_size]

    # Create datasets of embedded characters and their labels
    xs_train, ys_train, xs_dev, ys_dev, xs_test, ys_test = build_datasets(
        names, block_size=context_length, device=device, generator=generator
    )

    model = build_model(
        context_length=context_length,
        embedding_dim=embedding_dim,
        hidden_layer_neurons=hidden_layer_neurons,
        device=device,
        generator=generator,
    )

    fit_model(
        model,
        xs=xs_train,
        ys=ys_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        regularization_loss_factor=regularization_loss_factor,
        device=device,
        enable_wandb=enable_wandb,
    )

    with torch.no_grad():
        dev_loss = forward_pass(model, xs_dev, ys_dev, regularization_loss_factor)
        if enable_wandb:
            wandb.log({"dev_loss": dev_loss})
        print(f"{dev_loss=}")

    for _ in range(print_model_samples):
        sample = sample_model(
            model=model,
            context_length=context_length,
            generator=generator,
            prefix=sample_prefix,
        )
        print(f"{sample=}")
