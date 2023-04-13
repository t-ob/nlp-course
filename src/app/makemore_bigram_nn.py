import argparse
import typing

import torch

S_TO_I = {c: i for i, c in enumerate(".abcdefghijklmnopqrstuvwxyz")}

I_TO_S = {i: c for c, i in S_TO_I.items()}


def get_lines(path: str) -> list[str]:
    lines: list[str] = []
    with open(path) as fd:
        for line in fd:
            lines.append(line.strip())
    return lines


def get_bigrams(words: list[str]) -> list[tuple[str, str]]:
    bigrams: list[tuple[str, str]] = []
    for word in words:
        word = ["."] + list(word) + ["."]
        for bigram in zip(word, word[1:]):
            bigrams.append(bigram)
    return bigrams


def build_training_set(
    words: list[str], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    bigrams = get_bigrams(words)
    for c1, c2 in bigrams:
        xs.append(S_TO_I[c1])
        ys.append(S_TO_I[c2])
    return torch.tensor(xs, device=device), torch.tensor(ys, device=device)


def sample_from_model(
    model: torch.Tensor,
    c: str = ".",
    generator: typing.Optional[torch.Generator] = None,
) -> str:
    chars: list[str] = []
    probs = model.softmax(dim=1)
    while True:
        i = S_TO_I[c]
        next_i = typing.cast(
            int,
            torch.multinomial(
                probs[i], num_samples=1, replacement=True, generator=generator
            ).item(),
        )
        c = I_TO_S[next_i]
        if c == ".":
            break
        chars.append(c)
    return "".join(chars)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple bigram model.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
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
    parser.add_argument("--epochs", type=int, default=1000, help="Epoch count")
    parser.add_argument(
        "--learning-rate", type=float, default=50.0, help="Learning rate"
    )
    parser.add_argument(
        "--regularization-factor",
        type=float,
        default=0.01,
        help="Regularization loss factor",
    )
    parser.add_argument(
        "--print-model-samples", type=int, help="Number of output samples to print"
    )

    args = parser.parse_args()

    g = torch.Generator(device=args.device).manual_seed(args.manual_seed)

    names = get_lines(args.path)
    if args.training_sample_size is not None:
        names = names[: args.training_sample_size]

    # Create training set of bigrams x, y
    xs_train, ys_train = build_training_set(names, device=args.device)
    # print(f"{xs_train=}, {ys_train=}")

    # dim 0 for xs, dim 1 for number of neurons. We choose 27 because we want to model a
    # probability distribution (for each character) of next characters
    num_neurons = 27
    W = torch.randn(
        size=(27, num_neurons), generator=g, requires_grad=True, device=args.device
    )

    epochs = args.epochs
    regularization_factor = torch.scalar_tensor(
        args.regularization_factor, device=args.device
    )
    learning_rate = torch.scalar_tensor(args.learning_rate, device=args.device)

    for epoch in range(epochs):
        # Forward pass

        logits = W[xs_train]
        probs = logits.softmax(dim=1)

        # negative log likelihood
        loss = (
            -probs[torch.arange(ys_train.shape[0]), ys_train].log().mean()
            + regularization_factor * (W**2).mean()
        )
        if epoch % 100 == 0:
            print(f"{loss=}")
        elif epoch == epochs - 1:
            print(f"(final) {loss=}")

        # Backward pass
        W.grad = None
        loss.backward()

        # Update
        W.data -= learning_rate * typing.cast(torch.Tensor, W.grad)

    print(f"{W=}")

    if args.print_model_samples is not None:
        for _ in range(args.print_model_samples):
            print(sample_from_model(W, generator=g))
