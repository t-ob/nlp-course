import argparse
import typing

import torch
import torch.nn.functional as F

S_TO_I = {c: i for i, c in enumerate(".abcdefghijklmnopqrstuvwxyz")}

I_TO_S = {i: c for c, i in S_TO_I.items()}


def get_lines(path: str) -> list[str]:
    lines: list[str] = []
    with open(path) as fd:
        for line in fd:
            lines.append(line.strip())
    return lines


def get_trigrams(words: list[str]) -> list[tuple[str, str, str]]:
    trigrams: list[tuple[str, str, str]] = []
    for word in words:
        word = ["."] + list(word) + ["."]
        for trigram in zip(word, word[1:], word[2:]):
            trigrams.append(trigram)
    return trigrams


def build_datasets(
    words: list[str], device: str, generator: torch.Generator
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    xs = []
    ys = []

    trigrams = get_trigrams(words)
    for c1, c2, c3 in trigrams:
        xs.append([S_TO_I[c1], S_TO_I[c2]])
        ys.append(S_TO_I[c3])

    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)

    distribution = torch.multinomial(
        torch.tensor([0.8, 0.1, 0.1]),
        len(trigrams),
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


def sample_from_model(
    probs: torch.Tensor,
    bigram: tuple[str, str] = (".", "."),
    generator: typing.Optional[torch.Generator] = None,
) -> str:
    chars: list[str] = []
    while True:
        c1, c2 = bigram
        i1, i2 = S_TO_I[c1], S_TO_I[c2]
        next_i = typing.cast(
            int,
            torch.multinomial(
                probs[i1, i2], num_samples=1, replacement=True, generator=generator
            ).item(),
        )
        c3 = I_TO_S[next_i]
        if c3 == ".":
            break
        bigram = (c2, c3)
        chars.append(c3)
    return "".join(chars)


def forward_pass(
    model: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor
) -> torch.Tensor:
    logits = model[xs[:, 0], xs[:, 1]]

    # negative log likelihood
    loss = F.cross_entropy(logits, ys) + regularization_factor * (W**2).mean()

    return loss


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
    parser.add_argument(
        "--eval-test", action="store_true", help="Evaluate on test split"
    )

    args = parser.parse_args()

    g = torch.Generator(device=args.device).manual_seed(args.manual_seed)

    names = get_lines(args.path)
    if args.training_sample_size is not None:
        names = names[: args.training_sample_size]

    # Create training set of trigrams x, y
    xs_train, ys_train, xs_dev, ys_dev, xs_test, ys_test = build_datasets(
        names, device=args.device, generator=g
    )

    # dim 0, 1 for xs, dim 2 for number of neurons. We choose 27 for number of neurons
    # because we want to model a probability distribution (for each character) of next
    # characters
    num_neurons = 27
    W = torch.randn(
        size=(27, 27, num_neurons), generator=g, requires_grad=True, device=args.device
    )

    epochs = args.epochs
    regularization_factor = torch.scalar_tensor(
        args.regularization_factor, device=args.device
    )
    learning_rate = torch.scalar_tensor(args.learning_rate, device=args.device)

    for epoch in range(epochs):
        # Forward pass
        training_loss = forward_pass(model=W, xs=xs_train, ys=ys_train)
        if epoch % 100 == 0:
            print(f"{training_loss=}")

        # Backward pass
        W.grad = None
        training_loss.backward()

        # Update
        W.data -= learning_rate * typing.cast(torch.Tensor, W.grad)

    with torch.no_grad():
        dev_loss = forward_pass(model=W, xs=xs_dev, ys=ys_dev)
        print(f"{dev_loss=}")

        if args.eval_test:
            test_loss = forward_pass(model=W, xs=xs_test, ys=ys_test)
            print(f"{test_loss=}")

    P = W.softmax(dim=2)

    if args.print_model_samples is not None:
        for _ in range(args.print_model_samples):
            print(sample_from_model(P, generator=g))
