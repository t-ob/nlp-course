import argparse
import typing
from typing import Optional, Tuple

import torch

S_TO_I = {c: i for i, c in enumerate(".abcdefghijklmnopqrstuvwxyz")}

I_TO_S = {i: c for c, i in S_TO_I.items()}


def get_lines(path: str) -> list[str]:
    lines: list[str] = []
    with open(path) as fd:
        for line in fd:
            lines.append(line.strip())
    return lines


def get_bigrams(words: list[str]) -> list[Tuple[str, str]]:
    bigrams: list[Tuple[str, str]] = []
    for word in words:
        word = ["."] + list(word) + ["."]
        for bigram in zip(word, word[1:]):
            bigrams.append(bigram)
    return bigrams


def build_bigram_model(bigrams: list[Tuple[str, str]]) -> torch.Tensor:
    counts = torch.zeros((27, 27), dtype=torch.int)
    for bigram in bigrams:
        c1, c2 = bigram
        counts[S_TO_I[c1], S_TO_I[c2]] += 1

    # model smoothing to avoid -infs when taking log probabilities
    smoothed_counts = (counts + 1).float()
    model = smoothed_counts / smoothed_counts.sum(dim=1, keepdim=True)

    return model


def sample_from_bigram_model(
    model: torch.Tensor, c: str = ".", generator: Optional[torch.Generator] = None
) -> str:
    chars: list[str] = []
    while True:
        i = S_TO_I[c]
        next_i = typing.cast(
            int,
            torch.multinomial(
                model[i], num_samples=1, replacement=True, generator=generator
            ).item(),
        )
        c = I_TO_S[next_i]
        if c == ".":
            break
        chars.append(c)
    return "".join(chars)


def compute_negative_log_likelihood(
    model: torch.Tensor, words: list[str]
) -> torch.Tensor:
    log_likelihood = torch.scalar_tensor(0)
    bigrams = get_bigrams(words)
    for bigram in bigrams:
        c1, c2 = bigram
        prob = model[S_TO_I[c1], S_TO_I[c2]]
        log_prob = torch.log(prob)
        log_likelihood += log_prob

    log_likelihood /= len(bigrams)

    return -log_likelihood


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

    args = parser.parse_args()

    names = get_lines(args.path)
    bigrams = get_bigrams(names)

    P = build_bigram_model(bigrams)

    g = torch.Generator(device="cpu").manual_seed(args.manual_seed)

    for _ in range(args.samples):
        print("".join(sample_from_bigram_model(P, generator=g)))

    negative_log_likelihood = compute_negative_log_likelihood(P, names)

    print(f"{negative_log_likelihood=}")
