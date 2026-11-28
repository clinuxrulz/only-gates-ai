#!/usr/bin/env python3
"""Masked-completion digit generator built from 4-input LUT cells.

The classifier in ``run_colab.py`` maps 784 pixels to 10 votes. This script
turns the same primitive into a generator: the input becomes

    (known pixels, mask, class one-hot)      784 + 784 + 10 = 1578 bits

and the output is the 784 pixels again. Training is masked completion -- hide a
random subset of pixels and predict them. Sampling starts from a blank canvas
and reveals the most confident pixels over a few rounds, so it draws a digit
the way a person does: a few strokes first, then the rest.

Every input and every output is a bit, so the trained model hardens to a pure
LUT4 netlist. That is the reason for masked completion over Gaussian diffusion:
a continuous epsilon estimate would need an output resolution the vote readout
does not have, while a mask is already binary and needs no timestep embedding.
"""

import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

WIRE_NAMES = ("wire_a", "wire_b", "wire_c", "wire_d")
PIXELS = 784
CLASSES = 10


def random_logits(shape, generator):
    return nn.Parameter(
        torch.randn([int(size) for size in shape], generator=generator)
    )


class GateDense4(nn.Module):
    """Four-pin cell with an arbitrary 4-input Boolean gate (16-bit table)."""

    def __init__(self, in_features, out_features, generator):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        for name in WIRE_NAMES:
            setattr(
                self, name, random_logits([out_features, in_features], generator)
            )
        self.table = nn.Parameter(
            torch.randn(out_features, 16, generator=generator) * 0.5
        )
        self.temperature = 1.0
        self.pins = ()

    def forward(self, x):
        wires = [
            F.softmax(getattr(self, name) / self.temperature, dim=1)
            for name in WIRE_NAMES
        ]
        values = [x @ wire.t() for wire in wires]
        weights = [torch.stack([1.0 - value, value], dim=-1) for value in values]
        weight = (
            weights[0].unsqueeze(3).unsqueeze(4).unsqueeze(5)
            * weights[1].unsqueeze(2).unsqueeze(4).unsqueeze(5)
            * weights[2].unsqueeze(2).unsqueeze(3).unsqueeze(5)
            * weights[3].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        ).reshape(x.shape[0], self.out_features, 16)
        table = torch.sigmoid(self.table / self.temperature)
        return (weight * table.unsqueeze(0)).sum(dim=-1)

    def hard_forward(self, x):
        pins = [
            x[:, getattr(self, name).argmax(dim=1)].float() for name in WIRE_NAMES
        ]
        index = sum(
            pin * (8 >> position) for position, pin in enumerate(pins)
        ).long().unsqueeze(-1)
        bits = (self.table > 0).float()
        return torch.gather(
            bits.unsqueeze(0).expand(x.shape[0], -1, -1), 2, index
        ).squeeze(-1)


class GateNet4(nn.Module):
    def __init__(self, input_size, hidden, output_size, generator):
        super().__init__()
        layers = []
        previous = input_size
        for size in [*hidden, output_size]:
            layers.append(GateDense4(previous, size, generator))
            previous = size
        self.layers = nn.ModuleList(layers)
        self.output_size = output_size

    def forward(self, x):
        value = x.flatten(1)
        for layer in self.layers:
            value = layer(value)
        return value

    def hard_forward(self, x):
        value = x.bool().flatten(1).float()
        for layer in self.layers:
            value = layer.hard_forward(value)
        return value


class GateConv4(nn.Module):
    """Three-by-three cell with a shared four-pin LUT.

    Wires and the 16-bit table are shared across every spatial position, which
    is the inductive bias the dense forest lacks: neighbouring output pixels
    then use the same detector, so strokes can be continuous.
    """

    def __init__(self, in_channels, filters, generator, kernel=3):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.kernel = kernel
        self.padding = kernel // 2
        fan_in = kernel * kernel * in_channels
        for name in WIRE_NAMES:
            setattr(self, name, random_logits([filters, fan_in], generator))
        self.table = nn.Parameter(
            torch.randn(filters, 16, generator=generator) * 0.5
        )
        self.temperature = 1.0
        self.pins = ()

    def _output_shape(self, x):
        height = (x.shape[2] + 2 * self.padding - self.kernel) + 1
        width = (x.shape[3] + 2 * self.padding - self.kernel) + 1
        return height, width

    def forward(self, x):
        patches = F.unfold(x, self.kernel, padding=self.padding)
        wires = [
            F.softmax(getattr(self, name) / self.temperature, dim=1)
            for name in WIRE_NAMES
        ]
        values = [torch.einsum("bfl,cf->blc", patches, wire) for wire in wires]
        weights = [torch.stack([1.0 - value, value], dim=-1) for value in values]
        weight = (
            weights[0].unsqueeze(4).unsqueeze(5).unsqueeze(6)
            * weights[1].unsqueeze(3).unsqueeze(5).unsqueeze(6)
            * weights[2].unsqueeze(3).unsqueeze(4).unsqueeze(6)
            * weights[3].unsqueeze(3).unsqueeze(4).unsqueeze(5)
        )
        table = torch.sigmoid(self.table / self.temperature).view(
            self.filters, 2, 2, 2, 2
        )
        result = (weight * table.unsqueeze(0).unsqueeze(0)).sum(
            dim=(3, 4, 5, 6)
        )
        height, width = self._output_shape(x)
        return result.permute(0, 2, 1).reshape(
            x.shape[0], self.filters, height, width
        )

    def hard_forward(self, x):
        patches = F.unfold(x.float(), self.kernel, padding=self.padding)
        pins = [
            patches[:, getattr(self, name).argmax(dim=1), :] for name in WIRE_NAMES
        ]
        index = sum(
            pin * (8 >> position) for position, pin in enumerate(pins)
        ).long()
        bits = (self.table > 0).float()
        table = bits.unsqueeze(0).expand(x.shape[0], -1, -1)
        result = torch.gather(table, 2, index)
        height, width = self._output_shape(x)
        return result.reshape(x.shape[0], self.filters, height, width)


class GateConvGen(nn.Module):
    """Fully convolutional generator at full 28x28 resolution.

    Input channels are (masked image, mask, ten broadcast class bits), so the
    class conditions every position without breaking translation equivariance.
    """

    def __init__(self, channels, layers, votes, generator):
        super().__init__()
        self.layers = nn.ModuleList()
        previous = 2 + CLASSES
        for _ in range(max(1, layers - 1)):
            self.layers.append(GateConv4(previous, channels, generator))
            previous = channels
        self.layers.append(GateConv4(previous, votes, generator))
        self.output_size = PIXELS * votes

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)

    def hard_forward(self, x):
        for layer in self.layers:
            x = layer.hard_forward(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)



# ---------------------------------------------------------------------------
# Wire pinning
# ---------------------------------------------------------------------------


def _wire_shape(module):
    if isinstance(module, GateConv4):
        return module.filters, module.kernel * module.kernel * module.in_channels
    if isinstance(module, GateDense4):
        return module.out_features, module.in_features
    return None


def pin_wires(model, generator, fraction):
    """Pin wire rows to a joint round-robin cover.

    The classifier pins each pin independently, which is fine at 784 inputs.
    Here the input is 1578 bits, so four independent covers would miss a
    sizeable slice of the input. Drawing all four pins from one shuffled pool
    guarantees every input bit reaches at least one pinned wire.
    """
    for module in model.modules():
        shape = _wire_shape(module)
        if shape is None:
            continue
        count, fan_in = shape
        keep = int(round(count * fraction))
        if keep <= 0:
            continue
        pool = []
        while len(pool) < len(WIRE_NAMES) * keep:
            pool.extend(torch.randperm(fan_in, generator=generator).tolist())
        pins = []
        for index, name in enumerate(WIRE_NAMES):
            rows = torch.randperm(count, generator=generator)[:keep]
            slots = torch.tensor(
                pool[index * keep : (index + 1) * keep], device=module.table.device
            )
            pins.append((rows, slots))
            with torch.no_grad():
                parameter = getattr(module, name)
                parameter[rows] = 0.0
                parameter[rows, slots] = 12.0
        module.pins = tuple(pins)


def reimpose_pinned(model):
    """Restore pinned wire rows after an optimiser step."""
    for module in model.modules():
        for name, (rows, slots) in zip(WIRE_NAMES, getattr(module, "pins", ())):
            with torch.no_grad():
                parameter = getattr(module, name)
                parameter[rows] = 0.0
                parameter[rows, slots] = 12.0


def set_temperature(model, value):
    for module in model.modules():
        if isinstance(module, (GateDense4, GateConv4)):
            module.temperature = value


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_digits(train_limit, test_limit, device):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        "/tmp/mnist", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        "/tmp/mnist", train=False, download=True, transform=transform
    )

    def pack(dataset, limit):
        total = len(dataset) if limit <= 0 else min(limit, len(dataset))
        images = torch.stack([dataset[index][0] for index in range(total)])
        labels = torch.tensor([dataset[index][1] for index in range(total)])
        bits = (images >= 0.25).float().flatten(1)
        return bits.to(device), labels.to(device)

    return pack(train_set, train_limit), pack(test_set, test_limit)


def build_input(bits, mask, labels):
    onehot = F.one_hot(labels, CLASSES).float()
    return torch.cat([bits * mask, mask, onehot], dim=1)


def build_conv_input(bits, mask, labels):
    images = (bits * mask).view(-1, 1, 28, 28)
    masks = mask.view(-1, 1, 28, 28)
    onehot = (
        F.one_hot(labels, CLASSES)
        .float()
        .view(-1, CLASSES, 1, 1)
        .expand(-1, -1, 28, 28)
    )
    return torch.cat([images, masks, onehot], dim=1)


def model_input(args, bits, mask, labels):
    if args.arch == "conv":
        return build_conv_input(bits, mask, labels)
    return build_input(bits, mask, labels)


def build_denoise_input(bits, level, labels):
    """Input for the denoising task: (noisy pixels, corruption level, class)."""
    images = bits.view(-1, 1, 28, 28)
    levels = level.view(-1, 1, 1, 1).expand(-1, 1, 28, 28)
    onehot = (
        F.one_hot(labels, CLASSES)
        .float()
        .view(-1, CLASSES, 1, 1)
        .expand(-1, -1, 28, 28)
    )
    return torch.cat([images, levels, onehot], dim=1)


def corrupt(bits, generator, device):
    """Flip each pixel with probability ``level``; returns (noisy, levels)."""
    level = torch.rand(len(bits), generator=generator, device=device)
    flip = (
        torch.rand(bits.shape, generator=generator, device=device) < level.unsqueeze(1)
    ).float()
    return torch.abs(bits - flip), level


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(model, train_x, train_y, args, device, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total = len(train_x)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        ratio = (epoch - 1) / max(1, args.epochs - 1)
        temperature = args.t_start * (args.t_min / args.t_start) ** ratio
        set_temperature(model, temperature)
        model.train()
        permutation = torch.randperm(total, device=device)
        loss_sum = 0.0
        batches = 0
        for start in range(0, total, args.batch):
            indices = permutation[start : start + args.batch]
            bits = train_x[indices]
            labels = train_y[indices]
            keep = torch.rand(len(indices), 1, generator=generator, device=device)
            keep = keep * (args.keep_max - args.keep_min) + args.keep_min
            mask = (
                torch.rand(bits.shape, generator=generator, device=device) < keep
            ).float()
            probability = model(model_input(args, bits, mask, labels))
            probability = probability.reshape(len(indices), -1, args.gen_k).mean(
                dim=2
            )
            unknown = 1.0 - mask
            # MNIST is ~19% ink, so an unweighted pixel loss is dominated by
            # blanks and the model learns to predict "empty" almost everywhere.
            weight = 1.0 + (args.ink_weight - 1.0) * bits
            loss = (
                F.binary_cross_entropy(
                    probability.clamp(1e-4, 1.0 - 1e-4), bits, reduction="none"
                )
                * unknown
                * weight
            ).sum() / (unknown * weight).sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            reimpose_pinned(model)
            loss_sum += loss.item()
            batches += 1
        print(
            f"  epoch {epoch:2d}/{args.epochs} T {temperature:.2f} "
            f"loss {loss_sum / max(1, batches):.4f} {time.time() - started:6.1f}s",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Denoising task
# ---------------------------------------------------------------------------


def train_denoise(model, train_x, train_y, args, device, seed):
    """Learn p(clean | noisy, level, class) by flipping pixels at random rates.

    Unlike masked completion, the corruption level is an input, so the model
    can express "I am mostly guessing" and is never asked to threshold its own
    marginal. That is what lets sampling start from pure noise.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total = len(train_x)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        ratio = (epoch - 1) / max(1, args.epochs - 1)
        temperature = args.t_start * (args.t_min / args.t_start) ** ratio
        set_temperature(model, temperature)
        model.train()
        permutation = torch.randperm(total, device=device)
        loss_sum = 0.0
        batches = 0
        for start in range(0, total, args.batch):
            indices = permutation[start : start + args.batch]
            bits = train_x[indices]
            labels = train_y[indices]
            noisy, level = corrupt(bits, generator, device)
            probability = model(build_denoise_input(noisy, level, labels))
            probability = probability.reshape(len(indices), -1, args.gen_k).mean(
                dim=2
            )
            weight = 1.0 + (args.ink_weight - 1.0) * bits
            loss = (
                F.binary_cross_entropy(
                    probability.clamp(1e-4, 1.0 - 1e-4), bits, reduction="none"
                )
                * weight
            ).sum() / weight.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            reimpose_pinned(model)
            loss_sum += loss.item()
            batches += 1
        print(
            f"  epoch {epoch:2d}/{args.epochs} T {temperature:.2f} "
            f"loss {loss_sum / max(1, batches):.4f} {time.time() - started:6.1f}s",
            flush=True,
        )


def denoise_metrics(models, test_x, test_y, args, device, levels=(0.2, 0.5, 0.7, 0.9)):
    """Recovery quality at several corruption levels."""
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 2)
    for model in models:
        model.eval()
    results = {}
    with torch.no_grad():
        for level_value in levels:
            ink = 0.0
            hits = 0.0
            predicted = 0.0
            for start in range(0, len(test_x), args.batch):
                bits = test_x[start : start + args.batch]
                labels = test_y[start : start + args.batch]
                level = torch.full((len(bits),), level_value, device=device)
                flip = (
                    torch.rand(
                        bits.shape, generator=generator, device=device
                    ) < level_value
                ).float()
                noisy = torch.abs(bits - flip)
                probability = torch.zeros(len(bits), PIXELS, device=device)
                for model in models:
                    probability += model(
                        build_denoise_input(noisy, level, labels)
                    ).reshape(len(bits), -1, args.gen_k).mean(dim=2)
                probability /= len(models)
                prediction = (probability > 0.5).float()
                ink += bits.sum().item()
                hits += (prediction * bits).sum().item()
                predicted += prediction.sum().item()
            results[level_value] = {
                "inkRecall": hits / max(1.0, ink),
                "inkPrecision": hits / max(1.0, predicted),
                "inkIou": hits / max(1.0, ink + predicted - hits),
            }
    return results


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample(models, args, device, labels, rounds=None):
    """Fill a blank canvas over ``rounds`` passes, most confident pixels first.

    With ``--remask`` a fraction of the already-revealed pixels with the least
    confidence is un-revealed each round, so an early mistake can still be
    corrected. Without it the canvas only ever grows, like MaskGIT.

    Passing several models votes their per-pixel probabilities every round,
    the same way the classifier sums circuit votes across a seed forest.
    """
    rounds = rounds or args.rounds
    for model in models:
        model.eval()
    count = len(labels)
    bits = torch.zeros(count, PIXELS, device=device)
    mask = torch.zeros_like(bits)
    for step in range(1, rounds + 1):
        share = (step / rounds) ** args.reveal_power
        features = model_input(args, bits, mask, labels)
        probability = torch.zeros(count, PIXELS, device=device)
        with torch.no_grad():
            for model in models:
                if args.hard:
                    raw = model.hard_forward(features)
                else:
                    raw = model(features)
                probability += raw.reshape(count, -1, args.gen_k).mean(dim=2)
        probability /= len(models)
        confidence = (probability - 0.5).abs()
        wanted = int(round(share * PIXELS))
        held = int(mask[0].sum().item())
        known = mask.bool()
        if step == rounds:
            selected = torch.ones_like(mask, dtype=torch.bool)
        else:
            drop = min(int(round(args.remask * held)), held)
            reveal = max(0, wanted - (held - drop))
            selected = torch.zeros_like(mask, dtype=torch.bool)
            if reveal > 0:
                fresh = confidence.masked_fill(known, -1.0)
                top = fresh.topk(min(reveal, PIXELS), dim=1).indices
                selected.scatter_(1, top, True)
            if drop > 0:
                # Least confident hits are the ones most likely to be wrong.
                shaky = confidence.masked_fill(~known, 2.0)
                worst = shaky.topk(drop, dim=1, largest=False).indices
                dropped = torch.zeros_like(mask, dtype=torch.bool)
                dropped.scatter_(1, worst, True)
                selected = selected | (known & ~dropped)
            else:
                selected = selected | known
        pixels = (probability > 0.5).float()
        bits = torch.where(selected, pixels, bits)
        mask = torch.where(selected, torch.ones_like(mask), mask)
    return bits


def sample_denoise(models, args, device, labels, steps=None):
    """Ancestral sampling: start from noise, denoise, re-noise at a lower level."""
    steps = steps or args.rounds
    for model in models:
        model.eval()
    count = len(labels)
    bits = (torch.rand(count, PIXELS, device=device) < 0.5).float()
    for step in range(steps):
        level = 1.0 - step / steps
        levels = torch.full((count,), level, device=device)
        probability = torch.zeros(count, PIXELS, device=device)
        with torch.no_grad():
            for model in models:
                probability += model(
                    build_denoise_input(bits, levels, labels)
                ).reshape(count, -1, args.gen_k).mean(dim=2)
        probability /= len(models)
        # Sample, never threshold: a marginal of 0.19 must still produce ink.
        clean = (torch.rand(count, PIXELS, device=device) < probability).float()
        next_level = max(0.0, 1.0 - (step + 1) / steps)
        if next_level > 0:
            noise = (torch.rand(count, PIXELS, device=device) < next_level).float()
            bits = torch.abs(clean - noise)
        else:
            bits = clean
    return bits


def generate(models, args, device, labels):
    if args.task == "denoise":
        return sample_denoise(models, args, device, labels)
    return sample(models, args, device, labels)


def completion_metrics(models, test_x, test_y, args, device, keep=0.5):
    """Score held-out completion on the hidden pixels, voting across the forest.

    Plain pixel accuracy is misleading here: predicting nothing scores ~81%
    because most pixels are blank. Ink recall, precision and IoU say whether
    the model can actually put strokes where they belong.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    for model in models:
        model.eval()
    correct = 0
    total = 0
    ink = 0.0
    ink_hits = 0.0
    predicted_ink = 0.0
    with torch.no_grad():
        for start in range(0, len(test_x), args.batch):
            bits = test_x[start : start + args.batch]
            labels = test_y[start : start + args.batch]
            mask = (
                torch.rand(bits.shape, generator=generator, device=device) < keep
            ).float()
            features = model_input(args, bits, mask, labels)
            probability = torch.zeros(len(bits), PIXELS, device=device)
            for model in models:
                if args.hard:
                    raw = model.hard_forward(features)
                else:
                    raw = model(features)
                probability += raw.reshape(len(bits), -1, args.gen_k).mean(dim=2)
            probability /= len(models)
            predicted = (probability > 0.5).float()
            unknown = 1.0 - mask
            correct += ((predicted == bits).float() * unknown).sum().item()
            total += unknown.sum().item()
            ink += (bits * unknown).sum().item()
            ink_hits += (predicted * bits * unknown).sum().item()
            predicted_ink += (predicted * unknown).sum().item()
    return {
        "pixel": correct / max(1, total),
        "inkRecall": ink_hits / max(1.0, ink),
        "inkPrecision": ink_hits / max(1.0, predicted_ink),
        "inkIou": ink_hits / max(1.0, ink + predicted_ink - ink_hits),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class FloatCNN(nn.Module):
    """Independent judge: are the samples recognisable digits?"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, CLASSES)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.flatten(1))


def train_judge(train_x, train_y, test_x, test_y, device, epochs=3):
    judge = FloatCNN().to(device)
    optimizer = torch.optim.Adam(judge.parameters(), lr=1e-3)
    for epoch in range(epochs):
        judge.train()
        permutation = torch.randperm(len(train_x), device=device)
        correct = 0
        for start in range(0, len(train_x), 256):
            indices = permutation[start : start + 256]
            logits = judge(train_x[indices].reshape(-1, 1, 28, 28))
            loss = F.cross_entropy(logits, train_y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(dim=1) == train_y[indices]).sum().item()
        judge.eval()
        with torch.no_grad():
            right = sum(
                (
                    judge(test_x[start : start + 512].reshape(-1, 1, 28, 28))
                    .argmax(dim=1)
                    == test_y[start : start + 512]
                ).sum().item()
                for start in range(0, len(test_x), 512)
            )
        print(
            f"  judge epoch {epoch + 1}/{epochs} train "
            f"{correct / len(train_x) * 100:5.2f}% test "
            f"{right / len(test_x) * 100:5.2f}%",
            flush=True,
        )
    judge.eval()
    return judge


def judge_agreement(judge, samples, labels, device):
    with torch.no_grad():
        logits = judge(samples.reshape(-1, 1, 28, 28))
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Display and export
# ---------------------------------------------------------------------------


def show(bits, label=None):
    if label is not None:
        print(f"--- generated {label} ---")
    array = bits.detach().cpu().reshape(28, 28)
    for row in array:
        print("".join("#" if value > 0.5 else "." for value in row))
    print(flush=True)


def export_hard(model, gen_k):
    if isinstance(model, GateConvGen):
        return export_conv(model, gen_k)
    layers = []
    for layer in model.layers:
        wires = (
            torch.stack(
                [getattr(layer, name).argmax(dim=1) for name in WIRE_NAMES], dim=1
            )
            .flatten()
            .tolist()
        )
        bits = (layer.table > 0).long()
        tables = [
            sum(int(bits[row, index].item()) << index for index in range(16))
            for row in range(bits.shape[0])
        ]
        layers.append({"wires": wires, "tables": tables})
    return {
        "kind": "lut4-generator",
        "shape": [model.layers[0].in_features] + [
            len(layer["tables"]) for layer in layers
        ],
        "inputSize": model.layers[0].in_features,
        "imageSize": PIXELS,
        "outputs": PIXELS,
        "circuitsPerLabel": gen_k,
        "layers": layers,
    }


def export_conv(model, votes):
    """Export a shared-rule conv generator.

    Wire indices address the unfolded 3x3 patch as ``channel * 9 + row * 3 +
    col``, matching ``torch.nn.functional.unfold`` with padding 1. The same
    four wires and 16-bit table apply at every spatial position.
    """
    layers = []
    for layer in model.layers:
        wires = (
            torch.stack(
                [getattr(layer, name).argmax(dim=1) for name in WIRE_NAMES], dim=1
            )
            .flatten()
            .tolist()
        )
        bits = (layer.table > 0).long()
        tables = [
            sum(int(bits[row, index].item()) << index for index in range(16))
            for row in range(bits.shape[0])
        ]
        layers.append(
            {
                "filters": layer.filters,
                "inChannels": layer.in_channels,
                "wires": wires,
                "tables": tables,
            }
        )
    return {
        "kind": "lut4-conv-generator",
        "inputSize": 2 * PIXELS + CLASSES,
        "imageSize": PIXELS,
        "outputs": PIXELS,
        "circuitsPerLabel": votes,
        "layers": layers,
    }



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=50000)
    parser.add_argument("--test", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--t-start", type=float, default=1.0)
    parser.add_argument("--t-min", type=float, default=0.4)
    parser.add_argument("--dense-hidden", default="512,512,512,512")
    parser.add_argument("--arch", choices=("dense", "conv"), default="dense")
    parser.add_argument(
        "--task",
        choices=("complete", "denoise"),
        default="complete",
        help="complete: masked completion. denoise: recover flipped pixels.",
    )
    parser.add_argument("--conv-channels", type=int, default=8)
    parser.add_argument("--conv-layers", type=int, default=4)
    parser.add_argument("--gen-k", type=int, default=4)
    parser.add_argument("--keep-min", type=float, default=0.0)
    parser.add_argument("--keep-max", type=float, default=1.0)
    parser.add_argument(
        "--ink-weight",
        type=float,
        default=1.5,
        help="Extra loss weight on ink pixels (1 disables).",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=200,
        help="Generated digits used to score judge agreement.",
    )
    parser.add_argument(
        "--context-curve",
        action="store_true",
        help="Report completion metrics at several context levels.",
    )
    parser.add_argument("--pin-fraction", type=float, default=0.8)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--reveal-power", type=float, default=1.0)
    parser.add_argument(
        "--remask",
        type=float,
        default=0.0,
        help="Fraction of revealed pixels to un-reveal each round.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated seeds to train a forest, e.g. 1337,1338,1339.",
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--export-hard", action="store_true")
    return parser.parse_args()


def main():
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} device {device}")
    seeds = (
        [int(item) for item in args.seeds.split(",") if item.strip()]
        if args.seeds
        else [args.seed]
    )

    (train_x, train_y), (test_x, test_y) = load_digits(
        args.train, args.test, device
    )
    print(f"train {len(train_x)} test {len(test_x)}")

    hidden = [int(size) for size in args.dense_hidden.split(",") if size.strip()]
    input_size = 2 * PIXELS + CLASSES
    models = []
    parameters = 0
    cells = 0
    for seed in seeds:
        torch.manual_seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        if args.arch == "conv":
            model = GateConvGen(
                args.conv_channels, args.conv_layers, args.gen_k, generator
            ).to(device)
            cells = sum(layer.filters * PIXELS for layer in model.layers)
        else:
            model = GateNet4(
                input_size, hidden, PIXELS * args.gen_k, generator
            ).to(device)
            cells = sum(layer.out_features for layer in model.layers)
        pin_wires(model, generator, args.pin_fraction)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"=== seed {seed} params {parameters:,} cells {cells:,} ===",
            flush=True,
        )
        if args.task == "denoise":
            train_denoise(model, train_x, train_y, args, device, seed)
        else:
            train(model, train_x, train_y, args, device, seed)
        models.append(model)

    # Cumulative curve, the same shape used for the classifier forest.
    curve = []
    if args.task == "denoise":
        denoise = denoise_metrics(models, test_x, test_y, args, device)
        for level, value in denoise.items():
            print(
                f"denoise level {level:4.2f} "
                + " ".join(
                    f"{name} {score * 100:.2f}%" for name, score in value.items()
                ),
                flush=True,
            )
    else:
        for count in range(1, len(models) + 1):
            metrics = completion_metrics(models[:count], test_x, test_y, args, device)
            print(
                f"forest of {count} "
                + " ".join(
                    f"{name} {value * 100:.2f}%" for name, value in metrics.items()
                ),
                flush=True,
            )
            curve.append(metrics)

    hard_metrics = None
    if not args.hard and args.task == "complete":
        args.hard = True
        hard_metrics = completion_metrics(models, test_x, test_y, args, device)
        print(
            "forest hardened "
            + " ".join(
                f"{name} {value * 100:.2f}%" for name, value in hard_metrics.items()
            ),
            flush=True,
        )
        args.hard = False

    labels = torch.arange(CLASSES, device=device).repeat_interleave(
        max(1, args.samples // CLASSES)
    )
    samples = generate(models, args, device, labels)
    for index in range(min(len(labels), args.samples)):
        show(samples[index], int(labels[index]))

    if args.context_curve and args.task == "complete":
        # Does more visible context help? If IoU is flat here, the model is
        # painting per-pixel marginals and ignoring the canvas.
        for keep in (0.0, 0.25, 0.5, 0.75):
            metrics = completion_metrics(
                models, test_x, test_y, args, device, keep=keep
            )
            print(
                f"context {keep:4.2f} "
                + " ".join(
                    f"{name} {value * 100:.2f}%" for name, value in metrics.items()
                ),
                flush=True,
            )

    agreement = None
    judge_curve = []
    if not args.skip_judge:
        judge = train_judge(train_x, train_y, test_x, test_y, device)
        eval_labels = torch.arange(CLASSES, device=device).repeat_interleave(
            max(1, args.eval_samples // CLASSES)
        )
        for count in range(1, len(models) + 1):
            eval_samples = generate(models[:count], args, device, eval_labels)
            score = judge_agreement(judge, eval_samples, eval_labels, device)
            print(
                f"judge agreement forest of {count} "
                f"{score * 100:.2f}% (n={len(eval_labels)})",
                flush=True,
            )
            judge_curve.append(score)
        agreement = judge_curve[-1]

    if args.export_hard:
        for seed, model in zip(seeds, models):
            exported = export_hard(model, args.gen_k)
            exported["seed"] = seed
            print(
                "EXPORT_HARD "
                + json.dumps(exported, separators=(",", ":")),
                flush=True,
            )

    summary = {
        "params": parameters,
        "cells": cells,
        "genK": args.gen_k,
        "epochs": args.epochs,
        "inkWeight": args.ink_weight,
        "rounds": args.rounds,
        "remask": args.remask,
        "seeds": seeds,
        "completionCurve": curve,
        "completionHard": hard_metrics,
        "judgeCurve": judge_curve,
        "judgeAgreement": agreement,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
