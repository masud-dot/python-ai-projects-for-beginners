# Chapter 6 tests. They use random tensors, not FashionMNIST,
# so they run offline and fast.
import runpy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from model import NeuralNetwork  # noqa: E402
from train import evaluate, train_one_epoch  # noqa: E402


def run(name, capsys):
    runpy.run_path(str(HERE / name), run_name="__main__")
    return capsys.readouterr().out


def test_tensors_script(capsys):
    out = run("tensors.py", capsys)
    assert "a: torch.Size([2, 3]) torch.float32" in out
    assert "shares memory: [99, 2, 3]" in out


def test_autograd_accumulates(capsys):
    out = run("autograd_demo.py", capsys)
    lines = {ln.split(":")[0]: ln for ln in out.splitlines()}
    first = float(out.split("w.grad = ")[1].split()[0])
    second = float(lines["after backward #2"].split(": ")[1])
    third = float(lines["after backward #3"].split(": ")[1])
    again = float(lines["after zero_grad"].split(": ")[1])
    assert second == pytest.approx(2 * first, abs=1e-5)
    assert third == pytest.approx(3 * first, abs=1e-5)
    assert again == pytest.approx(first, abs=1e-5)
    assert "tracked inside no_grad: False" in out


def test_shapes_script(capsys):
    out = run("shapes.py", capsys)
    assert "Flatten    (64, 784)" in out
    assert "parameters: 669,706" in out


def test_model_shapes():
    out = NeuralNetwork()(torch.rand(4, 1, 28, 28))
    assert out.shape == (4, 10)


def fake_loader(n=256):
    torch.manual_seed(0)
    X = torch.rand(n, 1, 28, 28)
    y = (X.mean(dim=(1, 2, 3)) > 0.5).long()
    return [(X[i:i + 64], y[i:i + 64])
            for i in range(0, n, 64)]


def test_training_lowers_loss():
    torch.manual_seed(0)
    data = fake_loader()
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before, _ = evaluate(data, model, loss_fn, "cpu")
    for _ in range(5):
        train_one_epoch(data, model, loss_fn, opt, "cpu")
    after, acc = evaluate(data, model, loss_fn, "cpu")
    assert after < before
    assert 0.0 <= acc <= 1.0


def test_state_dict_round_trip(tmp_path):
    model = NeuralNetwork()
    path = tmp_path / "m.pth"
    torch.save(model.state_dict(), path)
    other = NeuralNetwork()
    other.load_state_dict(torch.load(path, weights_only=True))
    x = torch.rand(3, 1, 28, 28)
    model.eval()
    other.eval()
    assert torch.equal(model(x), other(x))
