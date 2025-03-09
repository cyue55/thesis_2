import tempfile

import pytest
import torch

from mbchl.training.ema import EMARegistry


def _init_model():
    return torch.nn.Linear(4, 4, bias=False)


@pytest.mark.parametrize(
    "ema_name, ema_kw",
    [
        ["classic", {"beta": 0.99}],
        ["karras", {"sigma_rels": [0.05, 0.1]}],
    ],
)
def test_state_dict(ema_name, ema_kw):
    first_val = 1.0
    second_val = 0.0
    third_val = 0.5

    ema_cls = EMARegistry.get(ema_name)

    # init the model and simulate an update
    model = _init_model()
    with torch.no_grad():
        model.weight.fill_(first_val)
    ema = ema_cls(model, **ema_kw)
    with torch.no_grad():
        model.weight.fill_(second_val)
    ema.update()

    # simulate saving and loading the state dict
    with tempfile.TemporaryFile() as f:
        torch.save(ema.state_dict(), f)
        f.seek(0)
        state_dict = torch.load(f, weights_only=True)

    # create a new model and load the state dict
    model = _init_model()
    ema = ema_cls(model, **ema_kw)
    ema.load_state_dict(state_dict)

    # check that the loaded state dict is correct
    if ema_name != "karras":
        beta = ema_kw["beta"]
        ema_param = ema.ema_params[0]
        target = beta * first_val + (1 - beta) * second_val
        assert torch.allclose(ema_param, torch.tensor(target))
    else:
        target = second_val
        for sigma_rel in ema_kw["sigma_rels"]:
            ema_param = ema.ema_params[sigma_rel][0]
            assert torch.allclose(ema_param, torch.tensor(second_val))

    # try another update
    with torch.no_grad():
        model.weight.fill_(third_val)
    ema.update()
    if ema_name != "karras":
        target = beta * target + (1 - beta) * third_val
        assert torch.allclose(ema_param, torch.tensor(target))
    else:
        # TODO: check the actual target value for Karras EMA
        for sigma_rel in ema_kw["sigma_rels"]:
            ema_param = ema.ema_params[sigma_rel][0]
            assert not torch.allclose(ema_param, torch.tensor(second_val))


def test_post_hoc_ema():
    model = _init_model()
    ema_cls = EMARegistry.get("karras")
    ema = ema_cls(model, sigma_rels=[0.05, 0.1])
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as d:
        for i in range(10):
            x = torch.randn(16, 4)
            y = model(x)
            loss = y.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            ema.update()
            torch.save(ema.state_dict(), f"{d}/{i}.ckpt")

        sigma_rel_r = 0.2
        ema.post_hoc_ema(d, sigma_rel_r)
