import os
import time

import pytest
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from infirun.util.schematic import from_schema
from ..trainer import *


@pytest.fixture
def tmp_dir():
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'tmp_out'))
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return path


class ConvBase(nn.Module):
    def __init__(self, input_channel, output_channel, n_layers):
        super().__init__()
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(input_channel, output_channel, 3, padding=1))
        for i in range(n_layers - 1):
            layers.append(nn.Conv2d(output_channel, output_channel, 3, padding=1))
        self.layers = layers

    def forward(self, inp):
        x = inp
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x


class ClassificationTop(nn.Module):
    def __init__(self, conv_base, img_size, output_channel):
        super().__init__()
        self.conv_base = conv_base
        self.proj = nn.Linear(img_size[0] * img_size[1] * output_channel, 1)

    def forward(self, inp, labels):
        nbatch, _, _, _ = inp.size()
        features = self.conv_base(inp)
        output = self.proj(features.view(nbatch, -1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(output, labels)
        accuracy = ((output > 0) == labels.byte()).float().mean()
        return loss, accuracy


class SampleGen:
    def __init__(self, batch_size, img_size, input_channel):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_channel = input_channel
        self.count = 0

    def get(self):
        if self.count == 200:
            raise StopIteration
        self.count += 1
        inp = torch.zeros(self.batch_size, self.input_channel, *self.img_size).normal_()
        label = torch.randint(2, size=(self.batch_size,), dtype=torch.float)
        return inp, label


class SampleTrainer(Trainer):
    def setup_optimizer(self, model, settings):
        return torch.optim.Adam(model.parameters())

    def train_step(self, model, next_value, settings):
        loss, acc = model(*next_value)
        return loss, {'mean_loss': loss,
                      'stupid/mean_acc': acc}

    def validate(self, model, next_values, settings):
        loss, acc = model(*next_values[0])
        return {'validate/loss': loss,
                'validate/acc': acc}


class SampleTrainerWithScheduler(SampleTrainer):
    def setup_scheduler(self, optimizer, settings):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)


def test_model():
    batch_size = 4
    img_size = [28, 28]
    input_channel = 3
    output_channel = 16
    n_layers = 2

    gen = SampleGen(batch_size, img_size, input_channel)
    assert gen.get()[0].shape == (batch_size, input_channel, *img_size)
    assert gen.get()[1].shape == (batch_size,)

    conv_base = ConvBase(input_channel, output_channel, n_layers)
    top = ClassificationTop(conv_base, img_size, output_channel)

    loss, acc = top(*gen.get())
    assert 0 <= acc <= 1


def test_schema():
    img_size = [28, 28]
    input_channel = 3
    output_channel = 16
    n_layers = 2
    conv_base = ModelComponent(ConvBase)(input_channel, output_channel, n_layers)
    top = ModelComponent(ClassificationTop)(conv_base, img_size, output_channel)
    schema = top.get_schema()
    # print(json.dumps(schema, indent=2))
    model = from_schema(schema).build()

    batch_size = 4
    gen = SampleGen(batch_size, img_size, input_channel)
    loss, acc = model(*gen.get())
    assert 0 <= acc <= 1


@pytest.mark.parametrize('use_scheduler', [True, False])
def test_trainer(tmp_dir, use_scheduler):
    batch_size = 4
    img_size = [28, 28]
    input_channel = 3
    output_channel = 16
    n_layers = 2

    gen = SampleGen(batch_size, img_size, input_channel)
    val_gen = SampleGen(batch_size, img_size, input_channel)

    conv_base = ModelComponent(ConvBase)(input_channel, output_channel, n_layers)
    top = ModelComponent(ClassificationTop)(conv_base, img_size, output_channel)

    model_dir = os.path.join(tmp_dir, f'model_dir_{1 if use_scheduler else 0}')
    log_dir = os.path.join(tmp_dir, f'log_dir_{1 if use_scheduler else 0}')

    try:
        shutil.rmtree(model_dir)
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(log_dir)
    except FileNotFoundError:
        pass

    if use_scheduler:
        Train = SampleTrainerWithScheduler
    else:
        Train = SampleTrainer

    trainer = Train(model=top,
                    train_source=gen,
                    validate_source=val_gen,
                    log_per_n_steps=10,
                    save_per_n_steps=100,
                    save_per_minutes=0.00001,
                    keep_last_n_saves=2,
                    validate_per_n_steps=100,
                    validate_steps=10,
                    validate_per_n_epochs=None,
                    schedule_per_n_steps=100,
                    schedule_per_n_epochs=None,
                    model_dir=os.path.join(model_dir),
                    log_dir=os.path.join(log_dir))
    trainer.start()
    save_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    assert len(save_files) == 3
    trainer.restore_model_and_state(model_dir)
