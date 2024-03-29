#!/usr/bin/env python
import click
import numpy as np
from ai import constants, model_architectures, model, dataset as _dataset


@click.group()
def cli():
    pass

@cli.command()
@click.argument('ephocs')
def train(ephocs):
    model.train(int(ephocs))

@cli.command()
@click.argument('image_path')
def predict(image_path):
    _model = model.load_model()
    model.predict(_model, image_path)
    
    
@cli.group()
def test():
    pass

@test.command()
def dataset():
    print(constants.TRAINING_IMAGES_PATH)
    data = _dataset.load_images(constants.TRAINING_IMAGES_PATH)
    
    
    _dataset.image_to_file(data[5], "image1.png")
    
    
cli()
