import click
import dataset as _dataset
import model
import constants
import numpy as np

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
    model.predict(image_path)
    
    
@cli.group()
def test():
    pass

@test.command()
def dataset():
    print(constants.TRAINING_IMAGES_PATH)
    data = _dataset.load_images(constants.TRAINING_IMAGES_PATH)
    
    
    _dataset.image_to_file(data[5], "image1.png")
    
