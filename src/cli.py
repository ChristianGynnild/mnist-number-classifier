import click
import dataset
import model

@click.group()
def cli():
    pass

@cli.command()
def train():
    model.train()

@cli.command()
@click.argument('image_path')
def predict(image_path):
    model.predict(image_path)