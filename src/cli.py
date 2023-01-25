import click
import dataset
import model

@click.group()
def cli():
    pass

@cli.group()
def test():
    pass

@cli.command()
def train():
    model.train_and_test()

@cli.command()
@click.argument('image_path')
def predict(image_path):
    model.predict(image_path)

@test.command()
def load_label():
    dataset.load_labels("datasets/t10k-labels-idx1-ubyte")