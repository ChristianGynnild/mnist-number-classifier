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

@test.command()
def load_label():
    dataset.load_labels("datasets/t10k-labels-idx1-ubyte")