import click
import dataset

@click.group()
def cli():
    pass

@cli.group()
def test():
    pass

@test.command()
def load_label():
    dataset.load_labels("datasets/t10k-labels-idx1-ubyte")