import click

@click.group()
def main():
    pass

from .segment import segment
main.add_command(segment)
from .depth import depth
main.add_command(depth)

if __name__ == "__main__":
    main()
