from pathlib import Path

import click
import networkx as nx

from skymap import SkyMap
from skymap.constants import banner


@click.group()
def cli():
    print(banner)


@cli.command()  # @cli, not @click!
@click.option(
    "-g", "--graph", "graph", type=click.Path(exists=True, dir_okay=False), required=True
)
@click.option("-o", "--output", "output", type=click.Path(exists=False), required=True)
@click.option("-n", "--num_nodes", "num_nodes", type=int, required=False, default=None)
def mimic_graph(graph, output, num_nodes=None):
    graph_path = Path(graph)
    output_path = Path(output)
    if output_path.is_file():
        output_path.parent.mkdir(exist_ok=True)
    else:
        output_path = output_path / "skymap_graph.gml"

    match graph_path.suffix:
        case ".gml":
            graph = nx.read_gml(graph_path)
        case _:
            raise AttributeError("Only .gml supported")
    skymap = SkyMap()
    generated_graph = skymap.mimic_graph(graph, num_nodes)
    nx.write_gml(generated_graph, output_path)


if __name__ == "__main__":
    cli()
