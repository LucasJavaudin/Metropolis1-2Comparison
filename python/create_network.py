from zipfile import ZipFile

import polars as pl
import geopandas as gpd
from shapely.geometry import LineString

# Path to METROPOLIS1 input zipfile.
METROPOLIS1_INPUT = "./metropolis_data/Île-de-France.zip"
# Path to the Parquet file where the network should be saved.
OUTPUT_FILE = "./metropolis_data/network.parquet"


def find_file(zipfile, filename):
    for file in zipfile.filelist:
        if file.filename.endswith(filename):
            return zipfile.open(file.filename)


def create_network():
    zipfile = ZipFile(METROPOLIS1_INPUT)
    print("Reading zones...")
    zone_file = find_file(zipfile, "zones.tsv")
    if zone_file is None:
        raise Exception("Missing file: zones.tsv")
    zones = pl.read_csv(zone_file.read(), separator="\t")
    print("Reading intersections...")
    intersection_file = find_file(zipfile, "intersections.tsv")
    if intersection_file is None:
        raise Exception("Missing file: intersections.tsv")
    intersections = pl.read_csv(intersection_file.read(), separator="\t")
    # We need to split centroids in a source node and a target node in MetroSim (to have the
    # constraint that they cannot be crossed).
    # The source centroids keep the same id.
    # For the target centroids, we create new ids.
    M = max(zones["id"].max(), intersections["id"].max())
    target_id_map = dict()
    for i in zones["id"]:
        target_id_map[i] = M + 1 + i
    for i in intersections["id"]:
        target_id_map[i] = i
    nodes = zones.vstack(intersections)
    print("Reading congestion functions...")
    function_file = find_file(zipfile, "congestion_functions.tsv")
    if function_file is None:
        raise Exception("Missing file: congestion_functions.tsv")
    functions = pl.read_csv(function_file.read(), separator="\t")
    functions_map = dict()
    for function in functions.iter_rows(named=True):
        if function["name"].lower() in ("free flow", "freeflow"):
            functions_map[function["id"]] = "freeflow"
        elif function["name"].lower() in ("bottleneck function", "bottleneck"):
            functions_map[function["id"]] = "bottleneck"
    print("Reading links...")
    link_file = find_file(zipfile, "links.tsv")
    if link_file is None:
        raise Exception("Missing file: links.tsv")
    links = pl.read_csv(link_file.read(), separator="\t")
    links = links.with_columns(pl.col("function").map_dict(functions_map))
    links = links.with_columns(pl.col("destination").map_dict(target_id_map).alias("destination2"))
    links = links.join(
        nodes.select("id", pl.col("x").alias("x0"), pl.col("y").alias("y0")),
        left_on="origin",
        right_on="id",
        how="left",
    )
    links = links.join(
        nodes.select("id", pl.col("x").alias("x1"), pl.col("y").alias("y1")),
        left_on="destination",
        right_on="id",
        how="left",
    )
    linestrings = [
        LineString([(x0, y0), (x1, y1)])
        for (x0, y0, x1, y1) in zip(links["x0"], links["y0"], links["x1"], links["y1"])
    ]
    gdf = gpd.GeoDataFrame(links.to_pandas(), geometry=linestrings, crs="epsg:27561")
    return gdf


if __name__ == "__main__":
    gdf = create_network()
    gdf.to_file(OUTPUT_FILE, driver="Parquet")
