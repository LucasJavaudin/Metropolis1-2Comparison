import os
from zipfile import ZipFile

import numpy as np
import polars as pl
import json

# Path to METROPOLIS1 input zipfile (to be downloaded from METROPOLIS1's web interface, with the
# `Export` button).
METROPOLIS1_INPUT = "./metropolis_input/Île-de-France.zip"
# Path to METROPOLIS1 user-level output (to be downloaded from METROPOLIS1's web interface, as the
# traveler-specific results of a run).
METROPOLIS1_OUTPUT_USERS = "./metropolis_output/user_results.tsv"
# Path to METROPOLIS1 path output (only useful when `FORCE_ROUTE` is True, to be downloaded from
# METROPOLIS1's web interface, as the traveler paths of a run).
METROPOLIS1_OUTPUT_PATHS = "./metropolis_output/user_paths.tsv"
# Path to METROPOLIS2 input directory where the simulation should be stored.
METROPOLIS2_RUN_DIR = "./runs/fixed_routes/"
# If `True`, use the raw METROPOLIS1's user-level output file (the modified METROPOLIS1 version is
# required for that, not compatible with the web interface).
RAW_OUTPUT = False
# If `True`, reverse the `mu` values given in the user-level output (only relevant if `RAW_OUTPUT`
# is True).
REVERSE_MUS = False
# If `True`, enable spillback for the METROPOLIS2's simulation.
SPILLBACK = True
# Vehicle length in meters (only relevant if `SPILLBACK` is True).
VEHICLE_LENGTH = 8.0
# If `True`, bottlenecks are represented using METROPOLIS2 dynamic queues version; otherwise, use
# METROPOLIS1 bottleneck's speed-density function (Note that setting this to True can lead to very
# different results between the two simulators).
TRUE_BOTTLENECK = False
# If `True`, the bottleneck's capacities are adjusted so that no congestion appears whenever two
# vehicles follow each other with more than the link's free-flow travel time delay (only relevant if
# `TRUE_BOTTLENECK` is True).
FIX_CAPACITY = False
# If `True`, add time penalties to each edges to reflect the rounding done in METROPOLIS1.
TT_PENALTIES = False
# If `True`, try to adjust base speeds so that the free-flow travel times are rounded in the same
# way in both METROPOLIS1 and METROPOLIS2 (best to use with `TT_PENALTIES`).
FIX_ROUNDING = False
# If `True`, force the agents' departure time in METROPOLIS2 to be equal to the departure times from
# METROPOLIS1's output.
FORCE_DT = True
# If `True`, force the agents' route in METROPOLIS2 to be equal to the routes from METROPOLIS2's
# output.
FORCE_ROUTE = True

# Time in seconds between two recording points (for the travel-time functions).
RECORDING_INTERVAL = 20.0 * 60.0
# Random seed to generate the `u` random values.
SEED = 13081996
# Simulated period, in seconds from midnight.
PERIOD = [4.0 * 3600.0, 13.0 * 3600.0]
# Timescale shift in METROPOLIS1 in seconds (when METROPOLIS1 shows the warning "Time scale shift:
# xx hours").
TIMESCALE_SHIFT = 0.0
# Parameters for METROPOLIS2's simulation.
PARAMETERS = {
    "input_files": {
        "agents": "agents.parquet",
        "alternatives": "alts.parquet",
        "trips": "trips.parquet",
        "edges": "edges.parquet",
        "vehicle_types": "vehicles.parquet",
    },
    "output_directory": "output",
    "period": PERIOD,
    "learning_model": {
        "type": "Linear",
    },
    "max_iterations": 1,
    "road_network": {
        "recording_interval": RECORDING_INTERVAL,
        "spillback": SPILLBACK,
        "max_pending_duration": 30.0,
        "algorithm_type": "Intersect",
    },
    "saving_format": "Parquet",
    "nb_threads": 0,
}


def find_file(zipfile, filename):
    for file in zipfile.filelist:
        if file.filename.endswith(filename):
            return zipfile.open(file.filename)


def metropolis_to_metrosim():
    zipfile = ZipFile(METROPOLIS1_INPUT)
    print("===== Road network =====")
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

    speed_density_schema = {"type": pl.Utf8, "capacity": pl.Float64}
    if FIX_ROUNDING:
        links = links.with_columns(
            (
                pl.col("length").cast(pl.Float32)
                / pl.col("speed").cast(pl.Float32)
                * pl.lit(3600.0)
            ).alias("fftt")
        )
        links = links.with_columns(
            pl.when(pl.col("fftt") % 1.0 == 0.0)
            .then(
                (pl.lit(1.0000001).cast(pl.Float32) * pl.col("speed").cast(pl.Float32)).cast(
                    pl.Float64
                )
            )
            .otherwise(pl.col("speed"))
            .alias("speed")
        )
    links = links.select(
        pl.col("id").cast(pl.Int64).alias("edge_id"),
        pl.col("origin").alias("source"),
        pl.col("destination").replace(target_id_map).alias("target"),
        (pl.col("speed") / 3.6),
        pl.col("length") * 1000,
        pl.col("lanes").cast(pl.Int64),
        pl.when(TRUE_BOTTLENECK)
        .then(
            pl.struct(
                pl.lit("FreeFlow").alias("type"),
                schema=speed_density_schema,
            )
        )
        .otherwise(
            pl.when(pl.col("function").replace(functions_map) == "bottleneck")
            .then(
                pl.struct(
                    pl.lit("Bottleneck").alias("type"),
                    (pl.col("capacity") * VEHICLE_LENGTH / 3600).alias("capacity"),
                    schema=speed_density_schema,
                )
            )
            .otherwise(
                pl.struct(
                    pl.lit("FreeFlow").alias("type"),
                    schema=speed_density_schema,
                )
            )
        )
        .alias("speed_density"),
        pl.when(TRUE_BOTTLENECK).then(pl.col("capacity") / 3600).alias("bottleneck_flow"),
    )
    if not TRUE_BOTTLENECK:
        links = links.drop("bottleneck_flow")
    if TRUE_BOTTLENECK and FIX_CAPACITY:
        links = links.with_columns(
            pl.max_horizontal(
                pl.col("bottleneck_flow"), pl.col("base_speed") / pl.col("length")
            ).alias("bottleneck_flow")
        )
    if TT_PENALTIES:
        links = links.with_columns(
            (-pl.col("length") / pl.col("base_speed") % 1).alias("constant_travel_time")
        )

    vehicles = pl.DataFrame(
        {
            "vehicle_id": [1],
            "headway": [VEHICLE_LENGTH],
            "pce": [1.0],
        }
    )

    print("Writing road network...")
    links.write_parquet(os.path.join(METROPOLIS2_RUN_DIR, "edges.parquet"))
    vehicles.write_parquet(os.path.join(METROPOLIS2_RUN_DIR, "vehicles.parquet"))

    print("===== Agents =====")

    #  print("Reading public transit...")
    #  pt_file = find_file(zipfile, "public_transit.tsv")
    #  if pt_file is None:
    #  raise Exception("Missing file: public_transit.tsv")
    #  pt = pl.read_csv(pt_file.read(), separator="\t")

    if FORCE_ROUTE:
        print("Reading routes...")
        routes_df = pl.read_csv(METROPOLIS1_OUTPUT_PATHS, separator="\t")
        routes_df = routes_df.with_columns(pl.col("in_time").str.to_datetime("%H:%M:%S"))
        routes_df = routes_df.sort("traveler_id", "in_time")
        routes_df = routes_df.group_by("traveler_id").agg(pl.col("link_id").alias("route"))
    else:
        routes_df = pl.DataFrame()

    print("Reading traveler types...")
    classes_file = find_file(zipfile, "traveler_types.tsv")
    if classes_file is None:
        raise Exception("Missing file: traveler_types.tsv")
    classes = pl.read_csv(classes_file.read(), separator="\t")

    print("Reading agents...")
    agents = list()
    if RAW_OUTPUT:
        user_results = pl.read_csv(
            METROPOLIS1_OUTPUT_USERS,
            separator="\t",
            has_header=False,
            new_columns=[
                "origin",
                "destination",
                "travelerType",
                "driveCar",
                "alphaTI",
                "beta",
                "gamma",
                "alphaPT",
                "ptPenalty",
                "departureMu",
                "routeMu",
                "modeMu",
                "epsiDeparture",
                "epsiMode",
                "td",
                "ta",
                "xpta",
                "ltstart",
                "htstar",
                "fee",
                "surplus",
            ],
        )
        user_results = user_results.with_columns(
            pl.int_range(0, len(user_results)).alias("traveler_id")
        )
        if REVERSE_MUS:
            user_results = user_results.with_columns(-pl.col("departureMu") / 3600.0)
        # We need to convert db node ids to user ids.
        db_to_user_id = dict(
            ((db_id, user_id) for db_id, user_id in zip(zones["db_id"], zones["id"]))
        )
        db_to_user_id.update(
            (
                (db_id, user_id)
                for db_id, user_id in zip(intersections["db_id"], intersections["id"])
            )
        )
        user_results = user_results.with_columns(
            pl.col("origin").replace(db_to_user_id),
            pl.col("destination").replace(db_to_user_id),
        )
    else:
        user_results = pl.read_csv(METROPOLIS1_OUTPUT_USERS, separator="\t")
        user_results = user_results.join(
            classes.select("name", pl.col("departureMu_mean").alias("departureMu")),
            left_on="travelerType",
            right_on="name",
        )
        if FORCE_ROUTE:
            user_results = user_results.join(routes_df, on="traveler_id")
    print("Creating agents...")
    rng = np.random.default_rng(SEED)
    agents = user_results.select(
        pl.col("traveler_id").alias("agent_id"),
    )
    if FORCE_DT:
        dt_choice_expr = pl.struct(
            pl.lit("Constant").alias("type"),
            (pl.col("td") + PERIOD[0] + TIMESCALE_SHIFT).alias("departure_time"),
        ).alias("dt_choice")
    else:
        if RAW_OUTPUT:
            us = user_results["epsiDeparture"]
        else:
            us = pl.Series(rng.uniform(0, 1, size=len(user_results)))
        dt_choice_expr = pl.struct(
            pl.lit("Continuous").alias("type"),
            pl.struct(
                pl.lit("Logit").alias("type"),
                pl.Series(us).alias("u"),
                pl.col("departureMu").alias("mu"),
            ).alias("model"),
        ).alias("dt_choice")
    if FORCE_ROUTE:
        class_expr = pl.struct(
            pl.lit("Road").alias("type"),
            pl.col("origin"),
            pl.col("destination").replace(target_id_map),
            pl.lit(1).alias("vehicle"),
            pl.col("route"),
        ).alias("class")
    else:
        class_expr = pl.struct(
            pl.lit("Road").alias("type"),
            pl.col("origin"),
            pl.col("destination").replace(target_id_map),
            pl.lit(1).alias("vehicle"),
        ).alias("class")
    alts = user_results.select(
        pl.col("traveler_id").alias("agent_id"),
        pl.lit(0).alias("alt_id"),
        dt_choice_expr,
    )
    trips = user_results.select(
        pl.col("traveler_id").alias("agent_id"),
        pl.lit(0).alias("alt_id"),
        pl.col("traveler_id").alias("trip_id"),
        class_expr,
        pl.struct(
            (-pl.col("alphaTI") / 3600.0).alias("one"),
        ).alias("travel_utility"),
        pl.struct(
            pl.lit("AlphaBetaGamma").alias("type"),
            ((pl.col("ltstart") + pl.col("htstar")) / 2.0 + PERIOD[0] + TIMESCALE_SHIFT).alias(
                "tstar"
            ),
            (pl.col("htstar") - pl.col("ltstart")).alias("delta"),
            pl.col("beta") / 3600.0,
            pl.col("gamma") / 3600.0,
        ).alias("schedule_utility"),
    )
    print("Writing population...")
    agents.write_parquet(os.path.join(METROPOLIS2_RUN_DIR, "agents.parquet"))
    alts.write_parquet(os.path.join(METROPOLIS2_RUN_DIR, "alts.parquet"))
    trips.write_parquet(os.path.join(METROPOLIS2_RUN_DIR, "trips.parquet"))

    print("===== Parameters =====")
    print("Writing JSON file...")
    with open(os.path.join(METROPOLIS2_RUN_DIR, "parameters.json"), "w") as f:
        f.write(json.dumps(PARAMETERS))


if __name__ == "__main__":
    if not os.path.isdir(METROPOLIS2_RUN_DIR):
        os.makedirs(METROPOLIS2_RUN_DIR)

    metropolis_to_metrosim()
