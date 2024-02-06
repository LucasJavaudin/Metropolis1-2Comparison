import os
from zipfile import ZipFile

import numpy as np
import polars as pl
import json

# Path to METROPOLIS1 input zipfile (to be downloaded from METROPOLIS1's web interface, with the
# `Export` button).
METROPOLIS1_INPUT = "./metropolis_data/Île-de-France.zip"
# Path to METROPOLIS1 user-level output (to be downloaded from METROPOLIS1's web interface, as the
# traveler-specific results of a run).
METROPOLIS1_OUTPUT_USERS = "./metropolis_data/user_results.tsv"
# Path to METROPOLIS1 path output (only useful when `FORCE_ROUTE` is True, to be downloaded from
# METROPOLIS1's web interface, as the traveler paths of a run).
METROPOLIS1_OUTPUT_PATHS = "./metropolis_data/user_paths.tsv"
# Path to METROPOLIS2 input directory where the simulation should be stored.
METROPOLIS2_RUN_DIR = "./runs/main/"
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
FORCE_DT = False
# If `True`, force the agents' route in METROPOLIS2 to be equal to the routes from METROPOLIS2's
# output.
FORCE_ROUTE = False

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
    "period": PERIOD,
    "learning_model": {
        "type": "Linear",
    },
    "stopping_criteria": [
        {"type": "MaxIteration", "value": 200},
    ],
    "update_ratio": 1.0,
    "random_seed": SEED,  # The random seed here is not relevant when "update_ratio" = 1.
    "network": {
        "road_network": {
            "recording_interval": RECORDING_INTERVAL,
            "spillback": SPILLBACK,
            "max_pending_duration": 30.0,
            "algorithm_type": "Intersect",
        }
    },
    "nb_threads": 0,
    "saving_format": "Parquet",
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

    sources = links["origin"]
    targets = links["destination"].map_dict(target_id_map)

    speed_density_schema = {"type": pl.Utf8, "value": pl.Float64}
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
        pl.col("id").cast(pl.Int64),
        (pl.col("speed") / 3.6).alias("base_speed"),
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
            pl.when(pl.col("function").map_dict(functions_map) == "bottleneck")
            .then(
                pl.struct(
                    pl.lit("Bottleneck").alias("type"),
                    (pl.col("capacity") * VEHICLE_LENGTH / 3600).alias("value"),
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

    edges = list(zip(sources, targets, links.to_dicts()))
    graph = {
        "edges": edges,
    }
    vehicles = [
        {
            "length": VEHICLE_LENGTH,
            "pce": 1.0,
        }
    ]
    road_network = {
        "graph": graph,
        "vehicles": vehicles,
    }

    print("Writing JSON file...")
    with open(os.path.join(METROPOLIS2_RUN_DIR, "road_network.json"), "w") as f:
        f.write(json.dumps(road_network))

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
        routes_dict = routes_df.partition_by("traveler_id", as_dict=True)
    else:
        routes_dict = dict()

    print("Reading traveler types...")
    classes_file = find_file(zipfile, "traveler_types.tsv")
    if classes_file is None:
        raise Exception("Missing file: traveler_types.tsv")
    classes = pl.read_csv(classes_file.read(), separator="\t")
    classes = classes.partition_by("name", as_dict=True)

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
            pl.col("origin").map_dict(db_to_user_id),
            pl.col("destination").map_dict(db_to_user_id),
        )
    else:
        user_results = pl.read_csv(METROPOLIS1_OUTPUT_USERS, separator="\t")
    print("Creating agents...")
    rng = np.random.default_rng(SEED)
    us = iter(rng.uniform(0, 1, size=len(user_results)))
    for user in user_results.iter_rows(named=True):
        t_star_low = PERIOD[0] + TIMESCALE_SHIFT + user["ltstart"]
        t_star_high = PERIOD[0] + TIMESCALE_SHIFT + user["htstar"]
        car_leg = {
            "class": {
                "type": "Road",
                "value": {
                    "origin": user["origin"],
                    "destination": target_id_map[user["destination"]],
                    "vehicle": 0,
                },
            },
            "travel_utility": {
                "type": "Polynomial",
                "value": {
                    "b": -float(user["alphaTI"]) / 3600.0,
                },
            },
            "schedule_utility": {
                "type": "AlphaBetaGamma",
                "value": {
                    "t_star_low": t_star_low,
                    "t_star_high": t_star_high,
                    "beta": float(user["beta"]) / 3600.0,
                    "gamma": float(user["gamma"]) / 3600.0,
                },
            },
        }
        if FORCE_ROUTE and PERIOD[0] + TIMESCALE_SHIFT + user["ta"] <= PERIOD[1]:
            # Path is not available for users who arrived after the end of the period.
            car_leg["class"]["value"]["route"] = routes_dict[user["traveler_id"]][
                "link_id"
            ].to_list()
        if FORCE_DT:
            departure_time_model = {
                "type": "Constant",
                "value": PERIOD[0] + TIMESCALE_SHIFT + user["td"],
            }
        else:
            u = user.get("epsiDeparture", next(us))
            if "departureMu" in user:
                mu = user["departureMu"]
            else:
                mu = classes[user["travelerType"]]["departureMu_mean"].item()
            departure_time_model = {
                "type": "ContinuousChoice",
                "value": {
                    "period": PERIOD,
                    "choice_model": {
                        "type": "Logit",
                        "value": {
                            "u": u,
                            "mu": mu,
                        },
                    },
                },
            }
        car = {
            "type": "Trip",
            "value": {
                "legs": [car_leg],
                "departure_time_model": departure_time_model,
                "utility_model": {
                    "Proportional": -float(user["alphaTI"]) / 3600.0,
                },
            },
        }
        agent = {
            "id": user["traveler_id"],
            "modes": [car],
        }
        agents.append(agent)

    print("Writing JSON file...")
    with open(os.path.join(METROPOLIS2_RUN_DIR, "agents.json"), "w") as f:
        f.write(json.dumps(agents))

    print("===== Parameters =====")
    print("Writing JSON file...")
    with open(os.path.join(METROPOLIS2_RUN_DIR, "parameters.json"), "w") as f:
        f.write(json.dumps(PARAMETERS))


if __name__ == "__main__":
    if not os.path.isdir(METROPOLIS2_RUN_DIR):
        os.makedirs(METROPOLIS2_RUN_DIR)

    metropolis_to_metrosim()
