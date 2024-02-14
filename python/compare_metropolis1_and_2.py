import os
from zipfile import ZipFile

from math import sqrt
import numpy as np
import polars as pl
import matplotlib.ticker as mticker

import mpl_utils

# Path to METROPOLIS1 input zipfile.
METROPOLIS1_INPUT = "./metropolis_data/Île-de-France.zip"
# Path to METROPOLIS1 output directory.
METROPOLIS1_OUTPUT = "./metropolis1/main/"
# Path to METROPOLIS2 output directory.
METROPOLIS2_OUTPUT = "./runs/main/output/"
# Period of the simulation.
PERIOD = [4.0 * 3600.0, 13.0 * 3600.0]
# Length of the recording periods.
RECORDING_INTERVAL = 20.0 * 60.0
# Number of iterations that were run.
NB_ITERATIONS = 200
# Path to the directory where the graphs should be saved.
GRAPH_DIR = "./graph"


def seconds_to_time_str(t):
    if t < 0.0:
        return "-{}".format(seconds_to_time_str(-t))
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = int(round(t % 60))
    return f"{h:02}:{m:02}:{s:02}"


def find_file(zipfile, filename):
    for file in zipfile.filelist:
        if file.filename.endswith(filename):
            return zipfile.open(file.filename)
    else:
        raise Exception("Find not found: {}".format(filename))


def read_raw_output(iteration):
    filename = os.path.join(METROPOLIS1_OUTPUT, "metrosim_users_{}.txt".format(iteration))
    df = pl.read_csv(
        filename,
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
    df = df.with_columns(-pl.col("departureMu") / 3600.0)
    df = df.with_columns(pl.arange(1, len(df) + 1).alias("traveler_id"))
    df = df.with_columns(
        pl.col("td") + PERIOD[0],
        pl.col("ta") + PERIOD[0],
        pl.col("xpta") + PERIOD[0],
        pl.col("ltstart") + PERIOD[0],
        pl.col("htstar") + PERIOD[0],
    )
    return df


def read_raw_paths(iteration, links):
    filename = os.path.join(METROPOLIS1_OUTPUT, "metrosim_events_{}.txt".format(iteration))
    df = pl.read_csv(
        filename,
        separator="\t",
        has_header=False,
        new_columns=[
            "traveler_id",
            "entry_time",
            "link_id",
        ],
    )
    # We need to convert db link ids to user ids.
    db_to_user_id = dict(((db_id, user_id) for db_id, user_id in zip(links["db_id"], links["id"])))
    df = df.with_columns(pl.col("link_id").map_dict(db_to_user_id))
    df = df.with_columns(pl.col("entry_time").str.to_time("%H:%M:%S"))
    df = df.with_columns(
        (
            pl.col("entry_time").dt.hour() * 3600
            + pl.col("entry_time").dt.minute() * 60
            + pl.col("entry_time").dt.second()
        ).alias("entry_time_seconds")
    )
    return df


def stats():
    mp_users = read_raw_output(NB_ITERATIONS - 1)
    mp_users = mp_users.with_columns((pl.col("ta") - pl.col("td")).alias("tt"))
    mp_users = mp_users.with_columns(
        (
            pl.col("beta") * pl.max_horizontal(0.0, pl.col("ltstart") - pl.col("ta")) / 3600.0
            + pl.col("gamma") * pl.max_horizontal(0.0, pl.col("ta") - pl.col("htstar")) / 3600.0
        ).alias("sc_cost")
    )
    mp_users = mp_users.with_columns(
        (-pl.col("alphaTI") * pl.col("tt") / 3600.0 - pl.col("sc_cost")).alias("utility")
    )
    zipfile = ZipFile(METROPOLIS1_INPUT)
    links = pl.read_csv(find_file(zipfile, "links.tsv").read(), separator="\t")
    links = links.with_columns((3600.0 * pl.col("length") / pl.col("speed")).alias("fftt"))
    paths = read_raw_paths(NB_ITERATIONS - 1, links)
    paths = paths.with_columns(
        pl.arange(0, pl.count()).over(["traveler_id", "link_id"]).alias("order_id")
    )
    paths = paths.join(
        links.select("id", "length", "fftt"), left_on="link_id", right_on="id", how="left"
    )
    mp_users = mp_users.join(
        paths.group_by("traveler_id").agg(pl.col("fftt").sum(), pl.col("length").sum(), pl.count()),
        on="traveler_id",
        how="left",
    )
    # Results of the before-last iteration.
    blast_df = read_raw_output(NB_ITERATIONS - 2)
    td_diff = blast_df["td"] - mp_users["td"]
    td_diff_rmse = sqrt((td_diff**2).mean())
    exp_bias = mp_users["ta"] - mp_users["xpta"]
    expect = sqrt((exp_bias**2).mean())
    # Route RMSE.
    prev_paths = read_raw_paths(NB_ITERATIONS - 2, links)
    prev_paths = prev_paths.with_columns(
        pl.arange(0, pl.count()).over(["traveler_id", "link_id"]).alias("order_id")
    )
    # We also join by `order_id` to account for loops in the routes.
    common_paths = paths.select("traveler_id", "link_id", "order_id", "length").join(
        prev_paths.select("traveler_id", "link_id", "order_id"),
        on=["traveler_id", "link_id", "order_id"],
        how="inner",
    )
    common_length = common_paths.group_by("traveler_id").agg(pl.col("length").sum())
    total_length = paths.group_by("traveler_id").agg(pl.col("length").sum())
    length_df = total_length.join(
        common_length, on="traveler_id", suffix="_common", how="left"
    ).with_columns(pl.col("length_common").fill_null(0.0))
    route_rmse = sqrt(((1.0 - length_df["length_common"] / length_df["length"]) ** 2).mean())
    # Convert mean surplus in METROPOLIS2 format.
    period_length = PERIOD[1] - PERIOD[0]
    mean_surplus = mp_users.select(
        pl.col("surplus")
        + pl.col("departureMu")
        * (np.euler_gamma + np.log(period_length) - np.log(period_length / 3600))
    ).mean()
    print("===== METROPOLIS =====")
    print("RMSE departure time: {:.4}s".format(td_diff_rmse))
    print("RMSE route: {:.2%}".format(route_rmse))
    print("RMSE T: {:.4}s".format(ttime_rmse(NB_ITERATIONS - 1, links)))
    print("RMSE expect: {}".format(seconds_to_time_str(expect)))
    print("Average surplus: {:.4f}".format(mean_surplus))
    print("Average utility: {:.4f}".format(mp_users["utility"].mean()))
    print("Average departure time: {}".format(seconds_to_time_str(mp_users["td"].mean())))
    print("Average travel time: {}".format(seconds_to_time_str(mp_users["tt"].mean())))
    print("Average free-flow travel time: {}".format(seconds_to_time_str(mp_users["fftt"].mean())))
    print("Average route length: {:.4f}km".format(mp_users["length"].mean()))
    print("Average nb edges taken: {:.2f}".format(mp_users["count"].mean()))
    it_res = pl.read_parquet(os.path.join(METROPOLIS2_OUTPUT, "iteration_results.parquet"))
    ms_legs = pl.read_parquet(os.path.join(METROPOLIS2_OUTPUT, "leg_results.parquet"))
    ms_legs = ms_legs.with_columns(
        (pl.col("length_diff") / pl.col("length")).alias("share_length_diff")
    )
    ms_legs = ms_legs.with_columns(
        (
            (pl.col("arrival_time") - pl.col("exp_arrival_time")).abs()
            / (pl.col("arrival_time") - pl.col("departure_time"))
        ).alias("expect")
    )
    print("===== METROPOLIS2 =====")
    print("RMSE departure time: {:.4}s".format(it_res["trip_dep_time_rmse"][-1]))
    print("RMSE route: {:.2%}".format(sqrt(ms_legs["share_length_diff"].pow(2).mean())))
    print("RMSE T: {:.4}s".format(it_res["exp_road_network_weights_rmse"][-1]))
    print(
        "RMSE expect: {}".format(
            seconds_to_time_str(it_res["road_leg_exp_travel_time_diff_rmse"][-1])
        )
    )
    print("Average surplus: {:.4f}".format(it_res["expected_utility_mean"]))
    print("Average utility: {:.4f}".format(it_res["trip_utility_mean"][-1]))
    print(
        "Average departure time: {}".format(
            seconds_to_time_str(it_res["trip_departure_time_mean"][-1])
        )
    )
    print(
        "Average travel time: {}".format(seconds_to_time_str(it_res["trip_travel_time_mean"][-1]))
    )
    print(
        "Average free-flow travel time: {}".format(
            seconds_to_time_str(it_res["road_leg_route_free_flow_travel_time_mean"][-1])
        )
    )
    print("Average route length: {:.4f}km".format(it_res["road_leg_length_mean"][-1] / 1000.0))
    print("Average nb edges taken: {:.2f}".format(it_res["road_leg_edge_count_mean"][-1]))


def compare_departure_time_distributions(filename=None):
    mp_users = read_raw_output(NB_ITERATIONS - 1)
    ms_users = pl.read_parquet(os.path.join(METROPOLIS2_OUTPUT, "agent_results.parquet"))
    bins = np.arange(PERIOD[0] / 3600.0, PERIOD[1] / 3600.0 + 1e-6, 15.0 / 60.0)
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.hist(
        mp_users["td"] / 3600.0,
        bins=bins,
        histtype="step",
        alpha=0.7,
        color=mpl_utils.CMP(0),
        label="METROPOLIS",
    )
    ax.hist(
        (ms_users["departure_time"] / 3600.0),
        bins=bins,
        histtype="step",
        alpha=0.7,
        color=mpl_utils.CMP(1),
        label=r"METROPOLIS2",
    )
    ax.set_xlabel("Departure time")
    ax.set_xlim(PERIOD[0] / 3600.0, PERIOD[1] / 3600.0)
    ax.xaxis.set_major_formatter(lambda x, _: seconds_to_time_str(x * 3600.0))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter("{x:n}")
    ax.legend()
    fig.tight_layout()
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)


def compare_travel_time_distributions(filename=None, tt_max=1.5 * 3600.0):
    mp_users = read_raw_output(NB_ITERATIONS - 1)
    mp_users = mp_users.with_columns((pl.col("ta") - pl.col("td")).alias("tt"))
    ms_users = pl.read_parquet(os.path.join(METROPOLIS2_OUTPUT, "agent_results.parquet"))
    bins = np.arange(0.0, tt_max / 60.0 + 1e-4, 2.0)
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.hist(
        mp_users["tt"] / 60.0,
        bins=bins,
        histtype="step",
        alpha=0.7,
        color=mpl_utils.CMP(0),
        label="METROPOLIS",
    )
    ax.hist(
        ms_users["total_travel_time"] / 60.0,
        bins=bins,
        histtype="step",
        alpha=0.7,
        color=mpl_utils.CMP(1),
        label=r"METROPOLIS2",
    )
    ax.set_xlabel("Travel time")
    ax.set_xlim(0.0, tt_max / 60.0)
    ax.xaxis.set_major_formatter(lambda x, _: seconds_to_time_str(x * 60.0))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter("{x:n}")
    ax.legend()
    fig.tight_layout()
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)


def compare_travel_time_pairwise_scatter(filename=None, tt_max=1.5 * 3600.0):
    mp_users = read_raw_output(NB_ITERATIONS - 1)
    mp_users = mp_users.with_columns((pl.col("ta") - pl.col("td")).alias("tt"))
    mp_users = mp_users.sort("traveler_id")
    ms_users = pl.read_parquet(os.path.join(METROPOLIS2_OUTPUT, "agent_results.parquet"))
    ms_users = ms_users.sort("id")
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.scatter(
        mp_users["tt"] / 60.0,
        ms_users["total_travel_time"] / 60.0,
        alpha=0.01,
        color=mpl_utils.CMP(0),
    )
    ax.set_xlabel("Travel time (METROPOLIS)")
    ax.set_xlim(0.0, tt_max / 60.0)
    ax.xaxis.set_major_formatter(lambda x, _: seconds_to_time_str(x * 60.0))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.set_ylabel(r"Travel time (METROPOLIS2)")
    ax.set_ylim(0.0, tt_max / 60.0)
    ax.yaxis.set_major_formatter(lambda x, _: seconds_to_time_str(x * 60.0))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    fig.tight_layout()
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename, dpi=300)


def compare_travel_time_pairwise_hist(filename=None, tt_max=15.0 * 60.0):
    mp_users = read_raw_output(NB_ITERATIONS - 1)
    mp_users = mp_users.with_columns((pl.col("ta") - pl.col("td")).alias("tt"))
    mp_users = mp_users.sort("traveler_id")
    ms_users = pl.read_parquet(os.path.join(METROPOLIS2_OUTPUT, "agent_results.parquet"))
    ms_users = ms_users.sort("id")
    tt_diff = mp_users["tt"] - ms_users["total_travel_time"]
    bins = np.arange(-tt_max / 60.0, tt_max / 60.0, 0.5)
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.hist(
        tt_diff / 60.0,
        bins=bins,
        histtype="step",
        alpha=0.7,
        color=mpl_utils.CMP(0),
    )
    ax.set_xlabel(r"Travel time (METROPOLIS $-$ METROPOLIS2)")
    ax.set_xlim(-tt_max / 60.0, tt_max / 60.0)
    ax.xaxis.set_major_formatter(lambda x, _: seconds_to_time_str(x * 60.0))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_formatter("{x:n}")
    fig.tight_layout()
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename, dpi=300)


def compare_flows(filename=None, max_flow=6000):
    zipfile = ZipFile(METROPOLIS1_INPUT)
    links = pl.read_csv(find_file(zipfile, "links.tsv").read(), separator="\t")
    paths = read_raw_paths(NB_ITERATIONS - 1, links)
    mp_flows = paths["link_id"].value_counts().sort("link_id")
    routes = pl.read_parquet(os.path.join(METROPOLIS2_OUTPUT, "route_results.parquet"))
    ms_flows = routes["edge_id"].value_counts().sort("edge_id")
    flows = (
        mp_flows.rename({"counts": "mp_counts"})
        .join(
            ms_flows.rename({"counts": "ms_counts"}),
            left_on="link_id",
            right_on=pl.col("edge_id").cast(pl.Int64),
            how="outer",
        )
        .with_columns(pl.col("mp_counts").fill_null(0), pl.col("ms_counts").fill_null(0))
    )
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.scatter(
        flows["mp_counts"], flows["ms_counts"], marker=".", alpha=0.5, color=mpl_utils.CMP(0), s=1
    )
    ax.plot([0, max_flow], [0, max_flow], color=mpl_utils.CMP(1), alpha=0.9)
    ax.set_xlabel("Flows (METROPOLIS)")
    ax.set_xlim(0, max_flow)
    ax.set_ylabel(r"Flows (METROPOLIS2)")
    ax.set_ylim(0, max_flow)
    fig.tight_layout()
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename, dpi=300)


def compare_convergence_dep_time(filename=None):
    prev_df = None
    mp_dep_time_rmses = list()
    for i in range(NB_ITERATIONS):
        df = read_raw_output(i)
        if prev_df is not None:
            rmse = sqrt(((df["td"] - prev_df["td"]) ** 2).mean())
            mp_dep_time_rmses.append(rmse)
        prev_df = df
    it_res = pl.read_parquet(os.path.join(METROPOLIS2_OUTPUT, "iteration_results.parquet"))
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.plot(
        np.arange(2, NB_ITERATIONS + 1),
        mp_dep_time_rmses,
        alpha=0.7,
        color=mpl_utils.CMP(0),
        label="METROPOLIS",
    )
    ax.plot(
        np.arange(2, NB_ITERATIONS + 1),
        it_res["trip_dep_time_rmse"][1:],
        alpha=0.7,
        color=mpl_utils.CMP(1),
        label=r"METROPOLIS2",
    )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}^{\text{dep}}_{\kappa}$ (seconds)")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)


def compare_convergence_expect(filename=None):
    mp_expect_rmses = list()
    for i in range(NB_ITERATIONS):
        df = read_raw_output(i)
        rmse = sqrt(((df["ta"] - df["xpta"]) ** 2).mean())
        mp_expect_rmses.append(rmse)
    it_res = pl.read_parquet(os.path.join(METROPOLIS2_OUTPUT, "iteration_results.parquet"))
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.plot(
        np.arange(1, NB_ITERATIONS + 1),
        mp_expect_rmses,
        alpha=0.7,
        color=mpl_utils.CMP(0),
        label="METROPOLIS",
    )
    ax.plot(
        np.arange(1, NB_ITERATIONS + 1),
        it_res["road_leg_exp_travel_time_diff_rmse"],
        alpha=0.7,
        color=mpl_utils.CMP(1),
        label=r"METROPOLIS2",
    )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}^{\text{expect}}_{\kappa}$ (minutes)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)


def compare_convergence_ttime(filename=None):
    zipfile = ZipFile(METROPOLIS1_INPUT)
    links = pl.read_csv(find_file(zipfile, "links.tsv").read(), separator="\t")
    mp_ttime_rmses = list()
    for i in range(1, NB_ITERATIONS):
        rmse = ttime_rmse(i, links)
        mp_ttime_rmses.append(rmse)
    it_res = pl.read_parquet(os.path.join(METROPOLIS2_OUTPUT, "iteration_results.parquet"))
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.plot(
        np.arange(2, NB_ITERATIONS + 1),
        mp_ttime_rmses,
        alpha=0.7,
        color=mpl_utils.CMP(0),
        label="METROPOLIS",
    )
    ax.plot(
        np.arange(2, NB_ITERATIONS + 1),
        it_res["exp_road_network_weights_rmse"][1:],
        alpha=0.7,
        color=mpl_utils.CMP(1),
        label=r"METROPOLIS2",
    )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}^{T}_{\kappa}$ (seconds)")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename)


def read_raw_ttime(iteration, suffix, db_to_user_id):
    filename = os.path.join(
        METROPOLIS1_OUTPUT, "metrosim_net_arcs_{}_ttime_{}.txt".format(iteration, suffix)
    )
    nb_periods = round((PERIOD[1] - PERIOD[0]) / RECORDING_INTERVAL)
    df = pl.read_csv(
        filename,
        separator="\t",
        has_header=False,
        columns=list(range(nb_periods + 1)),
        new_columns=["link_id"] + list(map(str, range(nb_periods))),
    )
    # We need to convert db link ids to user ids.
    df = df.with_columns(pl.col("link_id").map_dict(db_to_user_id))
    return df


# Compute the RMSE between simulated and historical ttime weights.
def ttime_rmse(iteration, links):
    db_to_user_id = dict(((db_id, user_id) for db_id, user_id in zip(links["db_id"], links["id"])))
    sim_ttimes = read_raw_ttime(iteration, "S", db_to_user_id).drop("link_id").to_numpy()
    hist_ttimes = read_raw_ttime(iteration - 1, "H", db_to_user_id).drop("link_id").to_numpy()
    xs = np.arange(PERIOD[0] + RECORDING_INTERVAL / 2, PERIOD[1], RECORDING_INTERVAL)
    all_xs = np.linspace(PERIOD[0], PERIOD[1], 200)
    rmse = sqrt(
        sum(
            (
                np.mean((np.interp(all_xs, xs, array1) - np.interp(all_xs, xs, array2)) ** 2)
                for (array1, array2) in zip(sim_ttimes, hist_ttimes)
            )
        )
        / len(sim_ttimes)
    )
    return rmse


if __name__ == "__main__":
    if not os.path.isdir(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)
    stats()
    compare_departure_time_distributions(os.path.join(GRAPH_DIR, "departure_time_hist.pdf"))
    compare_travel_time_distributions(os.path.join(GRAPH_DIR, "travel_time_hist.pdf"))
    compare_travel_time_pairwise_scatter(os.path.join(GRAPH_DIR, "travel_time_scatter.png"))
    compare_travel_time_pairwise_hist(os.path.join(GRAPH_DIR, "travel_time_hist_diff.pdf"))
    compare_flows(os.path.join(GRAPH_DIR, "flows_scatter.png"))
    compare_convergence_dep_time(os.path.join(GRAPH_DIR, "dep_time_rmse_comparison.pdf"))
    compare_convergence_expect(os.path.join(GRAPH_DIR, "expect_rmse_comparison.pdf"))
    compare_convergence_ttime(os.path.join(GRAPH_DIR, "ttime_rmse_comparison.pdf"))
