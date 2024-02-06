import numpy as np

import mpl_utils


if __name__ == "__main__":
    capacity = 1000
    length = 0.2
    ff_speed = 50
    vehicle_length = 0.008
    densities = np.linspace(0.0, 1.0, 200)
    threshold = densities <= vehicle_length * capacity / ff_speed
    speeds = ff_speed * threshold + vehicle_length * capacity / densities * (~threshold)

    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.plot(densities, speeds, color=mpl_utils.CMP(0), alpha=0.7)
    ax.set_xlabel("Density (occupied length / total length)")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Speed")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig("graph/bottleneck_speed_density_function.pdf")
