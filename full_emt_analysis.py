import matplotlib.pyplot as plt
import numpy as np

from analyze_measurement_campaigns import collect_config_paths, load_campaign_config
from analyze_porosity_emt import parse_args, collect_measurements

EPS_VACUUM = 1.0

def bruggemann_vi(epsilon_eff: complex, epsilon_ice: complex) -> complex:
    lhs = (EPS_VACUUM - epsilon_eff) / (EPS_VACUUM + 2 * epsilon_eff)
    rhs = (epsilon_ice - epsilon_eff) / (epsilon_ice + 2 * epsilon_eff)
    return lhs / (lhs - rhs)

def maxwell_garnett_vi(epsilon_eff: complex, epsilon_host: complex, epsilon_ice: complex) -> complex:
    numerator = (epsilon_eff - epsilon_host) * (epsilon_ice + 2 * epsilon_host)
    denominator = (epsilon_ice - epsilon_host) * (epsilon_eff + 2 * epsilon_host)
    return numerator / denominator


def lll_vi(epsilon_eff: complex, epsilon_host: complex, epsilon_ice: complex) -> complex:
    return (np.power(epsilon_eff, 1 / 3) - np.power(epsilon_host, 1 / 3)) / (
        np.power(epsilon_ice, 1 / 3) - np.power(epsilon_host, 1 / 3)
    )



if __name__ == "__main__":
    args = parse_args()
    config_paths = collect_config_paths(args.config_dir.resolve(), args.config)
    if not config_paths:
        raise FileNotFoundError("No campaign config files found.")

    campaigns = [load_campaign_config(path) for path in config_paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measurements = collect_measurements(campaigns)


