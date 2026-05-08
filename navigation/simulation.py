from config.config_loader import load_config

from paths import *



SCENARIO = SCENARIOS_DIR / "base_scenario.yaml"

def main():
    # Load configs
    config = load_config(SCENARIO)

    print(config.balloon.initial.latitude_deg)
    print(config.orbiter.initial_orbit.inclination_deg)










if __name__ == "__main__":
    main()