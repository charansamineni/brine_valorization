from parameter_sweep.loop_tool.loop_tool import loopTool, get_working_dir
import brine_valorization.flowsheets.bpmed as bs
import time

__author__ = "Alexander V. Dudchenko"


def main(save_location=None, config_location=None):

    ts = time.time()
    cwd = get_working_dir()
    loopTool(
        f"{config_location}/tds_recovery_sweeps.yaml",
        build_function=bs.build_bpmed,
        initialize_function=bs.initialize,
        optimize_function=bs.solve_model,
        save_name="tds_recovery_sweeps",
        saving_dir=save_location,
        number_of_subprocesses=1,
        num_loop_workers=1,
    )

    print("Total time: ", time.time() - ts)


if __name__ == "__main__":
    config_location = 'src/brine_valorization/analysis/data_generation'
    save_location = 'src/brine_valorization/analysis/data_generation/'
    main(save_location=save_location, config_location=config_location)
