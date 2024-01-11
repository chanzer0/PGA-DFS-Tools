import sys
from pga_optimizer import *
from pga_gpp_simulator import *

# from PGA_evolutionary_lineup_selector import *
# from PGA_showdown_optimizer import *
from windows_inhibitor import *

# from PGA_late_swaptimizer import *


def main(arguments):
    if len(arguments) < 3 or len(arguments) > 7:
        print("Incorrect usage. Please see `README.md` for proper usage.")
        exit()

    site = arguments[1]
    process = arguments[2]

    # print(arguments)

    if process == "opto":
        num_lineups = arguments[3]
        num_uniques = arguments[4]
        opto = PGA_Optimizer(site, num_lineups, num_uniques)
        opto.optimize()
        opto.output()

    # elif process == 'sd':
    #     num_lineups = arguments[3]
    #     num_uniques = arguments[4]
    #     opto = PGA_Showdown_Optimizer(site, num_lineups, num_uniques)
    #     opto.optimize()
    #     opto.output()

    if process == "sim":
        site = arguments[1]
        field_size = -1
        num_iterations = -1
        use_contest_data = False
        use_file_upload = False
        match_lineup_input_to_field_size = True
        if arguments[3] == "cid":
            use_contest_data = True
        else:
            field_size = arguments[3]

        if arguments[4] == "file":
            use_file_upload = True
            num_iterations = arguments[5]
        else:
            num_iterations = arguments[4]
        # if 'match' in arguments:
        #    match_lineup_input_to_field_size = True
        sim = PGA_GPP_Simulator(
            site,
            field_size,
            num_iterations,
            use_contest_data,
            use_file_upload,
            match_lineup_input_to_field_size,
        )
        # sim.generate_field_lineups()
        sim.run_tournament_simulation()
        sim.output()


if __name__ == "__main__":
    main(sys.argv)
