import csv
import json
import math
import os
import random
import time
import numpy as np
import pulp as plp
import multiprocessing as mp
import pandas as pd
import numba as nb 
import itertools
import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


@nb.jit(nopython=True)  # nopython mode ensures the function is fully optimized
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2

class PGA_GPP_Simulator:
    config = None
    player_dict = {}
    field_lineups = {}
    gen_lineup_list = []
    roster_construction = []
    salary = None
    optimal_score = None
    field_size = None
    team_list = []
    num_iterations = None
    site = None
    payout_structure = {}
    use_contest_data = False
    cut_event = False
    entry_fee = None
    use_lineup_input = None
    projection_minimum = 15
    randomness_amount = 100
    min_lineup_salary = 48000
    max_pct_off_optimal = 0.4
    seen_lineups = {}
    seen_lineups_ix = {}

    def __init__(
        self,
        site,
        field_size,
        num_iterations,
        use_contest_data,
        use_lineup_input,
        match_lineup_input_to_field_size,
    ):
        self.site = site
        self.use_lineup_input = use_lineup_input
        self.match_lineup_input_to_field_size = match_lineup_input_to_field_size
        self.load_config()
        self.load_rules()
        projection_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["projection_path"]),
        )
        self.load_projections(projection_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["player_path"]),
        )
        self.load_player_ids(player_path)

        if site == "dk":
            self.roster_construction = ["G", "G", "G", "G", "G", "G"]
            self.salary = 50000

        self.use_contest_data = use_contest_data
        if use_contest_data:
            contest_path = os.path.join(
                os.path.dirname(__file__),
                "../{}_data/{}".format(site, self.config["contest_structure_path"]),
            )
            self.load_contest_data(contest_path)
            print("Contest payout structure loaded.")
        else:
            self.field_size = int(field_size)
            self.payout_structure = {0: 0.0}
            self.entry_fee = 0
        self.num_iterations = int(num_iterations)
        self.get_optimal()
        if self.use_lineup_input:
            self.load_lineups_from_file()
        if self.match_lineup_input_to_field_size or len(self.field_lineups) == 0:
            self.generate_field_lineups()

    def load_rules(self):
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.min_lineup_salary = int(self.config["min_lineup_salary"])
        self.max_pct_off_optimal = float(self.config["max_pct_off_optimal"])
        self.cut_event = self.config["cut_event"]

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `nba_optimizer.py`
    def get_optimal(self):
        problem = plp.LpProblem("PGA", plp.LpMaximize)
        lp_variables = {
            player: plp.LpVariable(player, cat="Binary")
            for player, _ in self.player_dict.items()
        }

        # set the objective - maximize fpts
        problem += (
            plp.lpSum(
                self.player_dict[player]["FieldFpts"] * lp_variables[player]
                for player in self.player_dict
            ),
            "Objective",
        )

        # Set the salary constraints
        problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"] * lp_variables[player]
                for player in self.player_dict
            )
            <= self.salary
        )

        if self.site == "dk":
            # Can only roster 6 total players
            problem += (
                plp.lpSum(lp_variables[player] for player in self.player_dict) == 6
            )

        # Crunch!
        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print(
                "Infeasibility reached - we failed to generate an optimal lineup. Please check all files are present and formatted correctly. Otherwise, submit a ticket on the github."
            )

        score = str(problem.objective)
        for v in problem.variables():
            score = score.replace(v.name, str(v.varValue))

        self.optimal_score = eval(score)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = "Name" if self.site == "dk" else "Nickname"
                player_name = row[name_key].replace("-", "#").lower()
                if player_name in self.player_dict:
                    if self.site == "dk":
                        self.player_dict[player_name]["ID"] = int(row["ID"])
                    else:
                        self.player_dict[player_name]["ID"] = row["Id"]
                else:
                    print(player_name + " not found in player dict")

    def load_contest_data(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if self.field_size is None:
                    self.field_size = int(row["Field Size"])
                if self.entry_fee is None:
                    self.entry_fee = float(row["Entry Fee"])
                # multi-position payouts
                if "-" in row["Place"]:
                    indices = row["Place"].split("-")
                    # print(indices)
                    # have to add 1 to range to get it to generate value for everything
                    for i in range(int(indices[0]), int(indices[1]) + 1):
                        # print(i)
                        # Where I'm from, we 0 index things. Thus, -1 since Payout starts at 1st place
                        if i >= self.field_size:
                            break
                        self.payout_structure[i - 1] = float(
                            row["Payout"].split(".")[0].replace(",", "")
                        )
                # single-position payouts
                else:
                    if int(row["Place"]) >= self.field_size:
                        break
                    self.payout_structure[int(row["Place"]) - 1] = float(
                        row["Payout"].split(".")[0].replace(",", "")
                    )
        # print(self.payout_structure)
    
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#").lower()
                fpts = float(row["fpts"])
                if fpts < self.projection_minimum:
                    continue
                if "fieldfpts" in row:
                    if row["fieldfpts"] == "":
                        fieldFpts = fpts
                    else:
                        fieldFpts = float(row["fieldfpts"])
                else:
                    fieldFpts = fpts
                if row["salary"]:
                    sal = int(row["salary"].replace(",", ""))
                if 'makecut' in row:
                    makecut = row['makecut']
                else:
                    makecut = 0
                if 'winprob' in row:
                    winprob = row['winprob']
                else:
                    winprob = 0
                if "stddev" in row:
                    if row["stddev"] == "" or float(row["stddev"]) == 0:
                        stddev = fpts * self.default_var
                    else:
                        stddev = float(row["stddev"])
                else:
                    stddev = fpts * self.default_var     
                if "ceiling" in row:
                    if row["ceiling"] == "" or float(row["ceiling"]) == 0:
                        ceil = fpts + stddev
                    else:
                        ceil = float(row["ceiling"])
                else:
                    ceil = fpts + stddev  
                own = float(row["own%"].replace("%", ""))
                if own == 0:
                    own = 0.1         
                self.player_dict[player_name] = {
                    "Fpts": fpts,
                    "Position": ['G'],
                    "ID": 0,
                    "Salary": sal,
                    "StdDev": stddev,
                    "Ceiling": ceil,
                    "Ownership": own,
                    "In Lineup": False,
                    "Name": row['name'],
                    "FieldFpts": fieldFpts,
                    "MakeCut": makecut,
                    "WinProb": winprob
                }

    def remap(self, fieldnames):
        return ["PG", "PG2", "SG", "SG2", "SF", "SF2", "PF", "PF2", "C"]

    def extract_id(self,cell_value):
        if "(" in cell_value and ")" in cell_value:
            return cell_value.split("(")[1].replace(")", "")
        elif ":" in cell_value:
            return cell_value.split(":")[0]
        else:
            return cell_value

    def load_lineups_from_file(self):
        print("loading lineups")
        i = 0
        path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(self.site, "tournament_lineups.csv"),
        )
        with open(path) as file:
            reader = pd.read_csv(file)
            lineup = []
            bad_lus = []
            bad_players = []
            j = 0
            for i, row in reader.iterrows():
                if i == self.field_size:
                    break
                lineup = [
                    int(self.extract_id(str(row[q])))
                    for q in range(len(self.roster_construction))
                ]
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print("player id {} in lineup {} not found in player dict".format(l, i))
                        #if l in self.id_name_dict:
                        #    print(self.id_name_dict[l])
                        bad_players.append(l)
                        error = True
                if len(lineup) < len(self.roster_construction):
                    print("lineup {} doesn't match roster construction size".format(i))
                    continue
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                if not error:
                    lineup_list = sorted(lineup)           
                    lineup_set = frozenset(lineup_list)

                    # Keeping track of lineup duplication counts
                    if lineup_set in self.seen_lineups:
                        self.seen_lineups[lineup_set] += 1
                    else:
                        self.field_lineups[j] = {
                                "Lineup": lineup,
                                "Wins": 0,
                                "Top1Percent": 0,
                                "ROI": 0,
                                "Cashes": 0,
                                "Type": "opto",
                                "Count": 1
                        }
                        # Add to seen_lineups and seen_lineups_ix
                        self.seen_lineups[lineup_set] = 1
                        self.seen_lineups_ix[lineup_set] = j
                        j += 1
        print("loaded {} lineups".format(j))
        # print(self.field_lineups)

    @staticmethod
    def generate_lineups(
        lu_num,
        names,
        in_lineup,
        pos_matrix,
        ownership,
        salary_floor,
        salary_ceiling,
        optimal_score,
        salaries,
        projections,
        max_pct_off_optimal,
    ):
        # new random seed for each lineup (without this there is a ton of dupes)
        rng = np.random.Generator(np.random.PCG64())
        min_salary = np.quantile(salaries, 0.3)
        lus = {}
        # make sure nobody is already showing up in a lineup
        if sum(in_lineup) != 0:
            in_lineup.fill(0)
        reject = True
        while reject:
            salary = 0
            proj = 0
            if sum(in_lineup) != 0:
                in_lineup.fill(0)
            lineup = []
            q = 0
            for pos in pos_matrix:
                # calculate difference between current lineup salary and salary ceiling
                if q < 5:
                    salary_diff = salary_ceiling - (salary + min_salary)
                else:
                    salary_diff = salary_ceiling - salary
                # check for players eligible for the position and make sure they arent in a lineup and the player's salary is less than or equal to salary_diff, returns a list of indices of available player
                valid_players = np.where((pos > 0) & (in_lineup == 0) & (salaries <= salary_diff))
                # grab names of players eligible
                plyr_list = names[valid_players]
                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                prob_list = ownership[valid_players]
                prob_list = prob_list / prob_list.sum()  # normalize to ensure it sums to 1
                if q == 5:
                    boosted_salaries = np.array([salary_boost(s, salary_ceiling) for s in salaries[valid_players]])
                    boosted_probabilities = prob_list * boosted_salaries
                    boosted_probabilities /= boosted_probabilities.sum()  # normalize to ensure it sums to 1
                try:
                    if q == 5:
                        choice = rng.choice(plyr_list, p=boosted_probabilities)
                    else:
                        choice = rng.choice(plyr_list, p=prob_list)
                except:
                    # if remaining_salary <= np.min(salaries):
                    #     reject_counters["salary_too_high"] += 1
                    # else:
                    #     reject_counters["salary_too_low"]
                    salary = 0
                    proj = 0
                    lineup = []
                    in_lineup.fill(0)  # Reset the in_lineup array
                    k = 0  # Reset the player index
                    continue  # Skip to the next iteration of the while loop
                choice_idx = np.where(names == choice)[0]
                lineup.append(choice)
                in_lineup[choice_idx] = 1
                salary += salaries[choice_idx]
                proj += projections[choice_idx]
                q+=1
            # Must have a reasonable salary
            if salary >= salary_floor and salary <= salary_ceiling and len(lineup) == 6:
                # Must have a reasonable projection (within 60% of optimal) **people make a lot of bad lineups
                reasonable_projection = optimal_score - (
                    max_pct_off_optimal * optimal_score
                )
                if proj >= reasonable_projection:
                    reject = False
                    lu = {
                        "Lineup": lineup,
                        "Wins": 0,
                        "Top1Percent": 0,
                        "ROI": 0,
                        "Cashes": 0,
                        "Type": "generated",
                        "Count": 0
                    }
        return lu

    def generate_field_lineups(self):
        diff = self.field_size - len(self.field_lineups)
        if diff <= 0:
            print(
                "supplied lineups >= contest field size. only retrieving the first "
                + str(self.field_size)
                + " lineups"
            )
        else:
            print("Generating " + str(diff) + " lineups.")
            names = list(self.player_dict.keys())
            in_lineup = np.zeros(shape=len(names))
            i = 0
            ownership = np.array(
                [
                    self.player_dict[player_name]["Ownership"] / 100
                    for player_name in names
                ]
            )
            salaries = np.array(
                [self.player_dict[player_name]["Salary"] for player_name in names]
            )
            projections = np.array(
                [self.player_dict[player_name]["FieldFpts"] for player_name in names]
            )
            positions = []
            for pos in self.roster_construction:
                pos_list = []
                own = []
                for player_name in names:
                    if pos in self.player_dict[player_name]["Position"]:
                        pos_list.append(1)
                    else:
                        pos_list.append(0)
                i += 1
                positions.append(np.array(pos_list))
            pos_matrix = np.array(positions)
            names = np.array(names)
            optimal_score = self.optimal_score
            salary_floor = (
                self.min_lineup_salary
            )  # anecdotally made the most sense when looking at previous contests
            salary_ceiling = self.salary
            max_pct_off_optimal = self.max_pct_off_optimal
            problems = []
            # creating tuples of the above np arrays plus which lineup number we are going to create
            for i in range(diff):
                lu_tuple = (
                    i,
                    names,
                    in_lineup,
                    pos_matrix,
                    ownership,
                    salary_floor,
                    salary_ceiling,
                    optimal_score,
                    salaries,
                    projections,
                    max_pct_off_optimal,
                )
                problems.append(lu_tuple)
            start_time = time.time()
            with mp.Pool() as pool:
                output = pool.starmap(self.generate_lineups, problems)
                print(
                    "number of running processes =",
                    pool.__dict__["_processes"]
                    if (pool.__dict__["_state"]).upper() == "RUN"
                    else None,
                )
                pool.close()
                pool.join()
            print("pool closed")
            self.update_field_lineups(output, diff)
            end_time = time.time()
            print("lineups took " + str(end_time - start_time) + " seconds")
            print(str(diff) + " field lineups successfully generated")
            # print(self.field_lineups)

    def update_field_lineups(self, output, diff):
        if len(self.field_lineups) == 0:
            new_keys = list(range(0, self.field_size))
        else:
            new_keys = list(range(max(self.field_lineups.keys()) + 1, max(self.field_lineups.keys()) + 1 + diff))
        nk = new_keys[0]
        for i, o in enumerate(output):
            #print(o.values())
            lineup_list = sorted(o['Lineup'])
            lineup_set = frozenset(lineup_list)
            #print(lineup_set)

            # Keeping track of lineup duplication counts
            if lineup_set in self.seen_lineups:
                self.seen_lineups[lineup_set] += 1
                        
                # Increase the count in field_lineups using the index stored in seen_lineups_ix
                self.field_lineups[self.seen_lineups_ix[lineup_set]]["Count"] += 1
            else:
                self.seen_lineups[lineup_set] = 1
                
                if nk in self.field_lineups.keys():
                    print("bad lineups dict, please check dk_data files")
                else:
                    # Convert dict_values to a dictionary before assignment
                    lineup_data = dict(o)
                    lineup_data['Lineup'] = lineup_list
                    lineup_data['Count'] += self.seen_lineups[lineup_set]

                    # Now assign the dictionary to the field_lineups
                    self.field_lineups[nk] = lineup_data                 
                    # Store the new nk in seen_lineups_ix for quick access in the future
                    self.seen_lineups_ix[lineup_set] = nk
                    nk += 1

    @staticmethod
    @nb.jit(nopython=True)
    def calculate_payouts(args):
        (
            ranks,
            payout_array,
            entry_fee,
            field_lineup_keys,
            use_contest_data,
            field_lineups_count,
        ) = args
        num_lineups = len(field_lineup_keys)
        combined_result_array = np.zeros(num_lineups)

        payout_cumsum = np.cumsum(payout_array)

        for r in range(ranks.shape[1]):
            ranks_in_sim = ranks[:, r]
            payout_index = 0
            for lineup_index in ranks_in_sim:
                lineup_count = field_lineups_count[lineup_index]
                prize_for_lineup = (
                    (
                        payout_cumsum[payout_index + lineup_count - 1]
                        - payout_cumsum[payout_index - 1]
                    )
                    / lineup_count
                    if payout_index != 0
                    else payout_cumsum[payout_index + lineup_count - 1] / lineup_count
                )
                combined_result_array[lineup_index] += prize_for_lineup
                payout_index += lineup_count
        return combined_result_array

    def run_tournament_simulation(self):
        print("Running " + str(self.num_iterations) + " simulations")
        start_time = time.time()
        plot_folder = "simulation_plots"
        temp_fpts_dict = {}

        # Create a directory for plots if it does not exist
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        if self.cut_event:
            kmeans_5 = pickle.load(open("src/cluster_data/kmeans_5_model.pkl", "rb"))
            scaler = pickle.load(open("src/cluster_data/scaler.pkl", "rb"))
            for k, v in self.player_dict.items():
                makecut = v['MakeCut']
                salary = v['Salary']
                winprob = v['WinProb']

                # Create a DataFrame for the new data
                # Note the double square brackets to ensure it's treated as one row
                new_data = pd.DataFrame([[salary, makecut, winprob]], columns=['salary', 'make_cut_prob', 'win_prob'])
                
                # Scale the new data
                scaled_new_data = scaler.transform(new_data)

                # Predict the cluster
                player_cluster = kmeans_5.predict(scaled_new_data)[0]  #

                # Load the GMM model for the predicted cluster
                gmm = pickle.load(open(f"src/cluster_data/gmm_cluster_{player_cluster}.pkl", "rb"))

                # Sample from the GMM
                samples = gmm.sample(self.num_iterations)
                temp_fpts_dict[k] = samples[0][:, 0]  # Assuming fantasy points are the first dimension

                # # Save the distribution of simulated fantasy points for each player as an image
                # plt.figure(figsize=(8, 4))
                # plt.hist(temp_fpts_dict[k], bins=30, density=True, alpha=0.6, color='g')
                # plt.title(f'Simulated Fantasy Points Distribution for Player {k} with Cluster {player_cluster}')
                # plt.xlabel('Fantasy Points')
                # plt.ylabel('Density')
                # plot_path = os.path.join(plot_folder, f'player_{k}_simulation.png')
                # plt.savefig(plot_path)
                # plt.close()

        else:
            temp_fpts_dict = {
                p: np.random.normal(
                    s["Fpts"],
                    s["StdDev"] * self.randomness_amount / 100,
                    size=self.num_iterations,
                )
                for p, s in self.player_dict.items()
            }

                # Uncomment the next line to disable plotting during non-testing runs
                # break  # Remove or comment out this line to enable plotting for all players

        # generate arrays for every sim result for each player in the lineup and sum
        fpts_array = np.zeros(shape=(len(self.field_lineups), self.num_iterations))
        # converting payout structure into an np friendly format, could probably just do this in the load contest function
        # print(self.field_lineups)
        # print(temp_fpts_dict)
        # print(payout_array)
        # print(self.player_dict[('patrick mahomes', 'FLEX', 'KC')])
        field_lineups_count = np.array(
            [self.field_lineups[idx]["Count"] for idx in self.field_lineups.keys()]
        )

        for index, values in self.field_lineups.items():
            try:
                fpts_sim = sum([temp_fpts_dict[player] for player in values["Lineup"]])
            except KeyError:
                for player in values["Lineup"]:
                    if player not in temp_fpts_dict.keys():
                        print(player)
                        # for k,v in self.player_dict.items():
                        # if v['ID'] == player:
                        #        print(k,v)
                # print('cant find player in sim dict', values["Lineup"], temp_fpts_dict.keys())
            # store lineup fpts sum in 2d np array where index (row) corresponds to index of field_lineups and columns are the fpts from each sim
            fpts_array[index] = fpts_sim

        fpts_array = fpts_array.astype(np.float16)
        # ranks = np.argsort(fpts_array, axis=0)[::-1].astype(np.uint16)
        ranks = np.argsort(-fpts_array, axis=0).astype(np.uint32)

        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)
        cashes, cash_counts = np.unique(ranks[0:len(list(self.payout_structure.values()))], return_counts=True)

        top1pct, top1pct_counts = np.unique(
            ranks[0 : math.ceil(0.01 * len(self.field_lineups)), :], return_counts=True
        )

        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        # Adjusted ROI calculation
        # print(field_lineups_count.shape, payout_array.shape, ranks.shape, fpts_array.shape)

        # Split the simulation indices into chunks
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        chunk_size = self.num_iterations // 16  # Adjust chunk size as needed
        simulation_chunks = [
            (
                ranks[:, i : min(i + chunk_size, self.num_iterations)].copy(),
                payout_array,
                self.entry_fee,
                field_lineups_keys_array,
                self.use_contest_data,
                field_lineups_count,
            )  # Adding field_lineups_count here
            for i in range(0, self.num_iterations, chunk_size)
        ]

        # Use the pool to process the chunks in parallel
        with mp.Pool() as pool:
            results = pool.map(self.calculate_payouts, simulation_chunks)

        combined_result_array = np.sum(results, axis=0)

        total_sum = 0
        index_to_key = list(self.field_lineups.keys())
        for idx, roi in enumerate(combined_result_array):
            lineup_key = index_to_key[idx]
            lineup_count = self.field_lineups[lineup_key][
                "Count"
            ]  # Assuming "Count" holds the count of the lineups
            total_sum += roi * lineup_count
            self.field_lineups[lineup_key]["ROI"] += roi

        for idx in self.field_lineups.keys():
            if idx in wins:
                self.field_lineups[idx]["Wins"] += win_counts[np.where(wins == idx)][0]
            if idx in top1pct:
                self.field_lineups[idx]["Top1Percent"] += top1pct_counts[
                    np.where(top1pct == idx)
                ][0]
            if idx in cashes:
                self.field_lineups[idx]["Cashes"] += cash_counts[np.where(cashes == idx)][0]

        end_time = time.time()
        diff = end_time - start_time
        print(
            str(self.num_iterations)
            + " tournament simulations finished in "
            + str(diff)
            + " seconds. Outputting."
        )

    def output(self):
        unique = {}
        for index, x in self.field_lineups.items():
            salary = sum(self.player_dict[player]["Salary"] for player in x["Lineup"])
            fpts_p = sum(self.player_dict[player]["Fpts"] for player in x["Lineup"])
            ceil_p = sum(self.player_dict[player]["Ceiling"] for player in x["Lineup"])
            own_p = np.prod(
                [
                    self.player_dict[player]["Ownership"] / 100.0
                    for player in x["Lineup"]
                ]
            )
            win_p = round(x["Wins"] / self.num_iterations * 100, 2)
            top10_p = round(x["Top1Percent"] / self.num_iterations * 100, 2)
            cash_p = round(x["Cashes"] / self.num_iterations * 100, 2)
            lu_type = x["Type"]
            simDupes = x['Count']
            if self.site == "dk":
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(x["ROI"] / self.num_iterations, 2)
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{}%,{}%,{}%,{}%,{},${},{},{}".format(
                        x["Lineup"][0].replace("#", "-"),
                        self.player_dict[x["Lineup"][0].replace("-", "#")]["ID"],
                        x["Lineup"][1].replace("#", "-"),
                        self.player_dict[x["Lineup"][1].replace("-", "#")]["ID"],
                        x["Lineup"][2].replace("#", "-"),
                        self.player_dict[x["Lineup"][2].replace("-", "#")]["ID"],
                        x["Lineup"][3].replace("#", "-"),
                        self.player_dict[x["Lineup"][3].replace("-", "#")]["ID"],
                        x["Lineup"][4].replace("#", "-"),
                        self.player_dict[x["Lineup"][4].replace("-", "#")]["ID"],
                        x["Lineup"][5].replace("#", "-"),
                        self.player_dict[x["Lineup"][5].replace("-", "#")]["ID"],
                        fpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        cash_p,
                        roi_p,
                        own_p,
                        roi_round,
                        lu_type,
                        simDupes
                    )
                else:
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{}%,{}%,{},{}%,{},{}".format(
                        x["Lineup"][0].replace("#", "-"),
                        self.player_dict[x["Lineup"][0]]["ID"],
                        x["Lineup"][1].replace("#", "-"),
                        self.player_dict[x["Lineup"][1]]["ID"],
                        x["Lineup"][2].replace("#", "-"),
                        self.player_dict[x["Lineup"][2]]["ID"],
                        x["Lineup"][3].replace("#", "-"),
                        self.player_dict[x["Lineup"][3]]["ID"],
                        x["Lineup"][4].replace("#", "-"),
                        self.player_dict[x["Lineup"][4]]["ID"],
                        x["Lineup"][5].replace("#", "-"),
                        self.player_dict[x["Lineup"][5]]["ID"],
                        fpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        own_p,
                        cash_p,
                        lu_type,
                        simDupes
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(x["ROI"] / self.num_iterations, 2)
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{}%,{}%,{}%,{}%,{},${},{},{}".format(
                        self.player_dict[x["Lineup"][0].replace("-", "#")]["ID"],
                        x["Lineup"][0].replace("#", "-"),
                        self.player_dict[x["Lineup"][1].replace("-", "#")]["ID"],
                        x["Lineup"][1].replace("#", "-"),
                        self.player_dict[x["Lineup"][2].replace("-", "#")]["ID"],
                        x["Lineup"][2].replace("#", "-"),
                        self.player_dict[x["Lineup"][3].replace("-", "#")]["ID"],
                        x["Lineup"][3].replace("#", "-"),
                        self.player_dict[x["Lineup"][4].replace("-", "#")]["ID"],
                        x["Lineup"][4].replace("#", "-"),
                        self.player_dict[x["Lineup"][5].replace("-", "#")]["ID"],
                        x["Lineup"][5].replace("#", "-"),
                        self.player_dict[x["Lineup"][6].replace("-", "#")]["ID"],
                        x["Lineup"][6].replace("#", "-"),
                        self.player_dict[x["Lineup"][7].replace("-", "#")]["ID"],
                        x["Lineup"][7].replace("#", "-"),
                        self.player_dict[x["Lineup"][8].replace("-", "#")]["ID"],
                        x["Lineup"][8].replace("#", "-"),
                        fpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        cash_p,
                        roi_p,
                        own_p,
                        roi_round,
                        lu_type,
                        simDupes
                    )
                else:
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{}%,{}%,{},{}%,{},{}".format(
                        self.player_dict[x["Lineup"][0].replace("-", "#")]["ID"],
                        x["Lineup"][0].replace("#", "-"),
                        self.player_dict[x["Lineup"][1].replace("-", "#")]["ID"],
                        x["Lineup"][1].replace("#", "-"),
                        self.player_dict[x["Lineup"][2].replace("-", "#")]["ID"],
                        x["Lineup"][2].replace("#", "-"),
                        self.player_dict[x["Lineup"][3].replace("-", "#")]["ID"],
                        x["Lineup"][3].replace("#", "-"),
                        self.player_dict[x["Lineup"][4].replace("-", "#")]["ID"],
                        x["Lineup"][4].replace("#", "-"),
                        self.player_dict[x["Lineup"][5].replace("-", "#")]["ID"],
                        x["Lineup"][5].replace("#", "-"),
                        self.player_dict[x["Lineup"][6].replace("-", "#")]["ID"],
                        x["Lineup"][6].replace("#", "-"),
                        self.player_dict[x["Lineup"][7].replace("-", "#")]["ID"],
                        x["Lineup"][7].replace("#", "-"),
                        self.player_dict[x["Lineup"][8].replace("-", "#")]["ID"],
                        x["Lineup"][8].replace("#", "-"),
                        fpts_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        own_p,
                        cash_p,
                        lu_type,
                        simDupes
                    )
            unique[index] = lineup_str

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_lineups_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            if self.site == "dk":
                if self.use_contest_data:
                    f.write(
                        "G,G,G,G,G,G,Fpts Proj,Ceiling,Salary,Win %,Top 1%,Cash%,ROI%,Proj. Own. Product, Avg. Return,Type,Simulated Duplicates\n"
                    )
                else:
                    f.write(
                        "G,G,G,G,G,G,Fpts Proj,Ceiling,Salary,Win %,Top 1%,Proj. Own. Product,Cash %,Type,Simulated Duplicates\n"
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    f.write(
                        "PG,PG,SG,SG,SF,SF,PF,PF,C,Fpts Proj,Ceiling,Salary,Win %,Top 1%,Cash%,ROI%,Proj. Own. Product, Avg. Return,Type,Simulated Duplicates\n"
                    )
                else:
                    f.write(
                        "PG,PG,SG,SG,SF,SF,PF,PF,C,Fpts Proj,Ceiling,Salary,Win %,Top 1%,Proj. Own. Product,Cash %,Type,Simulated Duplicates\n"
                    )

            for fpts, lineup_str in unique.items():
                f.write("%s\n" % lineup_str)

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_player_exposure_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(out_path, "w") as f:
            f.write("Player,Win%,Top1%,Sim. Own%,Proj. Own%,Avg. Return\n")
            unique_players = {}
            for val in self.field_lineups.values():
                for player in val["Lineup"]:
                    if player not in unique_players:
                        unique_players[player] = {
                            "Wins": val["Wins"],
                            "Top1Percent": val["Top1Percent"],
                            "In": 1,
                            "ROI": val["ROI"],
                            "Cashes": val["Cashes"],
                        }
                    else:
                        unique_players[player]["Wins"] = (
                            unique_players[player]["Wins"] + val["Wins"]
                        )
                        unique_players[player]["Top1Percent"] = (
                            unique_players[player]["Top1Percent"] + val["Top1Percent"]
                        )
                        unique_players[player]["In"] = unique_players[player]["In"] + 1
                        unique_players[player]["ROI"] = (
                            unique_players[player]["ROI"] + val["ROI"]
                        )
                        unique_players[player]["Cashes"] = (
                            unique_players[player]["Cashes"] + val["Cashes"]
                        )                        

            for player, data in unique_players.items():
                field_p = round(data["In"] / self.field_size * 100, 2)
                win_p = round(data["Wins"] / self.num_iterations * 100, 2)
                top10_p = round(data["Top1Percent"] / self.num_iterations / 10 * 100, 2)
                roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 2)
                cash_p = round(data["Cashes"] / self.num_iterations * 100, 2)
                proj_own = self.player_dict[player]["Ownership"]
                f.write(
                    "{},{}%,{}%,{}%,{}%,{}%,${}\n".format(
                        player.replace("#", "-"),
                        win_p,
                        top10_p,
                        cash_p,
                        field_p,
                        proj_own,
                        roi_p,
                    )
                )
