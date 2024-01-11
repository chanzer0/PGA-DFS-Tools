import itertools
import json
import csv
import os
import datetime
import numpy as np
import pulp as plp


class PGA_Optimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    team_list = []
    lineups = []
    player_dict = {}
    at_least = {}
    at_most = {}
    exactly = {}
    team_limits = {}
    matchup_limits = {}
    matchup_at_least = {}
    global_team_limit = None
    projection_minimum = 0
    randomness_amount = 0

    def __init__(self, site=None, num_lineups=0, num_uniques=1):
        self.site = site
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.load_config()
        self.load_rules()

        self.problem = plp.LpProblem("PGA", plp.LpMaximize)

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

    # Load config from file
    def load_config(self):
        with open(
            os.path.join(os.path.dirname(__file__), "../config.json"),
            encoding="utf-8-sig",
        ) as json_file:
            self.config = json.load(json_file)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = "Name" if self.site == "dk" else "Nickname"
                player_name = row[name_key].replace("-", "#")
                if player_name in self.player_dict:
                    if self.site == "dk":
                        self.player_dict[player_name]["ID"] = int(row["ID"])
                    else:
                        self.player_dict[player_name]["ID"] = row["Id"]

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.exactly = self.config["exactly"]
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])

    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                player_name = row["name"].replace("-", "#")
                if float(row["projection"]) < self.projection_minimum:
                    continue
                self.player_dict[player_name] = {
                    "Fpts": float(row["projection"]),
                    "ID": -1,
                    "Salary": int(row["salary"].replace(",", "")),
                    "Name": row["name"],
                    "Ownership": float(row["ownership"] if "ownership" in row else 0.0),
                    "Optimal": float(row["optimal %"] if "optimal %" in row else 0.0),
                    "Leverage": float(row["leverage"] if "leverage" in row else 0.0),
                    "StdDev": float(row["sttdev"] if "sttdev" in row else 0.0),
                    "Ceiling": float(row["ceiling"] if "ceiling" in row else 0.0),
                    "Floor": float(row["floor"] if "floor" in row else 0.0),
                    "Top6%": float(row["top6%"] if "top6%" in row else 0.0),
                    "Withdraw": float(row["withdraw"] if "withdraw" in row else 0.0),
                    "Win%": float(row["win%"] if "win%" in row else 0.0),
                    "Swt6%": float(row["swt6%"] if "swt6%" in row else 0.0),
                }

    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row["Name"].replace("-", "#")
                if player_name in self.player_dict:
                    self.player_dict[player_name]["Ownership"] = float(
                        row["Ownership %"]
                    )

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        lp_variables = {
            player: plp.LpVariable(player, cat="Binary")
            for player, _ in self.player_dict.items()
        }

        # set the objective - maximize fpts & set randomness amount from config
        self.problem += (
            plp.lpSum(
                np.random.normal(
                    self.player_dict[player]["Fpts"],
                    (self.player_dict[player]["StdDev"] * self.randomness_amount / 100),
                )
                * lp_variables[player]
                for player in self.player_dict
            ),
            "Objective",
        )
        # Set the salary constraints
        max_salary = 50000 if self.site == "dk" else 60000
        min_salary = 49000 if self.site == "dk" else 59000

        # Set min salary if in config
        if (
            "min_lineup_salary" in self.config
            and self.config["min_lineup_salary"] is not None
        ):
            min_salary = self.config["min_lineup_salary"]

        self.problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"] * lp_variables[player]
                for player in self.player_dict
            )
            <= max_salary
        )
        self.problem += (
            plp.lpSum(
                self.player_dict[player]["Salary"] * lp_variables[player]
                for player in self.player_dict
            )
            >= min_salary
        )

        # Address limit rules if any
        for limit, groups in self.at_least.items():
            for group in groups:
                self.problem += plp.lpSum(
                    lp_variables[player.replace("-", "#")] for player in group
                ) >= int(limit)

        for limit, groups in self.at_most.items():
            for group in groups:
                self.problem += plp.lpSum(
                    lp_variables[player.replace("-", "#")] for player in group
                ) <= int(limit)

        if self.site == "dk":
            # Need 6 golfers. pretty easy.
            self.problem += (
                plp.lpSum(lp_variables[player] for player in self.player_dict) >= 6
            )
            self.problem += (
                plp.lpSum(lp_variables[player] for player in self.player_dict) <= 6
            )

        else:
            ## TODO - add fd
            pass

        # Crunch!
        for i in range(self.num_lineups):
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print(
                    "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                        len(self.num_lineups), self.num_lineups
                    )
                )

            if i % 100 == 0:
                print(i)

            # Get the lineup and add it to our list
            players = [
                player for player in lp_variables if lp_variables[player].varValue != 0
            ]

            self.lineups.append(players)

            # Ensure this lineup isn't picked again
            self.problem += (
                plp.lpSum(lp_variables[player] for player in players)
                <= len(players) - self.num_uniques,
                f"Lineup {i}",
            )

            # Set a new random fpts projection within their distribution
            if self.randomness_amount != 0:
                self.problem += (
                    plp.lpSum(
                        np.random.normal(
                            self.player_dict[player]["Fpts"],
                            (
                                self.player_dict[player]["StdDev"]
                                * self.randomness_amount
                                / 100
                            ),
                        )
                        * lp_variables[player]
                        for player in self.player_dict.keys()
                    ),
                    "Objective",
                )

    def output(self):
        print("Lineups done generating. Outputting.")

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_optimal_lineups_{}.csv".format(
                self.site, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ),
        )
        with open(out_path, "w") as f:
            f.write(
                "G,G,G,G,G,G,Salary,Fpts Proj,Ceiling,Own. Product,Own. Sum,Optimal%,Top6%,Win%,Leverage Sum,Swt6%,Withdraw\n"
            )
            for x in self.lineups:
                # print(id_sum, tple)
                salary = sum(self.player_dict[player]["Salary"] for player in x)
                fpts_p = sum(self.player_dict[player]["Fpts"] for player in x)
                own_p = np.prod(
                    [self.player_dict[player]["Ownership"] / 100.0 for player in x]
                )
                own_s = sum([self.player_dict[player]["Ownership"] for player in x])
                ceil = sum([self.player_dict[player]["Ceiling"] for player in x])
                optimal_p = np.prod(
                    [self.player_dict[player]["Optimal"] / 100.0 for player in x]
                )
                top6_p = np.prod(
                    [self.player_dict[player]["Top6%"] / 100.0 for player in x]
                )
                win_p = np.prod(
                    [self.player_dict[player]["Win%"] / 100.0 for player in x]
                )
                leverage_s = sum([self.player_dict[player]["Leverage"] for player in x])
                swt6_p = np.prod(
                    [self.player_dict[player]["Swt6%"] / 100.0 for player in x]
                )
                withdraw_p = np.prod(
                    [self.player_dict[player]["Withdraw"] / 100.0 for player in x]
                )
                # print(sum(self.player_dict[player]['Ownership'] for player in x))
                if self.site == "dk":
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{},{},{},{},{},{}".format(
                        self.player_dict[x[0]]["Name"],
                        self.player_dict[x[0]]["ID"],
                        self.player_dict[x[1]]["Name"],
                        self.player_dict[x[1]]["ID"],
                        self.player_dict[x[2]]["Name"],
                        self.player_dict[x[2]]["ID"],
                        self.player_dict[x[3]]["Name"],
                        self.player_dict[x[3]]["ID"],
                        self.player_dict[x[4]]["Name"],
                        self.player_dict[x[4]]["ID"],
                        self.player_dict[x[5]]["Name"],
                        self.player_dict[x[5]]["ID"],
                        salary,
                        round(fpts_p, 2),
                        ceil,
                        own_p,
                        own_s,
                        optimal_p,
                        top6_p,
                        win_p,
                        leverage_s,
                        swt6_p,
                        withdraw_p,
                    )
                    f.write("%s\n" % lineup_str)
                else:
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{},{},{},{},{},{}".format(
                        self.player_dict[x[0]]["ID"],
                        self.player_dict[x[0]]["Name"],
                        self.player_dict[x[1]]["ID"],
                        self.player_dict[x[1]]["Name"],
                        self.player_dict[x[2]]["ID"],
                        self.player_dict[x[2]]["Name"],
                        self.player_dict[x[3]]["ID"],
                        self.player_dict[x[3]]["Name"],
                        self.player_dict[x[4]]["ID"],
                        self.player_dict[x[4]]["Name"],
                        self.player_dict[x[5]]["ID"],
                        self.player_dict[x[5]]["Name"],
                        salary,
                        round(fpts_p, 2),
                        ceil,
                        own_p,
                        own_s,
                        optimal_p,
                        top6_p,
                        win_p,
                        leverage_s,
                        swt6_p,
                        withdraw_p,
                    )
                    f.write("%s\n" % lineup_str)
        print("Output done.")
