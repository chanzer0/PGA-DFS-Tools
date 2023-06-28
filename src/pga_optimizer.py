import json
import csv
import os
import datetime
import pytz
import timedelta
import numpy as np
import pulp as plp
from itertools import groupby
from random import shuffle, choice


class PGA_Optimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    team_list = []
    lineups = {}
    player_dict = {}
    at_least = {}
    at_most = {}
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

        ownership_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["ownership_path"]),
        )
        self.load_ownership(ownership_path)

        player_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["player_path"]),
        )
        self.load_player_ids(player_path)

        boom_bust_path = os.path.join(
            os.path.dirname(__file__),
            "../{}_data/{}".format(site, self.config["boom_bust_path"]),
        )
        self.load_boom_bust(boom_bust_path)

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
                        self.player_dict[player_name]["RealID"] = int(row["ID"])
                        self.player_dict[player_name]["ID"] = int(row["ID"][-3:])
                        self.player_dict[player_name]["Matchup"] = row[
                            "Game Info"
                        ].split(" ")[0]
                    else:
                        self.player_dict[player_name]["RealID"] = str(row["Id"])
                        self.player_dict[player_name]["ID"] = int(
                            row["Id"].split("-")[1]
                        )
                        self.player_dict[player_name]["Matchup"] = row["Game"].split(
                            " "
                        )[0]

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        # self.team_limits = self.config["team_limits"]
        # self.global_team_limit = int(self.config["global_team_limit"])
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        # self.matchup_limits = self.config["matchup_limits"]
        # self.matchup_at_least = self.config["matchup_at_least"]

    # Need standard deviations to perform randomness
    def load_boom_bust(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row["Name"].replace("-", "#")
                if player_name in self.player_dict:
                    self.player_dict[player_name]["Leverage"] = float(row["Leverage"])
                    self.player_dict[player_name]["StdDev"] = float(row["stddev"])
                    self.player_dict[player_name]["Optimal"] = float(row["Optimal %"])
                    self.player_dict[player_name]["Ceiling"] = float(row["ceiling"])
                    self.player_dict[player_name]["Floor"] = float(row["floor"])
                    self.player_dict[player_name]["Top6%"] = float(row["top6%"])
                    self.player_dict[player_name]["Withdraw"] = float(row["Withdraw"])
                    self.player_dict[player_name]["Win%"] = float(row["win%"])
                    self.player_dict[player_name]["Swt6%"] = float(row["swt6%"])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row["Name"].replace("-", "#")
                if float(row["Total Pts"]) < self.projection_minimum:
                    continue
                self.player_dict[player_name] = {
                    "Fpts": 0.1,
                    "ID": 0,
                    "Salary": 1000,
                    "Name": "",
                    "RealID": 0,
                    "Ownership": 0.1,
                    "Optimal": 0.1,
                    "Leverage": 0.0,
                    "StdDev": 0.1,
                    "Ceiling": 0.0,
                    "Floor": 0.0,
                    "Top6%": 0,
                    "Withdraw": 0,
                    "Win%": 0,
                    "Swt6%": 0,
                }
                self.player_dict[player_name]["Fpts"] = float(row["Total Pts"])
                self.player_dict[player_name]["Salary"] = int(
                    row["Salary"].replace(",", "")
                )
                self.player_dict[player_name]["Name"] = row["Name"]

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

            score = str(self.problem.objective)
            for v in self.problem.variables():
                score = score.replace(v.name, str(v.varValue))

            if i % 100 == 0:
                print(i)
            player_names = [
                v.name.replace("_", " ")
                for v in self.problem.variables()
                if v.varValue != 0
            ]
            fpts = eval(score)

            self.lineups[fpts] = player_names

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
                        for player in self.player_dict
                    ),
                    "Objective",
                )
            else:
                self.problem += plp.lpSum(
                    self.player_dict[player]["Fpts"] * lp_variables[player]
                    for player in self.player_dict
                ) <= (fpts - 0.001)

    def output(self):
        print("Lineups done generating. Outputting.")
        unique = {}
        for fpts, lineup in self.lineups.items():
            if lineup not in unique.values():
                unique[fpts] = lineup

        self.lineups = unique
        if self.num_uniques != 1:
            num_uniq_lineups = plp.OrderedDict(
                sorted(self.lineups.items(), reverse=False, key=lambda t: t[0])
            )
            self.lineups = {}
            for fpts, lineup in num_uniq_lineups.copy().items():
                temp_lineups = list(num_uniq_lineups.values())
                temp_lineups.remove(lineup)
                use_lineup = True
                for x in temp_lineups:
                    common_players = set(x) & set(lineup)
                    roster_size = 6 if self.site == "fd" else 6
                    if (roster_size - len(common_players)) < self.num_uniques:
                        use_lineup = False
                        del num_uniq_lineups[fpts]
                        break
                if use_lineup:
                    self.lineups[fpts] = lineup

        out_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_optimal_lineups_{}.csv".format(
                self.site, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ),
        )
        with open(out_path, "w") as f:
            if self.site == "dk":
                f.write(
                    "G,G,G,G,G,G,Salary,Fpts Proj,Ceiling,Own. Product,Own. Sum,Optimal%,Top6%,Win%,Leverage Sum,Swt6%,Withdraw\n"
                )
                for fpts, x in self.lineups.items():
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
                    leverage_s = sum(
                        [self.player_dict[player]["Leverage"] for player in x]
                    )
                    swt6_p = np.prod(
                        [self.player_dict[player]["Swt6%"] / 100.0 for player in x]
                    )
                    withdraw_p = np.prod(
                        [self.player_dict[player]["Withdraw"] / 100.0 for player in x]
                    )
                    # print(sum(self.player_dict[player]['Ownership'] for player in x))
                    lineup_str = "{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{},{},{},{},{},{}".format(
                        x[0].replace("#", "-"),
                        self.player_dict[x[0]]["RealID"],
                        x[1].replace("#", "-"),
                        self.player_dict[x[1]]["RealID"],
                        x[2].replace("#", "-"),
                        self.player_dict[x[2]]["RealID"],
                        x[3].replace("#", "-"),
                        self.player_dict[x[3]]["RealID"],
                        x[4].replace("#", "-"),
                        self.player_dict[x[4]]["RealID"],
                        x[5].replace("#", "-"),
                        self.player_dict[x[5]]["RealID"],
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
                ## TODO  - add fd
                pass
        print("Output done.")
