"""Smoke for kitchens with more than two cooks: the game's own agents, scripted (their
test mode replays a fixed plan per cook, so no LLM is called), through the loop of
main.py, on three_cooks and four_cooks. It checks that the layouts load, that every
cook reaches what its role says, that items travel from cook to cook over the shared
counters (place_obj_on_counter(<teammate>)), that the order is delivered, and that the
log holds one entry per cook (evaluation: success, subtasks, ticks).

    cd src && python nplayers_smoke.py            # every layout and order below
    python nplayers_smoke.py --layout four_cooks --order baked_potato_slices
"""

import argparse
import copy
import json
import os
import random
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # the game reads prompts/ from cwd

from overcooked_ai_py.agents.agent import AgentGroup  # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv  # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld  # noqa: E402
from overcooked_ai_py.planning.search import get_intersect_counter  # noqa: E402

from collab.modules import make_statistics_dict  # noqa: E402
from eval_utils import team_summary  # noqa: E402
from utils import combine_statistic_dicts, make_agent  # noqa: E402

# layout -> order -> each cook's plan (retried until the game accepts it)
PLANS = {
    "three_cooks": {
        "boiled_egg": {
            "runner": ["pickup(egg,ingredient_dispenser)", "place_obj_on_counter(assistant)"],
            "assistant": ["pickup(egg,counter)", "place_obj_on_counter(chef)"],
            "chef": ["pickup(egg,counter)", "put_obj_in_utensil(pot0)", "cook(pot0)",
                     "pickup(boiled_egg,pot0)", "deliver_soup()"],
        },
        "baked_potato_slices": {
            "runner": ["pickup(potato,ingredient_dispenser)", "place_obj_on_counter(assistant)"],
            "assistant": ["pickup(potato,counter)", "put_obj_in_utensil(chopping_board0)",
                          "cut(chopping_board0)", "pickup(potato_slices,chopping_board0)",
                          "place_obj_on_counter(chef)"],
            "chef": ["pickup(potato_slices,counter)", "put_obj_in_utensil(oven0)", "bake(oven0)",
                     "pickup(baked_potato_slices,oven0)", "deliver_soup()"],
        },
    },
    "four_cooks": {
        "boiled_egg": {
            "runner": ["pickup(egg,ingredient_dispenser)", "place_obj_on_counter(baker)"],
            "baker": ["pickup(egg,counter)", "place_obj_on_counter(chef)"],
            "assistant": [],
            "chef": ["pickup(egg,counter)", "put_obj_in_utensil(pot0)", "cook(pot0)",
                     "pickup(boiled_egg,pot0)", "deliver_soup()"],
        },
        "baked_potato_slices": {
            # the assistant and the baker share no counter: the chef passes it on
            "runner": ["pickup(potato,ingredient_dispenser)", "place_obj_on_counter(assistant)"],
            "assistant": ["pickup(potato,counter)", "put_obj_in_utensil(chopping_board0)",
                          "cut(chopping_board0)", "pickup(potato_slices,chopping_board0)",
                          "place_obj_on_counter(chef)"],
            "chef": ["pickup(potato_slices,counter)", "place_obj_on_counter(baker)",
                     "pickup(baked_potato_slices,counter)", "deliver_soup()"],
            "baker": ["pickup(potato_slices,counter)", "put_obj_in_utensil(oven0)", "bake(oven0)",
                      "pickup(baked_potato_slices,oven0)", "place_obj_on_counter(chef)"],
        },
    },
}  # fmt: skip


def check_layout(mdp, agents, state):
    """Every cook reaches the utensils and dispensers its role lists, and no other."""
    agents[0].build_access_utensil(state)
    problems = []
    dispensers = {
        "ingredient_dispenser": mdp.get_ingredient_dispenser_locations(),
        "dish_dispenser": mdp.get_dish_dispenser_locations(),
    }
    for i, (agent, spec) in enumerate(zip(agents, mdp.agents_spec)):
        kinds = {u.rstrip("0123456789") for u in mdp.utensil_lists[i]}
        if kinds != set(spec["utensils"]):
            problems.append(f"{spec['role']} reaches {sorted(kinds)}, role says {sorted(spec['utensils'])}")
        for name, locs in dispensers.items():
            goals = agent.mlam._get_ml_actions_for_positions(locs)
            ok = any(
                agent.mlam.motion_planner.is_valid_motion_start_goal_pair(state.players[i].pos_and_or, g)
                for g in goals
            )
            if ok != (name in spec["dispensers"]):
                problems.append(f"{spec['role']} reach of {name} is {ok}")
    shared = {
        (a.name, b.name): len(get_intersect_counter(state.players_pos_and_or[i],
                                                    state.players_pos_and_or[j], mdp, a.mlam))
        for i, a in enumerate(agents) for j, b in enumerate(agents) if i < j
    }  # fmt: skip
    return problems, shared


def run(layout, order, horizon=80):
    random.seed(0)
    np.random.seed(0)
    mdp = OvercookedGridworld.from_layout_name(layout)
    mdp.start_order_list = [order]
    mdp.one_task_mode = True
    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()
    roles = [a["role"] for a in mdp.agents_spec]
    agents = [make_agent("LLMPair", mdp, layout, model="none", retrival_method="recent_k",
                         K=0, actor=r) for r in roles]  # fmt: skip
    team = AgentGroup(*agents)
    team.reset()
    env.reset()
    problems, shared = check_layout(mdp, agents, env.state)
    plan = PLANS[layout][order]
    for agent in agents:
        agent.test_mode = True
        agent.test_ml_action = deque(plan.get(agent.actor, []))
        agent.order = order
    stats = make_statistics_dict(len(agents))
    stats.update(layout=layout, order=order, cooks=roles)
    r_total = 0
    for t in range(horizon):
        s_t = env.state
        mp = env.mdp.state_string(s_t).replace("ø", "o")
        a_t, params = team.joint_action(s_t)
        team.reset_dialogue()
        delivering = [k for k, a in enumerate(agents) if "deliver" in (a.current_ml_action or "")]
        _, reward, _, _ = env.step(a_t, params)
        r_total += reward
        if reward > 0:
            stats["total_order_finished"].append(order)
            agents[delivering[0] if delivering else 0].own_ml_actions.append(
                {"timestamp": t, "action": "deliver_soup()"}
            )
        stats["total_timestamp"].append(t)
        stats["total_score"] = r_total
        stats["total_action_list"] = [list(a.own_ml_actions) for a in agents]
        stats["content"].append(copy.deepcopy(combine_statistic_dicts(
            [a.turn_statistics_dict for a in agents], mp, reward)))  # fmt: skip
        if reward:
            break
    summary = team_summary(stats)
    return {
        "layout": layout,
        "order": order,
        "layout_problems": problems,
        "shared_counters": {f"{a}-{b}": n for (a, b), n in shared.items()},
        "per_cook_entries": len(stats["content"][0]["statistical_data"]["error"]),
        "left_in_plans": {a.actor: list(a.test_ml_action) for a in agents
                          if list(a.test_ml_action) not in ([], ["wait(1)"])},  # fmt: skip
        **{k: summary[k] for k in ("success", "subtasks_done", "subtasks_total", "ticks",
                                   "actions_per_cook")},  # fmt: skip
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="")
    ap.add_argument("--order", default="")
    args = ap.parse_args()
    failures = 0
    for layout, orders in PLANS.items():
        if args.layout and layout != args.layout:
            continue
        for order in orders:
            if args.order and order != args.order:
                continue
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):  # the game prints every tick
                row = run(layout, order)
            ok = row["success"] and not row["layout_problems"]
            failures += not ok
            print(json.dumps(row), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
