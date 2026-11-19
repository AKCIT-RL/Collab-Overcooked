"""Subtasks of an order, from the layout's recipes: partial credit without reference
trajectories (which exist only for the two-cook kitchen).

An order's subtasks are the products its recipe makes, leaves first (e.g.
``potato_slices`` on a chopping board, then ``boiled_potato_slices`` in a pot...),
then its delivery. A product counts as made once some cook picked it up from a
utensil (or, for the order itself, filled a dish with it or delivered it); making a
product also counts every product it was made from.
"""

import re


def recipe_steps(recipes, order):
    """[(product, utensil kind, ingredients)] that make ``order``, leaves first."""
    makers = {}
    for kind, dishes in recipes.items():
        if kind == "none":  # raw items served as they are
            continue
        for product, spec in dishes.items():
            makers.setdefault(product, (kind, list(spec["recipe"])))
    steps, seen = [], set()

    def visit(item):
        if item in seen or item not in makers:
            return
        seen.add(item)
        kind, ingredients = makers[item]
        for ingredient in ingredients:
            visit(ingredient)
        steps.append((item, kind, ingredients))

    visit(order)
    return steps


def subtasks(recipes, order):
    """Names of the order's subtasks: 'make <product>' per step, then 'deliver <order>'."""
    return [f"make {p}" for p, _, _ in recipe_steps(recipes, order)] + [f"deliver {order}"]


def completed_subtasks(recipes, order, actions, delivered=False):
    """The subtasks done, given the completed actions (any cook, game call strings
    such as ``pickup(boiled_egg, pot0)``) and whether the order was delivered."""
    calls = [re.sub(r"\s+", "", str(a)).lower() for a in actions]
    steps = recipe_steps(recipes, order)
    made = set()
    for product, _, _ in steps:
        picked = any(c.startswith(f"pickup({product.lower()},") for c in calls)
        served = product == order and (
            delivered or any(c.startswith("fill_dish_with_food(") for c in calls)
        )
        if picked or served:
            made.add(product)
    # a product was made from its ingredients: count those too
    for product, _, ingredients in reversed(steps):
        if product in made:
            made.update(i for i in ingredients if any(i == p for p, _, _ in steps))
    done = [f"make {p}" for p, _, _ in steps if p in made]
    if delivered:
        done.append(f"deliver {order}")
    return done
