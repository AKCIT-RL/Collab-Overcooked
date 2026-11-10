"""Prompt texts for kitchens with more than two cooks (layouts with an ``agents`` list).

The two-player layouts keep the game's own prompt files (``prompts/gpt``) untouched;
these texts take their place only when a layout gives more than two cooks, whose
roles, utensils and dispensers come from the layout (``mdp.agents_spec``).
"""

OPERATIONS = {
    "cook": ("pot", "To start cooking in the pot"),
    "cut": ("chopping_board", "To start cutting in the chopping_board"),
    "bake": ("oven", "To start baking in the oven"),
    "stir": ("blender", "To start blending in the blender"),
}
EVERYONE = ("everyone", "all", "team")


def describe(spec):
    """'Runner (reaches dish_dispenser, ingredient_dispenser)' style summary."""
    parts = []
    if spec["utensils"]:
        parts.append("works " + ", ".join(spec["utensils"]))
    if spec["dispensers"]:
        parts.append("reaches " + ", ".join(spec["dispensers"]))
    if spec["recipe"]:
        parts.append("has the recipe")
    if spec["deliver"]:
        parts.append("serves")
    return f"{spec['name']} ({'; '.join(parts) or 'carries items'})"


def team_line(specs):
    return "; ".join(describe(s) for s in specs)


def environment_rule(spec, specs):
    holders = [s["name"] for s in specs if "ingredient_dispenser" in s["dispensers"]]
    servers = [s["name"] for s in specs if s["deliver"]]
    return (
        "Suppose you are a {role} proficient in an adapted version of the Overcooked game. "
        f"Your goal is to cooperate with your {len(specs) - 1} teammates, who are also "
        "large language model agents, to complete a single order. \n"
        "{has_recipe}\n"
        "Due to the numerous settings that differ from the original Overcooked game, you "
        "must strictly adhere to the following game rules:\n\n"
        "**Game Rules:**\n\n"
        f"- The Overcooked_AI game has {len(specs)} players in separate areas of the "
        f"kitchen who work together to complete an order in the shortest time: "
        f"{team_line(specs)}.\n"
        "- To finish one order, your team needs to follow these steps:\n"
        "  1. Pick raw ingredients from the ingredient dispenser.\n"
        "  2. Place the ingredients in the correct utensil according to the recipe and "
        "initiate cooking using the appropriate action.\n"
        "  3. Your team may need to repeat Step 2 using several utensils according to "
        "the recipe. After cooking, you need to pick up the cooked ingredients and "
        "decide whether to move somewhere else.\n"
        "  4. Check if you need a dish to hold the cooked food. If so, the cook who "
        "serves must hold a dish first and then fill it with food from the utensil. "
        "Otherwise, it can directly pick up the cooked food from the utensil.\n"
        "  5. Deliver the food to the serving location immediately.\n\n"
        "*You cannot complete all of these steps alone. You need to think about what "
        "you can do in the situation and what you need your teammates to help you "
        "with.*\n"
        "- {workflow}\n\n"
        "- The recipe contains all the steps necessary to complete the order. Every "
        "choice you make must be based on the recipe.\n"
        "- You only need to complete one order, so focus solely on the progress of that "
        "dish.\n"
        f"- Only {' and '.join(holders) or 'nobody'} can pick up ingredients from the "
        "ingredient dispenser, which has an unlimited supply. Only "
        f"{' and '.join(servers) or 'nobody'} can deliver the order.\n"
        "- The utensil is a stationary unit that cannot be moved.\n"
        "- After placing an ingredient into a utensil, you need to use the correct action "
        "to start cooking.\n"
        "- Each counter between two areas is shared by the cooks of those two areas only: "
        "an item on it can be picked up by either of them. To pass an item to a cook "
        "whose area does not touch yours, hand it to a cook in between first.\n"
        "- If you wish to place something in another position, first check if you are "
        'holding it by verifying through "<{role}> holds XXX". Each player can only '
        "pick up one item at a time.\n"
        "- Players can only pass items by placing them on the counter; there is no "
        "direct \"pass\" action. To pass an item to a teammate, pick it up, then use "
        "`place_obj_on_counter(teammate)` with the teammate's name (it uses a counter "
        "you share with that teammate), and tell that teammate to pick it up."
    )


def workflow(spec):
    if spec["recipe"]:
        return (
            f"- The usual workflow for the {spec['name']} is:\n"
            "  1. Read the cooking process from your recipe. All of your decisions must be "
            "strictly guided by the recipe and should not lead to unfounded behavior.\n"
            "  2. Tell each teammate what to do with the ingredients and utensils it "
            "reaches, according to the recipe, and do your part in parallel.\n"
            "  3. Work in parallel with your teammates to finish the order in the shortest "
            "time possible. If you have nothing to do, you can wait.\n"
            "  4. Serve the dish (optional). If the recipe specifies that the dish needs to "
            "be served on a plate, use `fill_dish_with_food(utensil_name)` first; "
            "otherwise, just pick up the food from the utensil.\n"
            "  5. Use `deliver_soup()`."
        )
    return (
        f"The usual workflow for the {spec['name']} is:  \n"
        "- 1. Ask the cook who has the recipe for guidance, since you do not have the "
        "recipe.  \n"
        "- 2. Follow the instructions you receive unless they are incorrect. For "
        "example, if you are asked to use a utensil that is not in your area, refuse "
        "and say so. "
    )


def _op(name):
    kind, doc = OPERATIONS[name]
    return (
        f"def {name}({kind}_name):\n"
        f'    """ {doc} """\n'
        f"    if has_food({kind}_name):\n"
        "        return\n\n"
    )


def skill(spec, specs):
    """The skill block of the game's prompt, for this cook's operations."""
    text = (
        "**Skill**: \n"
        "In this game, you can ONLY perform the following allowed actions. Do not "
        "attempt to use any other actions that are not listed here. If there are "
        "parameters (like steps, utensil), you need to fill them without using quotation "
        "marks, angle brackets, etc. There are 2 kinds of actions: 'operation actions' and "
        "'requests'. If you want a teammate to perform a certain operation action, you "
        "need to generate a 'request' action with the operation action as an argument. \n"
        "**'Operation actions'**: \n"
        "def pickup(obj, place):\n"
        '    if object_in_hand() == "nothing": \n'
        '       if place in utensil_list or place == "counter" or place == '
        '"dish_dispenser" or place == "ingredient_dispenser":\n'
        "            if object_in_place(obj, place):\n"
        "                return\n\n"
    )
    for op in spec["utensils"].values():
        if op in OPERATIONS:
            text += _op(op)
    text += (
        "def place_obj_on_counter(teammate):\n"
        "    # teammate: the name of the teammate the item is for; the item goes on a\n"
        "    # counter you share with it (without a name: the nearest shared counter)\n"
        '    if object_in_hand() != "nothing":\n'
        "        return\n\n"
        "def put_obj_in_utensil(utensil):\n"
        '    if object_in_hand() != "nothing":\n'
        "        if utensil in utensil_list:\n"
        "            return\n\n"
        "def fill_dish_with_food(utensil):\n"
        '    if object_in_hand() == "dish":\n'
        "        if utensil_food_ready() or utensil_started_cooking():\n"
        "            return\n\n"
    )
    if spec["deliver"]:
        text += (
            "def deliver_soup():\n"
            '    if object_in_hand() == "soup":\n'
            "        return\n\n"
        )
    text += (
        "def wait(num):\n"
        "    # wait positive num timesteps\n"
        "    if isinstance(num, int) and 0 < num <= 20:\n"
        "        return\n"
        "**'Cooperative action'**: \n"
        '""" \n'
        "Ask a teammate to do an operation action. Each request can only contain one "
        "action with quotation marks, so if you need a teammate to perform multiple "
        "actions, you need to generate multiple requests. The detailed conversation "
        "content you want to convey is contained in '{role} says'. \n"
        "Example: request('pickup(potato, ingredient_dispenser)'); "
        "request('place_obj_on_counter(Chef)');... \n"
        '"""\n'
        "def request(operation_action):\n"
        "    teammate.plan_list.append(operation_action)\n"
        "    send_message(content=self.say, to=teammate)\n"
        "    return\n"
        "```"
    )
    if spec["recipe"]:
        for mate in specs:
            if mate is spec:
                continue
            ops = [op for op in mate["utensils"].values() if op in OPERATIONS]
            names = ["pickup(obj, place)"]
            names += [f"{op}({OPERATIONS[op][0]}_name)" for op in ops]
            names += ["place_obj_on_counter(teammate)", "put_obj_in_utensil(utensil)",
                      "fill_dish_with_food(utensil)", "wait(num)"]  # fmt: skip
            text += (
                f"\n\n**{mate['name']}'s Skill**: {mate['name']} can ONLY perform the "
                "following allowed actions"
                + (
                    f" and reaches {', '.join(mate['dispensers'])}"
                    if mate["dispensers"]
                    else ""
                )
                + ". If you need it to perform actions, you can use the actions below.\n"
                + "".join(f"def {n}:\n    pass\n" for n in names)
            )
    return text


def communication_note(spec, specs):
    names = ", ".join(s["name"] for s in specs if s is not spec)
    return (
        f"\n- Your teammates are {names}. Start what you say with the name of the "
        "teammate it is for and a colon (e.g. 'Runner: please bring an egg to the "
        "counter'), or with 'Everyone:' to tell all of them; a message without a name "
        "goes to everyone. The teammate you named answers you."
    )


def recipients(text, me, mates):
    """The teammates a message is for: named at its start ('Runner: ...', 'Runner and
    Baker: ...', '@Runner ...', 'To Runner, ...'), else all of them."""
    head = text.strip().lstrip("\"'").strip()
    low = head.lower()
    for prefix in ("to ", "@"):
        if low.startswith(prefix):
            head, low = head[len(prefix):], low[len(prefix):]
    cut = min([i for i in (low.find(":"), low.find(",")) if i > 0] or [len(low)])
    named = low[:cut].replace(" and ", ",").replace("&", ",").split(",")
    named = [n.strip().lstrip("@") for n in named if n.strip()]
    if any(n in EVERYONE for n in named):
        return list(mates)
    chosen = [
        m for m in mates
        if any(n in (m.name.lower(), m.actor.lower()) for n in named)
    ]  # fmt: skip
    return chosen or list(mates)
