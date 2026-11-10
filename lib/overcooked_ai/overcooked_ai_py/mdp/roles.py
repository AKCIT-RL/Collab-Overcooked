"""Cook roles of a layout: who works which utensils, who reaches the dispensers,
who sees the recipe and who serves.

A layout lists one entry per player (player i is the digit i+1 of the grid):

    "agents": [
        {"role": "chef", "name": "Chef", "utensils": {"pot": "cook"},
         "dispensers": [], "recipe": True, "deliver": True},
        {"role": "runner", "name": "Runner", "utensils": {},
         "dispensers": ["dish_dispenser", "ingredient_dispenser"]},
        ...
    ]

``role`` is the cook's id (lower case, unique); ``name`` its display name
(default: the role capitalized); ``utensils`` maps a utensil kind to the operation
the cook may start there; ``dispensers`` are the dispensers the cook reaches;
``recipe`` (default False) gives the cook the recipe; ``deliver`` (default: same as
``recipe``) lets it serve; ``prompts`` names the game's prompt files the cook reads
(``chef`` or ``assistant``; default ``chef`` for the recipe holder, else
``assistant``).

The two-player layouts of the original game have no ``agents`` list but
``utensil_agent0`` / ``utensil_agent1``: they load as a Chef (player 0: the recipe,
serving, its utensils) and an Assistant (player 1: its utensils and both
dispensers), the roles the game always had.
"""

DISPENSERS = ("dish_dispenser", "ingredient_dispenser")
PROMPT_ROLES = ("chef", "assistant")


def legacy_agents(utensil_agent0, utensil_agent1):
    """The original game's Chef and Assistant, from utensil_agent0/1."""
    return [
        {
            "role": "chef",
            "name": "Chef",
            "utensils": dict(utensil_agent0 or {}),
            "dispensers": [],
            "recipe": True,
            "deliver": True,
            "prompts": "chef",
        },
        {
            "role": "assistant",
            "name": "Assistant",
            "utensils": dict(utensil_agent1 or {}),
            "dispensers": list(DISPENSERS),
            "recipe": False,
            "deliver": False,
            "prompts": "assistant",
        },
    ]


def normalize_agents(agents, num_players, utensils=None):
    """Validate a layout's ``agents`` list and fill the defaults."""
    if len(agents) != num_players:
        raise ValueError(
            f"layout lists {len(agents)} agents for {num_players} players in the grid"
        )
    out, seen = [], set()
    for i, spec in enumerate(agents):
        role = str(spec["role"]).strip().lower()
        if not role or role in seen or role in ("all", "everyone"):
            raise ValueError(f"agent {i}: role {role!r} is empty, reserved or repeated")
        seen.add(role)
        recipe = bool(spec.get("recipe", False))
        entry = {
            "role": role,
            "name": str(spec.get("name") or role.replace("_", " ").title()),
            "utensils": dict(spec.get("utensils") or {}),
            "dispensers": list(spec.get("dispensers") or []),
            "recipe": recipe,
            "deliver": bool(spec.get("deliver", recipe)),
            "prompts": str(spec.get("prompts") or ("chef" if recipe else "assistant")),
        }
        for kind in entry["utensils"]:
            if utensils is not None and kind not in utensils:
                raise ValueError(f"agent {role}: unknown utensil kind {kind!r}")
        for d in entry["dispensers"]:
            if d not in DISPENSERS:
                raise ValueError(f"agent {role}: unknown dispenser {d!r}")
        if entry["prompts"] not in PROMPT_ROLES:
            raise ValueError(f"agent {role}: prompts must be one of {PROMPT_ROLES}")
        out.append(entry)
    return out


def layout_agents(params, num_players):
    """(agents, legacy) for a layout's parameters (the dict of a .layout file)."""
    if params.get("agents") is not None:
        return normalize_agents(params["agents"], num_players, params.get("utensils")), False
    if num_players != 2:
        raise ValueError(
            f"a layout with {num_players} players needs an 'agents' list (roles per player)"
        )
    return legacy_agents(params.get("utensil_agent0"), params.get("utensil_agent1")), True
