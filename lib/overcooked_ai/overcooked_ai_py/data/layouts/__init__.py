import os
from overcooked_ai_py.utils import load_dict_from_file

LAYOUTS_DIR = os.path.dirname(os.path.abspath(__file__))

def read_layout_dict(layout_name):
    """A layout's parameters. ``"extends": "<layout>"`` starts from another layout's
    parameters (recipes, utensils, ingredients...) and overrides the keys it sets."""
    params = load_dict_from_file(os.path.join(LAYOUTS_DIR, layout_name + ".layout"))
    parent = params.pop("extends", None)
    if parent is None:
        return params
    base = read_layout_dict(parent)
    base.update(params)
    if "agents" in params:  # the roles of the parent's players do not carry over
        base.pop("utensil_agent0", None)
        base.pop("utensil_agent1", None)
    return base


def layout_names():
    """Names of the layouts shipped in this folder."""
    return sorted(f[: -len(".layout")] for f in os.listdir(LAYOUTS_DIR) if f.endswith(".layout"))
