"""The fixed_orders flag: a queue of K orders drains and the episode ends.

    cd lib/overcooked_ai && python -m unittest overcooked_ai_py.mdp.fixed_orders_test
"""

import unittest

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld

ORDERS = ["boiled_egg", "baked_potato_slices"]


def finished(mdp, order):
    """The finished food of ``order`` as the Chef holds it to deliver."""
    category = "with_dish" if mdp.need_dish.get(order) else "finished"
    return ObjectState(order, category, (0, 0))


def deliver(mdp, state, order):
    player = state.players[0]
    soup = finished(mdp, order)
    player.set_object(soup)
    return mdp.deliver_soup(state, player, soup)


class FixedOrdersTest(unittest.TestCase):
    def env(self, **params):
        mdp = OvercookedGridworld.from_layout_name("new_env", **params)
        env = OvercookedEnv(mdp, horizon=500)
        env.reset()
        return mdp, env

    def test_queue_drains_and_ends(self):
        mdp, env = self.env(start_order_list=list(ORDERS), fixed_orders=True)
        self.assertTrue(mdp.fixed_orders)
        state = env.state
        self.assertEqual(state.order_list, ORDERS)
        state, reward = deliver(mdp, state, ORDERS[0])
        self.assertGreater(reward, 0)
        self.assertEqual(state.order_list, ORDERS[1:])
        self.assertFalse(mdp.is_terminal(state))
        state, reward = deliver(mdp, state, ORDERS[1])
        self.assertGreater(reward, 0)
        self.assertEqual(state.order_list, [])
        self.assertTrue(mdp.is_terminal(state))

    def test_only_the_first_order_counts(self):
        mdp, env = self.env(start_order_list=list(ORDERS), fixed_orders=True)
        state, reward = deliver(mdp, env.state, ORDERS[1])
        self.assertEqual(reward, 0)
        self.assertEqual(state.order_list, ORDERS)

    def test_set_after_construction(self):
        # the way the game's main.py and the akc_harness adapter set the orders
        mdp = OvercookedGridworld.from_layout_name("new_env")
        mdp.start_order_list = list(ORDERS)
        mdp.fixed_orders = True
        env = OvercookedEnv(mdp, horizon=500)
        env.reset()
        state, _ = deliver(mdp, env.state, ORDERS[0])
        state, _ = deliver(mdp, state, ORDERS[1])
        self.assertTrue(mdp.is_terminal(state))
        self.assertTrue(env.is_done() or mdp.is_terminal(state))

    def test_default_refills(self):
        mdp = OvercookedGridworld.from_layout_name("new_env")
        self.assertFalse(mdp.fixed_orders)
        mdp.start_order_list = [ORDERS[0]]
        env = OvercookedEnv(mdp, horizon=500)
        env.reset()
        state, reward = deliver(mdp, env.state, ORDERS[0])
        self.assertGreater(reward, 0)
        self.assertEqual(len(state.order_list), 3)  # topped up from order_probability
        self.assertFalse(mdp.is_terminal(state))


if __name__ == "__main__":
    unittest.main()
