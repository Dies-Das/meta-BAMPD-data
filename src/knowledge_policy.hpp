#include "optimal_policy.hpp"
#include "subgraph.hpp"
using ankerl::unordered_dense::map;

void build_belief(const State &root, StateSet &nodes, EdgeSet &edges,
                  const ui max_depth) {
    if (root.sum() == max_depth - 1) {
        return;
    }
    if (nodes.find(root) != nodes.end()) {
        return;
    }
    nodes.emplace(root);
    State child = root;
    for (ui k = 0; k < root.data.size(); k++) {
        child[k] += 1;
        edges.emplace(Edge({root, child}));
        build_belief(child, nodes, edges, max_depth);
        child[k]-=1;
    }
}
struct KnowledgeGradientPolicy {
        map<State, PolicyItem, StateHash> policy;
        map<State, double, StateHash> actual_reward;
        ui number_of_arms;
        ui max_depth;
        ui k_g;

        PolicyItem &operator[](State index) { return policy[index]; }
        KnowledgeGradientPolicy(ui _number_of_arms, ui _max_depth, ui _k_g)
            : number_of_arms(_number_of_arms),
              max_depth(_max_depth),
              k_g(_k_g) {
            State initial_state;
            initial_state.data.resize(2 * number_of_arms);
            create_policy(initial_state, max_depth);
        }
        PolicyItem create_policy(const State current, const ui max_depth) {
            auto policy_item = policy.find(current);
            // If we already computed the PolicyItem, just return it
            if (policy_item != policy.end()) {
                return policy_item->second;
            }

            auto probabilities = get_probabilities(current);
            PolicyItem item;

            // if we're at maximum depth, our reward is just the winning
            // probability of our action
            if (current.sum() == max_depth - 1) {
                item.action = argmax_winprobablity(probabilities);
                for (int k = 0; k < probabilities.size(); k++) {
                    item.expected_gains.push_back(probabilities[k]);
                }
                policy[current] = item;
                return item;
            }
            StateSet states;
            EdgeSet edges;
            ui belief_size = std::min(max_depth, current.sum() + k_g);
            build_belief(current, states, edges, belief_size);
            item.expected_gains =
                eval_basegraph(current, states, edges, max_depth);

            // Post-processing step for ties
            // 1. Find the maximum reward among arms
            double max_reward = std::numeric_limits<double>::lowest();
            for (int k = 0; k < (int)item.expected_gains.size(); k++) {
                if (item.expected_gains[k] > max_reward) {
                    max_reward = item.expected_gains[k];
                }
            }

            // 2. Identify all arms that are within 1e-7 of this max gain
            std::vector<ui> tied_arms;
            for (int k = 0; k < item.expected_gains.size(); k++) {
                if (std::fabs(item.expected_gains[k] - max_reward) < 1e-7) {
                    tied_arms.push_back(k);
                }
            }

            item.actions = tied_arms;
            item.action = tied_arms[0];
            policy[current] = item;
            for (int k = 0; k < current.data.size(); k++) {
                State child = current;
                child[k] += 1;
                create_policy(child, max_depth);
            }
            return item;
        }
        double value(State current) {
            auto item = policy[current];
            auto nr_of_actions = item.actions.size();
            auto probabilities = get_probabilities(current);

            if (current.sum() == max_depth - 1) {

                return item.reward();
            }
            double reward_current = 0.0;
            for (auto action : item.actions) {
                State winning_child = current;
                winning_child[2 * action] += 1;
                State losing_child = current;
                losing_child[2 * action + 1] += 1;

                auto reward_win = value(winning_child) + 1.0;

                auto reward_lose = value(losing_child);

                reward_current += probabilities[action] * reward_win +
                                  (1 - probabilities[action]) * reward_lose;
            }
            reward_current /= nr_of_actions;

            return reward_current;
        }
};
