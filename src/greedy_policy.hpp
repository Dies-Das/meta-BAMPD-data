#ifndef GREEDY_POLICY_HPP
#define GREEDY_POLICY_HPP
#include "bandit.hpp"
#include <algorithm>
#include <cmath>
using ankerl::unordered_dense::map;


struct GreedyPolicy
{
    map<State, PolicyItem, StateHash> policy;
    ui number_of_arms;
    ui max_depth;
    GreedyPolicy(ui _number_of_arms, ui _max_depth) : number_of_arms(_number_of_arms), max_depth(_max_depth)
    {
        State initial_state;
        initial_state.data.resize(2 * number_of_arms);
        create_policy(initial_state, max_depth);
    }
    PolicyItem create_policy(State current, ui max_depth)
    {
        auto policy_item = policy.find(current);
        // If we already computed the PolicyItem, just return it
        if (policy_item != policy.end())
        {
            return policy_item->second;
        }

        auto probabilities = get_probabilities(current);
        PolicyItem item;
        // We take the greedy action
        item.action = argmax_winprobablity(probabilities);
        // if we're at maximum depth, our reward is just the winning probability of our action
        if (current.sum() == max_depth - 1)
        {
            for (int k = 0; 2 * k < probabilities.size(); k++)
            {
                item.expeced_gains.push_back(probabilities[k]);
            }
            policy[current] = item;
            return item;
        }
        for (int k = 0; 2 * k < current.data.size(); k++)
        {
            State winning_child = current;
            winning_child[2 * k] += 1;
            State losing_child = current;
            losing_child[2 * k + 1] += 1;

            auto child_item_win = create_policy(winning_child, max_depth);
            double reward_win = child_item_win.reward() + 1.0;

            auto child_item_lose = create_policy(losing_child, max_depth);
            double reward_lose = child_item_lose.reward();

            double reward = probabilities[ k] * reward_win +(1-probabilities[ k ]) * reward_lose;
            item.expeced_gains.push_back(reward);
        }

        // Post-processing step for ties
        // 1. Find the maximum probability among arms
        double max_prob = std::numeric_limits<double>::lowest();
        for (int k = 0;  k < (int)probabilities.size(); k++)
        {
            if (probabilities[ k] > max_prob)
            {
                max_prob = probabilities[ k];
            }
        }

        // 2. Identify all arms that are within 1e-7 of this max probability
        std::vector<int> tied_arms;
        for (int k = 0;  k < probabilities.size(); k++)
        {
            if (std::fabs(probabilities[ k] - max_prob) < 1e-7)
            {
                tied_arms.push_back(k);
            }
        }

        // 3. Compute the average expected gain for these tied arms
        if (!tied_arms.empty())
        {
            double sum_gains = 0.0;
            for (auto idx : tied_arms)
            {
                sum_gains += item.expeced_gains[idx];
            }
            double avg_gain = sum_gains / tied_arms.size();

            // 4. Set all tied arms' expected gains to this average
            for (auto idx : tied_arms)
            {
                item.expeced_gains[idx] = avg_gain;
            }
        }

        policy[current] = item;
        return item;
    }
};

#endif