#ifndef OPTIMAL_POLICY_HPP
#define OPTIMAL_POLICY_HPP
#include "bandit.hpp"
#include <algorithm>
#include <cmath>
using ankerl::unordered_dense::map;


struct OptimalPolicy
{
    map<State, PolicyItem, StateHash> policy;
    ui number_of_arms;
    ui max_depth;
    PolicyItem& operator[](State index){
        return policy[index];
    }
    OptimalPolicy(ui _number_of_arms, ui _max_depth) : number_of_arms(_number_of_arms), max_depth(_max_depth)
    {
        State initial_state;
        initial_state.data.resize(2 * number_of_arms);
        create_policy(initial_state, max_depth);
    }
    PolicyItem create_policy(const State current,const  ui max_depth)
    {
        auto policy_item = policy.find(current);
        // If we already computed the PolicyItem, just return it
        if (policy_item != policy.end())
        {
            return policy_item->second;
        }

        auto probabilities = get_probabilities(current);
        PolicyItem item;
        
        // if we're at maximum depth, our reward is just the winning probability of our action
        if (current.sum() == max_depth - 1)
        {
            item.action = argmax_winprobablity(probabilities);
            for (int k = 0;  k < probabilities.size(); k++)
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
        // 1. Find the maximum reward among arms
        double max_reward = std::numeric_limits<double>::lowest();
        for (int k = 0;  k < (int)item.expeced_gains.size(); k++)
        {
            if (item.expeced_gains[ k] > max_reward)
            {
                max_reward = item.expeced_gains[ k];
            }
        }

        // 2. Identify all arms that are within 1e-7 of this max gain
        std::vector<ui> tied_arms;
        for (int k = 0;  k < item.expeced_gains.size(); k++)
        {
            if (std::fabs(item.expeced_gains[ k] - max_reward) < 1e-7)
            {
                tied_arms.push_back(k);
            }
        }

        item.actions = tied_arms;
        item.action = tied_arms[0];
        policy[current] = item;
        return item;
    }
};

#endif