#ifndef GREEDY_POLICY_HPP
#define GREEDY_POLICY_HPP
#include "bandit.hpp"
#include <algorithm>
#include <cmath>
using ankerl::unordered_dense::map;

struct GreedyPolicyItem
{
    vector<double> expeced_gains;
    double reward;
    std::vector<ui> actions;
    

};
struct GreedyPolicy
{
    map<State, GreedyPolicyItem, StateHash> policy;
    ui number_of_arms;
    ui max_depth;
    GreedyPolicyItem& operator[](State index){
        return policy[index];
    }
    GreedyPolicy(ui _number_of_arms, ui _max_depth) : number_of_arms(_number_of_arms), max_depth(_max_depth)
    {
        State initial_state;
        initial_state.data.resize(2 * number_of_arms);
        create_policy(initial_state, max_depth);

    }
    GreedyPolicyItem& create_policy(const State current,const  ui max_depth)
    {
        auto policy_item = policy.find(current);
        // If we already computed the PolicyItem, just return it
        if (policy_item != policy.end())
        {
            return policy_item->second;
        }

        auto probabilities = get_probabilities(current);
        GreedyPolicyItem item;
        item.reward = 0;
        
        // if we're at maximum depth, our reward is just the winning probability of our action
        if (current.sum() == max_depth - 1)
        {

            for (int k = 0;  k < probabilities.size(); k++)
            {
                item.expeced_gains.push_back(probabilities[k]);
            }
            item.reward = *std::max_element(probabilities.begin(),probabilities.end());
            policy[current] = item;
            return policy[current];
        }
        
        auto max = std::max_element(probabilities.begin(),probabilities.end());
        for(int k=0; k<probabilities.size();k++){
            if(std::abs(*max-probabilities[k])<1e-7){
                item.actions.push_back(k);
            }
        }
        for (int k = 0; 2 * k < current.data.size(); k++)
        {
            State winning_child = current;
            winning_child[2 * k] += 1;
            State losing_child = current;
            losing_child[2 * k + 1] += 1;

            auto child_item_win = create_policy(winning_child, max_depth);
            double reward_win = child_item_win.reward + 1.0;

            auto child_item_lose = create_policy(losing_child, max_depth);
            double reward_lose = child_item_lose.reward;

            double reward = probabilities[ k] * reward_win +(1-probabilities[ k ]) * reward_lose;
            item.expeced_gains.push_back(reward);
        }
        for(auto index: item.actions){
            item.reward += item.expeced_gains[index];
        }
        item.reward /= item.actions.size();
        policy[current] = item;
        return policy[current];
    }
};

#endif