#ifndef SUBGRAPH_HPP
#define SUBGRAPH_HPP
#include <ankerl/unordered_dense.h>
#include "bandit.hpp"
#include <algorithm>
using ankerl::unordered_dense::set;
using StateSet = set<State, StateHash>;
using Edge = std::array<State, 2>;

struct StateSetHash
{
    using is_avalanching = void;

    auto operator()(StateSet const &x) const noexcept -> uint64_t
    {

        std::size_t h = 0;
        // Simple FNV-1a hash example:
        StateHash hasher;
        for (auto &d : x)
        {
            h ^= hasher(d);
            h *= 1099511628211ull; // FNV prime
        }
        return h;
    }
};
struct EdgeHash
{
    using is_avalanching = void;
    auto operator()(Edge const &x) const noexcept -> uint64_t
    {

        std::size_t h = 0;
        // Simple FNV-1a hash example:
        StateHash hasher;
        for (auto &d : x)
        {
            h ^= hasher(d);
            h *= 1099511628211ull; // FNV prime
        }
        return h;
    }
};

using EdgeSet = set<Edge, EdgeHash>;
using Belief = std::pair<StateSet, EdgeSet>;

struct EdgeSetHash
{
    using is_avalanching = void;

    auto operator()(EdgeSet const &x) const noexcept -> uint64_t
    {

        std::size_t h = 0;
        // Simple FNV-1a hash example:
        EdgeHash hasher;
        for (auto &d : x)
        {
            h ^= hasher(d);
            h *= 1099511628211ull; // FNV prime
        }
        return h;
    }
};
struct BeliefHash
{
    using is_avalanching = void;
    auto operator()(Belief const &x) const noexcept -> uint64_t
    {

        std::size_t h = 0;
        // Simple FNV-1a hash example:
        StateSetHash hasher;
        EdgeSetHash hasher2;

        h ^= hasher(x.first);
        h *= 1099511628211ull; // FNV prime
        h ^= hasher2(x.second);
        h *= 1099511628211ull; // FNV prime

        return h;
    }
};
// Compute the inherited belief at root given the prior belief (states, edges)
inline void expand_subgraph(const State &root, const StateSet &states, const EdgeSet &edges, StateSet &result_states, EdgeSet &result_edges, const ui max_depth)
{
    if (root.sum() == max_depth - 1)
    {
        return;
    }
    // go through all children
    for (int k = 0; k < root.data.size(); k++)
    {
        auto child = root;
        child[k] += 1;
        // if we find the child and root-child pair in our old belief, it will be in our new one
        if (states.find(child) != states.end() && edges.find({root, child}) != edges.end())
        {
            result_states.insert(child);
            result_edges.insert({root, child});
            // recursive call
            expand_subgraph(child, states, edges, result_states, result_edges, max_depth);
        }
    }
    return;
}

inline std::pair<StateSet, EdgeSet> subgraph(const State &root, const StateSet &states, const EdgeSet &edges, const ui max_depth)
{
    StateSet result_states;
    EdgeSet result_edges;

    // If the root node is not in the graph, return a set containing only the root
    if (!states.count(root))
    {
        result_states.insert(root);
        return {result_states, result_edges};
    }

    // Start expanding the subgraph
    result_states.insert(root);
    expand_subgraph(root, states, edges, result_states, result_edges, max_depth);

    return {result_states, result_edges};
};

// Get the gain estimates based on our belief. Will be needed in the Metagraph to check if we're in a M-State.
inline std::vector<double> eval_basegraph(const State &root, const StateSet &states, const EdgeSet &edges, const ui max_depth)
{
    double current_depth = root.sum();

    auto probabilities = get_probabilities(root);

    std::vector<double> result;
    if(current_depth == max_depth-1){
        return probabilities;
    }
    result.resize(root.data.size() / 2);
    // go through all children
    for (int k = 0; k < result.size(); k++)
    {
        auto winning_child = root;
        winning_child[2 * k] += 1;
        // if the child is in our belief, we need to continue traversing the belief and backpropagate the expected gains
        if (states.find(winning_child) != states.end() && edges.find({root, winning_child}) != edges.end())
        {
            auto [sub_graph_states, sub_graph_edges] = subgraph(winning_child, states, edges, max_depth);
            auto gains = eval_basegraph(winning_child, sub_graph_states, sub_graph_edges, max_depth);
            auto max = *std::max_element(gains.begin(), gains.end());
            result[k] += probabilities[k] * (max + 1.0);
            auto losing_child = root;
            losing_child[2 * k + 1] += 1;
            gains = eval_basegraph(losing_child, sub_graph_states, sub_graph_edges, max_depth);
            max = *std::max_element(gains.begin(), gains.end());
            result[k] += (1 - probabilities[k]) * (max);
        }
        // else, assume constant probabilities and estimate greedily
        else
        {
            double gain = 0.0;

            result[k] += ((max_depth - current_depth) * probabilities[k]);
        }
    }
    return result;
}

#endif