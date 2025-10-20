#ifndef META_HPP
#define META_HPP
#include "subgraph.hpp"
#include "optimal_policy.hpp"
#include "greedy_policy.hpp"
#include <variant>
#include <boost/unordered_map.hpp>
struct MStateResult
{
    bool is_m_state;
    std::size_t index;
    std::vector<std::size_t> indices; 
};
MStateResult check_m_state(
    const std::vector<double> &A,
    const std::vector<double> &B);
enum class BoundingCondition{
  NONE = 0,
  DEPTH,
  SIZE,
  COMPUTATIONS
};
struct BoundingParameters{
  BoundingCondition bounding_type;
  int max_belief_depth;
  int max_belief_size;
  int max_computations;
  BoundingParameters(BoundingCondition condition, int bound){
    bounding_type = condition;
    switch (condition)
    {
    case BoundingCondition::DEPTH:
        max_belief_depth = bound;
        break;
    case BoundingCondition::SIZE:
        max_belief_size = bound;
        break;
    case BoundingCondition::COMPUTATIONS:
        max_computations = bound;
        break;
    
    default:
        break;
    }
  }
};
struct MetaGraph;
struct MetaNode
{
    State state;
    Belief belief;
    std::vector<double> gains;
    map<ui, std::array<MetaNode*,2>> terminal_children;
    map<ui, std::vector<std::pair<std::array<MetaNode*,2>, ui>>> computational_children;
    
    MetaGraph *meta;
    bool expanded = false;
    MetaNode(State _state, Belief _belief, MetaGraph *_meta);
    MetaNode(ui nr_of_arms, MetaGraph*_meta);
    MetaNode() = default;
    MetaNode(const MetaNode& node) = default;
    void expand();
    void add_child(ui arm, const Belief& new_belief, bool terminal);
    void computational_expansion(ui terminal_action, ui candidate);
    set<State, StateHash> get_candidates( ui candidate);
};

struct MetaGraph
{
    ui max_depth;
    ui max_belief_size;
    ui max_belief_depth;
    ui nr_of_arms;
    MetaNode* root;

    OptimalPolicy optimal;
    GreedyPolicy greedy;
    boost::unordered_map<Belief, MetaNode, BeliefHash> nodes;
    BoundingParameters bounds;
    MetaGraph(BoundingParameters _bounds, ui _max_depth, ui _max_belief_size, ui _max_belief_depth, ui _nr_of_arms) : max_depth(_max_depth), max_belief_size(_max_belief_size), max_belief_depth(_max_belief_depth), nr_of_arms(_nr_of_arms),  optimal(_nr_of_arms, _max_depth), greedy(_nr_of_arms,_max_depth),bounds(_bounds)
    {
        State initial_state;
        initial_state.data.resize(2 * _nr_of_arms);
        auto [root_it, inserted_root] = nodes.emplace(Belief{StateSet{initial_state}, {}}, MetaNode(_nr_of_arms, this));
        root = &(root_it->second);
        root->expand();
    }
    MetaGraph(State& _initial_state, BoundingParameters _bounds, ui _max_depth, ui _max_belief_size, ui _max_belief_depth, ui _nr_of_arms) : max_depth(_max_depth), max_belief_size(_max_belief_size), max_belief_depth(_max_belief_depth), nr_of_arms(_nr_of_arms),  optimal(_nr_of_arms, _max_depth), greedy(_nr_of_arms,_max_depth),bounds(_bounds)
    {
        auto [root_it, inserted_root] = nodes.emplace(Belief{StateSet{_initial_state}, {}}, MetaNode(_initial_state,Belief{StateSet{_initial_state}, {}}, this));
        root = &(root_it->second);
        root->expand();
    }
    bool stop_expansion(Belief& current_belief, Belief& new_belief);
    bool stop_expansion(State& root_state, State& current_state);
};
#endif
