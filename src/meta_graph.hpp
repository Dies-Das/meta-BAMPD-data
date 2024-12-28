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

struct MetaGraph;
struct MetaNode
{
    State state;
    Belief belief;
    std::vector<double> gains;
    map<ui, std::array<MetaNode*,2>> terminal_children;
    map<ui, std::array<MetaNode*,2>> computational_children;
    
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
    MetaNode root;

    OptimalPolicy optimal;
    boost::unordered_map<Belief, MetaNode, BeliefHash> nodes;

    MetaGraph(ui _max_depth, ui _max_belief_size, ui _max_belief_depth, ui _nr_of_arms) : max_depth(_max_depth), max_belief_size(_max_belief_size), max_belief_depth(_max_belief_depth), nr_of_arms(_nr_of_arms), root(_nr_of_arms,this), optimal(_nr_of_arms, _max_depth)
    {
        std::cout << optimal[root.state] << std::endl;

        nodes.emplace(Belief{StateSet{root.state}, {}}, root);
        root.expand();
    }
};
#endif
