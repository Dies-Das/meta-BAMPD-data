#ifndef META_POLICY_HPP
#define META_POLICY_HPP
#include "meta_graph.hpp"
#include <boost/unordered_map.hpp>

State find_root(const Belief &belief);

struct MetaPolicyItem;
struct Action{
  ui arm;
  bool is_computational;
  
  double net_gain;   
  double cost_of_action;
  
  std::array<MetaPolicyItem*,2> children;
  auto operator<=>(const Action& other) const {
    return net_gain<=>other.net_gain;
  }
};
struct MetaPolicyItem{
 std::vector<Action> actions;
  Belief belief;
  double gross_gain;
  double net_gain;
};
struct MetaPolicy{
  boost::unordered_map<Belief, MetaPolicyItem, BeliefHash> data;
  MetaGraph* meta;
  MetaPolicyItem* root;
  double base_cost;
  MetaPolicy(MetaGraph* _meta, const double _base_cost);

MetaPolicyItem& expand(const Belief& belief);
Action terminal_action(std::array<MetaNode*,2>& meta_children, double probability, ui arm);
Action computational_action(std::pair<std::array<MetaNode*,2>,ui>& computational_children, double probability, ui arm);
void write(const std::string& filename) const;
};
#endif