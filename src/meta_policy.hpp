#include "meta_graph.hpp"
#include <boost/unordered_map.hpp>
struct MetaPolicyItem{
 unsigned int action;
 bool is_computational;
 double cost_of_node;
 double gross_gain;
 double net_gain;
 std::pair<MetaPolicyItem*,MetaPolicyItem*> children;
};
struct MetaPolicy{
  boost::unordered_map<Belief, MetaPolicyItem, BeliefHash> data;
  MetaGraph* meta;
  double base_cost;
  MetaPolicy(MetaGraph* _meta, const double _base_cost);

MetaPolicyItem expand(const Belief& belief);
  
};
