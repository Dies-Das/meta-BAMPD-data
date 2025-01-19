#include "meta_policy.hpp"
State find_root(const Belief &belief)
{
  ui depth = UINT32_MAX;
  State root;
  for (const auto &state : belief.first)
  {
    if (state.sum() < depth)
    {
      depth = state.sum();
      root = state;
    }
  }
  return root;
}
MetaPolicy::MetaPolicy(MetaGraph *_meta, const double _base_cost) : meta(_meta), base_cost(_base_cost)
{
}
MetaPolicyItem &MetaPolicy::expand(const Belief &belief)
{
  // check if we already did the node
  auto node_iterator = this->data.find(belief);
  if (node_iterator == this->data.end())
  {
    return node_iterator->second;
  }

  MetaPolicyItem result;
  MetaNode &current_meta = meta->nodes.at(belief);

  auto current_root = find_root(belief);
  auto probabilities = get_probabilities(current_root);
  // check if we're at the end of the time horizon
  if (current_root.sum() == meta->max_depth)
  {
    auto max = std::max_element(probabilities.begin(), probabilities.end());
    Action action;
    action.arm = std::distance(probabilities.begin(), max);
    action.cost_of_action = 0;
    action.is_computational = false;
    action.net_gain = *max;
    result.actions.push_back(action);
  }

  // if we're not at the end, we need to find the optimal action considering computational cost
  else
  {

    std::vector<Action> terminal;

    std::vector<Action> computational;
    for (auto &child : current_meta.terminal_children)
    {
      terminal.push_back(terminal_action(child.second, probabilities[child.first], child.first));
    }
    for (auto &arm_children : current_meta.computational_children)
    {
      for (auto &child : arm_children.second)
      {
        computational.push_back(computational_action(child, probabilities[arm_children.first], arm_children.first));
      }
    }
    auto max_terminal = std::max_element(terminal.begin(), terminal.end());
    auto max_computational = std::max_element(computational.begin(), computational.end());
    double max_value = 0;
    if (max_terminal == terminal.end())
    {
      max_value = max_computational->net_gain;
    }
    else if (max_computational == computational.end())
    {
      max_value = max_terminal->net_gain;
    }
    else
    {
      max_value = std::max(*max_terminal, *max_computational).net_gain;
    }
    for(auto& action : terminal){
      if(action.net_gain==max_value){
        result.actions.push_back(action);
      }
    }
    for(auto& action : computational){
      if(action.net_gain==max_value){
        result.actions.push_back(action);
      }
    }
  }
  auto [new_item_it, new_item_found] = this->data.emplace(belief,result);
  return new_item_it->second;
}

Action MetaPolicy::terminal_action(std::array<MetaNode *, 2> &meta_children, double probability, ui arm)
{
  Action action;
  action.arm = arm;
  auto &winning_policy = expand(meta_children[0]->belief);
  auto &losing_policy = expand(meta_children[1]->belief);
  action.children = {&winning_policy, &losing_policy};
  action.cost_of_action = 0;
  action.is_computational = false;
  action.net_gain = probability * (winning_policy.actions[0].net_gain + 1);
  action.net_gain += (1 - probability) * (losing_policy.actions[0].net_gain);
  return action;
}
Action MetaPolicy::computational_action(std::pair<std::array<MetaNode *, 2>, ui> &computational_children, double probability, ui arm)
{
  Action action;
  action.arm = arm;
  auto &winning_policy = expand(computational_children.first[0]->belief);
  auto &losing_policy = expand(computational_children.first[1]->belief);
  action.children = {&winning_policy, &losing_policy};
  action.cost_of_action = computational_children.second * base_cost;
  action.is_computational = true;
  action.net_gain = probability * (winning_policy.actions[0].net_gain + 1);
  action.net_gain += (1 - probability) * (losing_policy.actions[0].net_gain);
  action.net_gain -= action.cost_of_action;
  return action;
}