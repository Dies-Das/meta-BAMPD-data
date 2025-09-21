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
  State initial_state;
  initial_state.data.resize(2 * meta->nr_of_arms);
  Belief initial_belief = Belief{StateSet{initial_state}, {}};
  expand(initial_belief);
  this->root = &data[initial_belief];
}
MetaPolicyItem &MetaPolicy::expand(const Belief &belief)
{
  // check if we already did the node
  auto node_iterator = this->data.find(belief);
  if (node_iterator != this->data.end())
  {
    return node_iterator->second;
  }

  MetaPolicyItem result;
  result.belief = belief;
  result.gross_gain = 0;
  result.net_gain = 0;
  MetaNode &current_meta = meta->nodes.at(belief);

  auto current_root = find_root(belief);

  auto probabilities = get_probabilities(current_root);
  // check if we're at the end of the time horizon
  if (current_root.sum() == meta->max_depth - 1)
  {
    auto max = std::max_element(probabilities.begin(), probabilities.end());
    Action action;
    action.arm = std::distance(probabilities.begin(), max);
    action.cost_of_action = 0;
    action.is_computational = false;
    action.net_gain = *max;
    result.gross_gain = *max;
    result.net_gain = *max;
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
    auto terminal_average = 0.0;
    for (auto &action : terminal)
    {
      terminal_average += action.net_gain;
    }
    terminal_average /= terminal.size();
    auto max_computational = std::max_element(computational.begin(), computational.end());
    double max_value = 0;
    int terminal_maximizers = 0;
    int computational_maximizers = 0;

    if (max_computational == computational.end())
    {
      max_value = terminal_average;
    }
    else
    {
      max_value = std::max(terminal_average, max_computational->net_gain);
    }
    if (std::abs(terminal_average - max_value) < 1e-7)
    {
      for (auto &action : terminal)
      {

        terminal_maximizers++;
        result.actions.push_back(action);
      }
    }
    for (auto &action : computational)
    {
      if (std::abs(action.net_gain - max_value) < 1e-7)
      {
        computational_maximizers++;
        result.actions.push_back(action);
      }
    }
    if (terminal_maximizers > 0 && computational_maximizers == 0)
    {
      result.actions = terminal;
    }
    for (auto &action : result.actions)
    {
      result.gross_gain += probabilities[action.arm] * (action.children[0]->gross_gain + 1);
      result.gross_gain += (1 - probabilities[action.arm]) * action.children[1]->gross_gain;
      result.net_gain += action.net_gain;
    }
    result.net_gain /= result.actions.size();
    result.gross_gain /= result.actions.size();
  }

  auto [new_item_it, new_item_found] = this->data.emplace(belief, result);
  return new_item_it->second;
}

Action MetaPolicy::terminal_action(std::array<MetaNode *, 2> &meta_children, double probability, ui arm)
{
  Action action;
  action.arm = arm;
  auto &winning_policy = expand(meta_children[0]->belief);
  auto &losing_policy = expand(meta_children[1]->belief);
  action.children = {&winning_policy, &losing_policy};
  action.computations = 0;
  action.cost_of_action = 0;
  action.is_computational = false;
  action.net_gain = probability * (winning_policy.net_gain + 1);
  action.net_gain += (1 - probability) * (losing_policy.net_gain);
  return action;
}
Action MetaPolicy::computational_action(std::pair<std::array<MetaNode *, 2>, ui> &computational_children, double probability, ui arm)
{
  Action action;
  action.arm = arm;
  auto &winning_policy = expand(computational_children.first[0]->belief);
  auto &losing_policy = expand(computational_children.first[1]->belief);
  action.children = {&winning_policy, &losing_policy};
  action.computations = computational_children.second;
  action.cost_of_action = computational_children.second * base_cost;
  action.is_computational = true;
  action.net_gain = probability * (winning_policy.net_gain + 1);
  action.net_gain += (1 - probability) * (losing_policy.net_gain);
  action.net_gain -= action.cost_of_action;
  return action;
}
