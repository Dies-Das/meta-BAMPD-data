#include "meta_graph.hpp"
bool operator<(const Belief &lhs, const Belief &rhs) {

  for (auto &edge : lhs.second) {
    if (rhs.second.find(edge) == rhs.second.end()) {

      return false;
    }
  }
 return true;
}
struct ExpansionCandidate {
  Belief belief;
  set<State, StateHash> expansion_nodes;
};

struct PairHash {
  using is_avalanching = void;
  auto operator()(std::pair<ui, ui> const &x) const noexcept -> uint64_t {

    std::size_t h = 0;
    // Simple FNV-1a hash example:

    h ^= ankerl::unordered_dense::detail::wyhash::hash(x.first);
    h *= 1099511628211ull; // FNV prime
    h ^= ankerl::unordered_dense::detail::wyhash::hash(x.second);
    h *= 1099511628211ull; // FNV prime

    return h;
  }
};
struct StateBeliefPairHash {
  using is_avalanching = void;
  auto operator()(std::pair<State, Belief> const &x) const noexcept
      -> uint64_t {

    std::size_t h = 0;
    // Simple FNV-1a hash example:

    h ^= StateHash{}(x.first);
    h *= 1099511628211ull; // FNV prime
    h ^= BeliefHash{}(x.second);
    h *= 1099511628211ull; // FNV prime

    return h;
  }
};

MetaNode::MetaNode(const State _state, const Belief _belief, MetaGraph *_meta)
    : state(_state), belief(_belief), meta(_meta) {
  std::cout << "Constructing MetaNode: state = " << state << "\n";

  gains = eval_basegraph(state, belief.first, belief.second, meta->max_depth);
}
MetaNode::MetaNode(ui _nr_of_arms, MetaGraph* _meta){
  state.data.resize(2*(int)_nr_of_arms);
  
  meta=_meta;
  belief = Belief{StateSet{state},EdgeSet{}};
  gains = eval_basegraph(state, belief.first, belief.second, meta->max_depth);
}
void MetaNode::expand() {
  std::cout << "Expanding state " << this->state << std::endl;
  if (state.sum() == meta->max_depth - 1) {
    return;
  }
  if (expanded) { // if we already did this MetaNode, there is nothing to do.
    return;
  }
  if (state == State{std::vector<ui>{2, 1, 0, 0}}) {
    std::cout << "let's see.." << std::endl;
  }

  expanded = true;
  // find the greedy actions corresponding to current belief and if we are an
  // M-state
  auto &optimal_rewards = meta->optimal[state].expeced_gains;
  auto m_result = check_m_state(gains, optimal_rewards);
  // if our arms are identical, we don't need to do computational expansions
  bool are_identical = true;
  for (int k = 0; k < meta->nr_of_arms; k++) {
    if (this->state[2 * k] != this->state[0] ||
        this->state[2 * k + 1] != this->state[1]) {
      are_identical = false;
      break;
    }
  }

  // Add child for pure termination if we're an m-state
  if (m_result.is_m_state) {
    add_child(m_result.index, belief, true);
  }
  // else we need to add termination for all greedy-optimal arms, but not doing
  // identical ones
  else {
    set<std::pair<ui, ui>, PairHash> checked_arm_values;
    for (auto choice : m_result.indices) {
      // if we already looked at an arm with identical wins/losses, we don't
      // need to expand
      std::pair<ui, ui> arm_value{state[2 * choice], state[2 * choice + 1]};
      auto found = checked_arm_values.find(arm_value);
      if (found != checked_arm_values.end()) {
        continue;
      }
      checked_arm_values.emplace(arm_value);
      add_child(m_result.indices[choice], belief, true);
    }
  }
  // if we are an M-state, we don't need to do computational expansions

  if (!m_result.is_m_state && !are_identical) {
    // iterate over all optimal choices w.r.t. this belief
    set<std::pair<ui, ui>, PairHash> checked_arm_values;
    for (auto choice : m_result.indices) {
      // if we already looked at an arm with identical wins/losses, we don't
      // need to expand
      std::pair<ui, ui> arm_value{state[2 * choice], state[2 * choice + 1]};
      auto found = checked_arm_values.find(arm_value);
      if (found != checked_arm_values.end()) {
        continue;
      }
      checked_arm_values.emplace(arm_value);
      // now iterate over all indices to see if we can change our mind
      for (int k = 0; k < meta->nr_of_arms; k++) {
        if (k == choice) {
          continue;
        }
        computational_expansion(choice, k);
      }
    }
  }
  // expand all meta child nodes of this one, both for terminal actions and for
  // computations
  for (auto &[key, action_children_vector] : computational_children) {
    for(auto& action_children: action_children_vector)
    for (auto &meta_child : action_children.first) {
      meta_child->expand();
    }
  }
  for (auto &action_children : terminal_children) {
    for (auto meta_child : action_children.second) {
      meta_child->expand();
    }
  }
  return;
}

void MetaNode::add_child(ui arm, const Belief &new_belief, bool terminal) {
  auto winning_child = this->state;

  winning_child[2 * arm] += 1;
  auto [states, edges] = subgraph(winning_child, new_belief.first,
                                  new_belief.second, meta->max_depth);
  Belief winning_belief{states, edges};

  auto [winning_it, inserted_win] = meta->nodes.try_emplace(
      winning_belief, MetaNode(winning_child, winning_belief, meta));

  auto losing_child = this->state;
  losing_child[2 * arm + 1] += 1;
  auto [states_losing, edges_losing] = subgraph(
      losing_child, new_belief.first, new_belief.second, meta->max_depth);
  Belief losing_belief{states_losing, edges_losing};
  auto [losing_it, inserted_lose] = meta->nodes.try_emplace(
      losing_belief, MetaNode(losing_child, losing_belief, meta));
  if (terminal) {
    terminal_children.try_emplace(
        arm,
        std::array<MetaNode *, 2>{&(winning_it->second), &(losing_it->second)});

  } else {
    computational_children[arm].push_back(std::pair
        {std::array<MetaNode *, 2>{&(winning_it->second), &(losing_it->second)},new_belief.second.size()-belief.second.size()});
  }
}

void MetaNode::computational_expansion(ui terminal_action, ui candidate) {
//check if we can change our mind at all
  // We will use a vector
  std::cout << "Doing computational expansion at state " << this->state
            << ". Greedy action is " << terminal_action
            << ", we want to change our mind to " << candidate << "\n";
  std::vector<ExpansionCandidate> current_expansion;
  std::vector<ExpansionCandidate> next_expansion;

  // First, handle the base case, we're at the root state.
  auto winning_child = state;
  winning_child[2 * candidate] += 1;
  auto losing_child = state;
  losing_child[2 * candidate + 1] += 1;
  // Did we already expand in this direction?
  if (belief.first.find(winning_child) != belief.first.end()) {
    current_expansion.push_back(
        ExpansionCandidate{belief, get_candidates(candidate)});
  }
  // If not, we need to  check if we change our mind on the first try. Then we
  // don't need to do the loop.
  else {
    auto new_belief = belief;
    new_belief.first.emplace(winning_child);
    new_belief.first.emplace(losing_child);
    new_belief.second.emplace(Edge{state, winning_child});
    new_belief.second.emplace(Edge{state, losing_child});
    auto new_gains = eval_basegraph(state, new_belief.first, new_belief.second,
                                    meta->max_depth);
    if (new_gains[candidate] > new_gains[terminal_action] + 1e-7) {
      // We changed our mind. Add metanodes for the winning and losing child and
      // insert them to the children of this node.
      add_child(candidate, belief, false);
    } else // We didn't change our mind, so we need to loop.
    {
      // need to find the expansion candidates from the inherited belief
      set<State, StateHash> temp_candidates = get_candidates(candidate);
      temp_candidates.emplace(winning_child);
      temp_candidates.emplace(losing_child);
      temp_candidates.erase(state);
      current_expansion.push_back({new_belief, temp_candidates});
    }
  }
  // Preallocate the range for the hot loop.
  std::vector<std::size_t> arms;
  arms.reserve(meta->nr_of_arms);
  for (int i = 0; i < meta->nr_of_arms; ++i) {
    arms.push_back(i);
  }
  // store the minimal mindchangers
  std::vector<Belief> minimal_mindchangers;
  // store the beliefs we already checked
  while (!current_expansion.empty()) {
    int k = 0;
    set<Belief, BeliefHash> already_checked;
    for (auto &[current_belief, current_candidates] : current_expansion) {
      // Don't need to check beliefs twice. This is a necessary check, at least
      // for performance.
      if (already_checked.find(current_belief) != already_checked.end()) {
        continue;
      } else {
        already_checked.emplace(current_belief);
      }
      k++;
      if (k % 1000 == 0) {
        std::cout << "checked " << k << " candidates\n";
      }
      for (auto candidate_expansion : current_candidates) {
        if (candidate_expansion.sum() == meta->max_depth - 1) {
          continue;
        }

        // Check if the current candidate state is an M-state or not. If so,
        // only expand in optimal directions.
        auto new_gains =
            eval_basegraph(candidate_expansion, current_belief.first,
                           current_belief.second, meta->max_depth);
        auto &optimal_rewards =
            meta->optimal[candidate_expansion].expeced_gains;
        auto m_result = check_m_state(new_gains, optimal_rewards);
        if (m_result.indices.size() == meta->nr_of_arms) {
          m_result.is_m_state = false;
        }
        // auto &range = m_result.is_m_state ? m_result.indices : arms;
        auto &range = arms;
        // Expand the candidate along the chosen arm. Then, if we changed our
        // mind, we found a mindchanger.
        set<std::pair<ui,ui>, PairHash> checked_arm_values;
        for (auto arm : range) {
          if (m_result.is_m_state && arm != m_result.index) {
            continue;
          }
          std::pair<ui,ui> current_arm_values{candidate_expansion[2*arm],candidate_expansion[2*arm+1]};
          if(checked_arm_values.find(current_arm_values)!=checked_arm_values.end()){
            continue;
          }
          else{
            checked_arm_values.emplace(current_arm_values);
          }
          winning_child = candidate_expansion;
          winning_child[2 * arm] += 1;
          losing_child = candidate_expansion;
          losing_child[2 * arm + 1] += 1;
          auto new_belief = current_belief;
          new_belief.first.emplace(winning_child);
          new_belief.first.emplace(losing_child);
          new_belief.second.emplace(Edge{candidate_expansion, winning_child});
          new_belief.second.emplace(Edge{candidate_expansion, losing_child});
          bool is_minimal = true;
          for (auto &mindchanger : minimal_mindchangers) {
            // check if the new belief contains a minimal mindchanger. Then it
            // is not minimal and is excluded from further consideration
            if (mindchanger < new_belief) {
              is_minimal = false;
              break;
            }
          }
          if (!is_minimal) {
            continue;
          }
          // Now, check if we're a mindchanger.
          auto candidate_gains = eval_basegraph(
              state, new_belief.first, new_belief.second, meta->max_depth);
          if (candidate_gains[terminal_action] + 1e-7 <
              candidate_gains[candidate]) { // we changed our mind
            minimal_mindchangers.push_back(new_belief);
            add_child(candidate, new_belief, false);
          } else // we need to continue expanding
          {
            auto new_candidate =
                ExpansionCandidate{new_belief, current_candidates};
            new_candidate.expansion_nodes.erase(candidate_expansion);
            new_candidate.expansion_nodes.emplace(winning_child);
            new_candidate.expansion_nodes.emplace(losing_child);
            next_expansion.push_back(new_candidate);
          }
        }
      }
    }
    // iterate over the next expansion candidates
    std::swap(next_expansion, current_expansion);
    next_expansion.clear();
  }
}

set<State, StateHash> MetaNode::get_candidates(ui candidate) {
  set<State, StateHash> result;
  std::vector<std::size_t> arms;
  arms.reserve(meta->nr_of_arms);
  for (int i = 0; i < meta->nr_of_arms; ++i) {
    arms.push_back(i);
  }
  for (auto node : belief.first) {
    if (node == this->state) {
      auto child = this->state;
      child[2 * candidate] += 1;
      if (belief.second.find({node, child}) == belief.second.end()) {
        result.emplace(child);
        child[2 * candidate] -= 1;
        child[2 * candidate + 1]++;
        result.emplace(child);
      }
      continue;
    }
    auto new_gains =
        eval_basegraph(node, belief.first, belief.second, meta->max_depth);
    auto &optimal_rewards = meta->optimal[node].expeced_gains;
    auto m_result = check_m_state(new_gains, optimal_rewards);
    auto &range = m_result.is_m_state ? m_result.indices : arms;
    for (auto arm : range) {
      auto child = node;
      child[2 * arm] += 1;
      if (belief.second.find({node, child}) == belief.second.end()) {
        result.emplace(child);
        child[2 * arm] -= 1;
        child[2 * arm + 1]++;
        result.emplace(child);
      }
    }
  }
  return result;
}

bool MetaGraph::stop_expansion(State& root_state, State& current_state){
  if(this->bounds.bounding_type!=BoundingCondition::DEPTH){
    return false;
  }
  return false;
}
