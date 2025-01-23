#include "meta_policy.hpp"
#include <boost/unordered_map.hpp>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <sstream>
#include <iomanip>

// ============ Forward declarations of helper functions =================
static std::string pointerToString(const MetaPolicyItem *item, const MetaPolicyItem *root);
static void writeBelief(std::ofstream &out, const Belief &belief, size_t indentLevel);
static void writeActions(std::ofstream &out,
                         const std::vector<Action> &actions,
                         const MetaPolicyItem *root,
                         std::unordered_map<const MetaPolicyItem *, bool> &visited,
                         std::queue<const MetaPolicyItem *> &toVisit,
                         size_t indentLevel,
                         bool is_leaf);
static void indent(std::ofstream &out, size_t indentLevel);
static void writeState(std::ofstream &out, const State &state);
static void writeEdge(std::ofstream &out, const Edge &edge);

// ============ Implementation of the MetaPolicy::write method ============
void MetaPolicy::write(const std::string &filename) const
{

    std::ofstream out(filename);
    if (!out.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
        return;
    }

    // Header info
    out << "# MetaPolicy Serialization\n";
    out << "version: 1\n";
    out << "base_cost: " << base_cost << "\n";
    out << "arms: " << meta->nr_of_arms << "\n";
    out << "time_horizon: " << meta->max_depth << "\n";
    switch (meta->bounds.bounding_type)
    {
    case BoundingCondition::COMPUTATIONS:
        out << "bounding_type: Computations\n";
        out << "bounding_number: " << meta->bounds.max_computations << "\n";
        break;
    case BoundingCondition::SIZE:
        out << "bounding_type: Size\n";
        out << "bounding_number: " << meta->bounds.max_belief_size << "\n";
        break;
    case BoundingCondition::DEPTH:
        out << "bounding_type: Depth\n";
        out << "bounding_number: " << meta->bounds.max_belief_depth << "\n";
        break;
    default:
        out << "bounding_type: None\n";
        out << "bounding_number: " << 0 << "\n";
        break;
    }

    out << "nodes:";
    // We'll do a BFS from the root to collect all reachable nodes
    std::queue<const MetaPolicyItem *> toVisit;
    std::unordered_map<const MetaPolicyItem *, bool> visited;

    visited[root] = true;
    toVisit.push(root);

    while (!toVisit.empty())
    {
        auto current = toVisit.front();
        toVisit.pop();
        auto current_state = find_root(current->belief);
        bool is_leaf = current_state.sum() == meta->max_depth - 1;
        // Print a blank line before each node (optional, for readability)
        out << "\n";
        // Print "  <pointer>:"
        std::string addrStr = pointerToString(current, root);
        indent(out, 1);
        out << addrStr << ":\n";
        indent(out,2);
        out << "state: ";
        writeState(out, current_state);
        out << "\n";
        // Write belief
        writeBelief(out, current->belief, 2);
        // Write gross gain
        indent(out,2);
        out << "gross_gain: " << current->gross_gain << "\n";
        indent(out,2);
        out << "optimal_gain: " << meta->optimal[current_state].reward() << "\n";
        indent(out,2);
        out << "greedy_gain: " << meta->greedy[current_state].reward << "\n";
        
        // Write actions
        writeActions(out, current->actions, root, visited, toVisit, 2, is_leaf);
        
        if (!is_leaf)
        {
            for (auto &action : current->actions)
            {
                for (auto &child : action.children)
                {
                    auto it = visited.find(child);
                    if (it == visited.end())
                    {
                        toVisit.push(child);
                    }
                }
            }
        }
    }

    out.close();
}

// ============ Helper Functions =========================================

// Indentation helper (2 spaces per indent level for readability)
static void indent(std::ofstream &out, size_t indentLevel)
{
    for (size_t i = 0; i < indentLevel; ++i)
    {
        out << "  ";
    }
}

// Convert pointer to string, with the root pointer as "0"
static std::string pointerToString(const MetaPolicyItem *item, const MetaPolicyItem *root)
{
    if (item == root)
    {
        return "0";
    }
    std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(item);
    std::ostringstream oss;
    oss << std::hex << raw; // Hex representation
    return oss.str();
}

// Write the Belief in YAML:
// Belief = std::pair<StateSet, EdgeSet>
// StateSet is a set of State, each State is a vector<int> (you mentioned).
// EdgeSet is a set of Edge, each Edge is an array<State,2>.
static void writeBelief(std::ofstream &out, const Belief &belief, size_t indentLevel)
{
    const auto &stateSet = belief.first; // StateSet
    const auto &edgeSet = belief.second; // EdgeSet

    indent(out, indentLevel);
    out << "belief:\n";

    // states
    indent(out, indentLevel + 1);
    out << "states:\n";
    for (const auto &state : stateSet)
    {
        indent(out, indentLevel + 2);
        out << "- ";
        writeState(out, state);
        out << "\n";
    }
    // If empty, we still want to show an empty list
    if (stateSet.empty())
    {
        indent(out, indentLevel + 2);
        out << "[]\n";
    }

    // edges
    indent(out, indentLevel + 1);
    out << "edges:\n";
    for (const auto &edge : edgeSet)
    {
        indent(out, indentLevel + 2);
        out << "- ";
        writeEdge(out, edge);
        out << "\n";
    }
    if (edgeSet.empty())
    {
        indent(out, indentLevel + 2);
        out << "[]\n";
    }
}

// Write the vector<int> that defines a State in bracket form, e.g. [1, 2, 3]
static void writeState(std::ofstream &out, const State &state)
{
    out << "[";
    for (size_t i = 0; i < state.data.size(); ++i)
    {
        out << state[i];
        if (i + 1 < state.data.size())
        {
            out << ", ";
        }
    }
    out << "]";
}

// Write an Edge, which is an array of 2 States
static void writeEdge(std::ofstream &out, const Edge &edge)
{
    // Example format: [[s0], [s1]]
    out << "[";
    writeState(out, edge[0]);
    out << ", ";
    writeState(out, edge[1]);
    out << "]";
}

// Write actions: For each action, we print its fields and the children pointers
static void writeActions(std::ofstream &out,
                         const std::vector<Action> &actions,
                         const MetaPolicyItem *root,
                         std::unordered_map<const MetaPolicyItem *, bool> &visited,
                         std::queue<const MetaPolicyItem *> &toVisit,
                         size_t indentLevel,
                         bool is_leaf)
{
    indent(out, indentLevel);
    out << "actions:\n";
    for (const auto &action : actions)
    {
        // Each action is listed as an item in the YAML array
        indent(out, indentLevel + 1);
        out << "- arm: " << action.arm << "\n";
        indent(out, indentLevel + 2);
        out << "is_computational: " << (action.is_computational ? "true" : "false") << "\n";
        indent(out, indentLevel + 2);
        out << "net_gain: " << action.net_gain << "\n";
        indent(out, indentLevel + 2);
        out << "cost_of_action: " << action.cost_of_action << "\n";

        // Children
        indent(out, indentLevel + 2);
        out << "children:\n";
        for (auto childPtr : action.children)
        {
            indent(out, indentLevel + 3);
            out << "- ";
            if (childPtr && !is_leaf)
            {
                std::string childAddr = pointerToString(childPtr, root);
                out << childAddr << "\n";

                // If we haven't visited this child, enqueue it
                if (!visited[childPtr])
                {
                    visited[childPtr] = true;
                    toVisit.push(childPtr);
                }
            }
            else
            {
                out << "null\n";
            }
        }
    }

}
