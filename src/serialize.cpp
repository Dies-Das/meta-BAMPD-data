#include <fstream>
#include <sstream>

#include <queue>
#include <stdexcept>
#include <iomanip> // for std::quoted
#include "meta_policy.hpp"
#include <boost/unordered_map.hpp>
static std::string pointerToString(const MetaPolicyItem *item, const MetaPolicyItem *root);
static void writeBeliefJson(std::ostringstream &out, const Belief &belief);
static void writeActionsJson(std::ostringstream &out,
                             const std::vector<Action> &actions,
                             const MetaPolicyItem *root,
                             std::unordered_map<const MetaPolicyItem *, bool> &visited,
                             std::queue<const MetaPolicyItem *> &toVisit,
                             bool is_leaf);
static void writeStateJson(std::ostringstream &out, const State &state);
static void writeEdgeJson(std::ostringstream &out, const Edge &edge);

void MetaPolicy::write(const std::string &filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::ostringstream json;

    // Write the root JSON object
    json << "{\n";

    // Metadata
    json << "  \"version\": 1,\n";
    json << "  \"base_cost\": " << base_cost << ",\n";
    json << "  \"arms\": " << meta->nr_of_arms << ",\n";
    json << "  \"time_horizon\": " << meta->max_depth << ",\n";
    json << "  \"bounding\": {\n";
    json << "    \"type\": ";
    switch (meta->bounds.bounding_type) {
        case BoundingCondition::COMPUTATIONS:
            json << "\"Computations\",\n";
            json << "    \"number\": " << meta->bounds.max_computations << "\n";
            break;
        case BoundingCondition::SIZE:
            json << "\"Size\",\n";
            json << "    \"number\": " << meta->bounds.max_belief_size << "\n";
            break;
        case BoundingCondition::DEPTH:
            json << "\"Depth\",\n";
            json << "    \"number\": " << meta->bounds.max_belief_depth << "\n";
            break;
        default:
            json << "\"None\",\n";
            json << "    \"number\": 0\n";
            break;
    }
    json << "  },\n";

    // Nodes
    json << "  \"nodes\": {\n";

    // BFS to write all reachable nodes
    std::queue<const MetaPolicyItem *> toVisit;
    std::unordered_map<const MetaPolicyItem *, bool> visited;

    visited[root] = true;
    toVisit.push(root);

    bool firstNode = true; // Track whether this is the first node to handle commas properly
    while (!toVisit.empty()) {
        auto current = toVisit.front();
        toVisit.pop();
        auto current_state = find_root(current->belief);
        bool is_leaf = current_state.sum() == meta->max_depth - 1;

        if (!firstNode) {
            json << ",\n";
        }
        firstNode = false;

        // Write node
        std::string addrStr = pointerToString(current, root);
        json << "    " << std::quoted(addrStr) << ": {\n";
        json << "      \"state\": ";
        writeStateJson(json, current_state);
        json << ",\n";
        json << "      \"belief\": ";
        writeBeliefJson(json, current->belief);
        json << ",\n";
        json << "      \"gross_gain\": " << current->gross_gain << ",\n";
        json << "      \"optimal_gain\": " << meta->optimal[current_state].reward() << ",\n";
        json << "      \"greedy_gain\": " << meta->greedy[current_state].reward << ",\n";
        json << "      \"actions\": ";
        writeActionsJson(json, current->actions, root, visited, toVisit, is_leaf);
        json << "\n    }";
    }

    json << "\n  }\n";
    json << "}";

    // Write the JSON content to the file
    out << json.str();
    out.close();
}

static std::string pointerToString(const MetaPolicyItem *item, const MetaPolicyItem *root) {
    if (item == root) {
        return "0";
    }
    std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(item);
    std::ostringstream oss;
    oss << std::hex << raw;
    return oss.str();
}

static void writeStateJson(std::ostringstream &out, const State &state) {
    out << "[";
    for (size_t i = 0; i < state.data.size(); ++i) {
        if (i > 0) out << ", ";
        out << state.data[i];
    }
    out << "]";
}

static void writeEdgeJson(std::ostringstream &out, const Edge &edge) {
    out << "[";
    writeStateJson(out, edge[0]);
    out << ", ";
    writeStateJson(out, edge[1]);
    out << "]";
}

static void writeBeliefJson(std::ostringstream &out, const Belief &belief) {
    out << "{\n";
    out << "        \"states\": [";
    bool first = true;
    for (const auto &state : belief.first) {
        if (!first) out << ", ";
        first = false;
        writeStateJson(out, state);
    }
    out << "],\n";
    out << "        \"edges\": [";
    first = true;
    for (const auto &edge : belief.second) {
        if (!first) out << ", ";
        first = false;
        writeEdgeJson(out, edge);
    }
    out << "]\n";
    out << "      }";
}

static void writeActionsJson(std::ostringstream &out,
                             const std::vector<Action> &actions,
                             const MetaPolicyItem *root,
                             std::unordered_map<const MetaPolicyItem *, bool> &visited,
                             std::queue<const MetaPolicyItem *> &toVisit,
                             bool is_leaf) {
    out << "[";
    bool firstAction = true;
    for (const auto &action : actions) {
        if (!firstAction) out << ", ";
        firstAction = false;

        out << "{\n";
        out << "          \"arm\": " << action.arm << ",\n";
        out << "          \"is_computational\": " << (action.is_computational ? "true" : "false") << ",\n";
        out << "          \"net_gain\": " << action.net_gain << ",\n";
        out << "          \"cost_of_action\": " << action.cost_of_action << ",\n";
        out << "          \"children\": [";
        bool firstChild = true;
        for (auto childPtr : action.children) {
            if (!firstChild) out << ", ";
            firstChild = false;

            if (childPtr && !is_leaf) {
                std::string childAddr = pointerToString(childPtr, root);
                out << std::quoted(childAddr);
                if (!visited[childPtr]) {
                    visited[childPtr] = true;
                    toVisit.push(childPtr);
                }
            } else {
                out << "null";
            }
        }
        out << "]\n        }";
    }
    out << "]";
}
