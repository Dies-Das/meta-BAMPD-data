#ifndef BANDIT_HPP
#define BANDIT_HPP
#include <ankerl/unordered_dense.h>
#include <vector>
#include<iostream>
using std::vector;
// using ui8 = std::uint8_t;
using ui = std::uint32_t;
struct State
{
    vector<ui> data;

    State() = default;

    bool operator==(const State& other) const{
        return data==other.data;

    }
    ui &operator[](const ui index)
    {
        return data[index];
    }
    const ui &operator[](const ui index) const
    {
        return data[index];
    }
    State(std::vector<ui> _data) : data(_data)
    {
    }
    std::vector<State> get_children()
    {
        std::vector<State> result;
        result.reserve(data.size());
        for (int k = 0; k < data.size(); k++)
        {
            result.push_back(data);
            result[k][k] += 1;
        }
        return result;
    }
    std::pair<State,State> get_children(ui action){
        State winning_child = data;
        winning_child[2*action]+=1;
        State losing_child = data;
        losing_child[2*action+1]+=1;
        return std::make_pair(winning_child, losing_child);
    }
    ui sum() const
    {
        ui result = 0;
        for (auto v : data)
        {
            result += v;
        }
        return result;
    }
    friend std::ostream& operator<<(std::ostream& os, const State& state){
        os << "(";
        for(auto v: state.data){
            os << v << ",";
        }
        os << ")";
        return os;
    }
};
struct StateHash {
    using is_avalanching = void;

    auto operator()(State const& x) const noexcept -> uint64_t {

        std::size_t h = 0;
        // Simple FNV-1a hash example:
        for (auto d : x.data) {
            h ^= (std::size_t)d;
            h *= 1099511628211ull; // FNV prime
        }
        return h;

    }
};
inline vector<double> get_probabilities(const State &state)
{
    vector<double> result;
    result.resize(state.data.size()/2);
    for (int k = 0; 2*k < state.data.size(); k += 1)
    {
        result[k] = (state[2*k] + 1.0) / (state[2*k] + state[2*k + 1] + 2.0);
    }
    return result;
}

struct PolicyItem
{
    vector<double> expeced_gains;
    ui action;
    std::vector<ui> actions;
    double reward(){
        return expeced_gains[action];
    }
    friend std::ostream& operator<<(std::ostream& os, const PolicyItem& item){
        os << "PolicyItem with expected gains: ";
        for(auto& v:item.expeced_gains){
            os << v << " ";
        }
        os << "and action " << (int)(item.action) << std::endl;
        return os;
    }
};
template <typename Container>
std::size_t argmax_winprobablity(const Container &c)
{
    std::size_t max_index = std::numeric_limits<std::size_t>::max();
    auto max_value = std::numeric_limits<typename Container::value_type>::lowest();

    for (std::size_t i = 0; i < c.size(); i += 1)
    { // Only iterate over even indices as these are the win probabilities for each arm
        if (c[i] > max_value)
        {
            max_value = c[i];
            max_index = i;
        }
    }

    return max_index;
}
#endif