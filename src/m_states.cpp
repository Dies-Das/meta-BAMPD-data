#include <variant>
#include<vector>
#include<algorithm>
struct MStateResult {
    bool is_m_state;
    std::vector<std::size_t> indices; // If hasSingleIndex == true, this will contain exactly one element.
};
MStateResult check_m_state(
    const std::vector<double>& belief_gains,
    const std::vector<double>& optimal_gains) 
{
    MStateResult mstate;
    double maxVal = *std::max_element(belief_gains.begin(), belief_gains.end());


    std::vector<std::size_t> maxIndices;
    for (std::size_t i = 0; i < belief_gains.size(); ++i) {
        if (belief_gains[i] == maxVal) {
            maxIndices.push_back(i);
        }
    }
    constexpr double tolerance = 1e-7;
    bool allSmaller = true;
    for (std::size_t i = 0; i < optimal_gains.size(); ++i) {
        if (std::find(maxIndices.begin(), maxIndices.end(), i) == maxIndices.end()) {
            // i is not in maxIndices
            if (optimal_gains[i] >= maxVal - tolerance) {
                allSmaller = false;
                break;
            }
        }
    }
    mstate.indices = maxIndices;
    mstate.is_m_state = allSmaller;
    return mstate;
}