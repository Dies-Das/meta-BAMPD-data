#include <variant>
#include<vector>
#include<algorithm>
struct MStateResult {
    bool is_m_state;
    std::size_t index;
    std::vector<std::size_t> indices; 
};
MStateResult check_m_state(
    const std::vector<double>& belief_gains,
    const std::vector<double>& optimal_gains) 
{
    MStateResult mstate;
    mstate.is_m_state=false;
    double maxVal = *std::max_element(belief_gains.begin(), belief_gains.end());


    // rewrite this to only loop once both for indices and m-state considerations
    for (std::size_t i = 0; i < belief_gains.size(); ++i) {
        if (belief_gains[i] == maxVal) {// maybe with tolerance? 
            mstate.indices.push_back(i);
        }
    }
    constexpr double tolerance = 1e-7;
    
    // check pathological case: we could have same (optimal) greedy value but greedy > optimal policy
    for(auto candidate_index:mstate.indices){
        bool allSmaller = true;
    for (std::size_t i = 0; i < optimal_gains.size(); ++i) {
        if (i != candidate_index) {
            // i is not in maxIndices
            if (optimal_gains[i] >= maxVal - tolerance) {
                allSmaller = false;
                break;
            }
        }
    }
    if(allSmaller){
        mstate.index = candidate_index;
        mstate.is_m_state=true;
        break;
    }
    }



    return mstate;
}
