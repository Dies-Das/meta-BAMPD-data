#include <iostream>
// #include "bandit.hpp"
#include "greedy_policy.hpp"
#include "optimal_policy.hpp"
#include <chrono>
#include "meta_graph.hpp"
#include "meta_policy.hpp"
std::ostream& operator<<(std::ostream& os, const Belief& belief){
    for(auto& v: belief.first){
        os << v << ";";
    }
    return os;
}
int main(){
    
    int k=5;
        std::cout << "###########" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    OptimalPolicy greedy = OptimalPolicy(2,k);
        auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "Execution time (seconds): " << elapsed.count() << "\n";
    std::cout <<"Root values: " <<  greedy.policy[State({0,0,0,0})] << std::endl;

    auto gr = MetaGraph(BoundingParameters(),8,0,0,2);
    std::cout << "constructed metagraph" << std::endl;
    std::cout << gr.nodes.size() << std::endl;
    for(auto& [key,item]:gr.nodes){
        for(auto& v: key.first){
        // if (v[0]==0)
        std::cout << key << std::endl;}
    }
    auto policy = MetaPolicy(&gr,0.01);
}
