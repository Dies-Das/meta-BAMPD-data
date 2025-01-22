#include <iostream>
// #include "bandit.hpp"
#include "greedy_policy.hpp"
#include "optimal_policy.hpp"
#include <chrono>
#include "meta_graph.hpp"
#include "meta_policy.hpp"
#include "argparse.hpp"
struct MyArgs : public argparse::Args
{
    std::string &filepath = kwarg("o,output", "Output path for meta policy").set_default("..data/");
    int &bound_type = kwarg("b,bound", "Bound type for meta graph computation. 0 is none, 1 is depth, 2 is size, 3 is computations").set_default(3);
    int &t = kwarg("t,time", "Time horizon of the game").set_default(12);
    int &arms = kwarg("a,arms", "Number of arms").set_default(2);
    int &bound = kwarg("n,number", "Bound size of specified bound").set_default(3);
    double &min = kwarg("min", "Minimum cost of computation").set_default(0.0);
    double &max = kwarg("max", "Maximum cost of computation").set_default(0.15);
    int &samples = kwarg("samples", "Number of samples").set_default(20);
};
int main(int argc, char *argv[])
{
    auto args = argparse::parse<MyArgs>(argc, argv);
    BoundingParameters param = BoundingParameters(static_cast<BoundingCondition>(args.bound_type), args.bound);

    auto start = std::chrono::high_resolution_clock::now();
    auto gr = MetaGraph(param, args.t, 0, 0, args.arms);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Execution time (seconds): " << elapsed.count() << "\n";
    std::cout << "Created Metagraph\n";
    double step = (args.max - args.min)/(args.samples-1);
    for (int k = 0; k < args.samples; k++)
    {
        std::ostringstream oss;
        oss << args.filepath << "meta_policy_t" << args.t << "_sample" << k << ".yaml";
        auto final_path = oss.str();
        auto policy = MetaPolicy(&gr, step*k);
        std::cout << "Created policy" << std::endl;
        policy.write(final_path);
    }

}
