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
    std::string &filepath = kwarg("o,output", "Output path for meta policy").set_default("../temp/");
    std::string &filename = kwarg("f,filename", "Name of the output file").set_default("NONE");
    int &bound_type = kwarg("b,bound", "Bound type for meta graph computation. 0 is none, 1 is depth, 2 is size, 3 is computations").set_default(3);
    int &t = kwarg("t,time", "Time horizon of the game").set_default(3);
    int &arms = kwarg("a,arms", "Number of arms").set_default(2);
    int &bound = kwarg("n,number", "Bound size of specified bound").set_default(3);
    double &min = kwarg("min", "Minimum cost of computation").set_default(0.0);
    double &max = kwarg("max", "Maximum cost of computation").set_default(0.001);
    int &samples = kwarg("samples", "Number of samples").set_default(20);
    bool &series = flag("s,series", "Create metapolicies for a range of costs");
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
    if (args.series)
    {
        double step = (args.max - args.min) / (args.samples - 1);

        for (int k = 0; k < args.samples; k++)
        {
            std::ostringstream oss;
            std::string filename = std::format("{}{}.yaml",args.filename,k);

            if (args.filename == "NONE")
            {
                filename = std::format("meta_policy_t{}_sample{}.yaml", args.t, k);
            }
            oss << args.filepath << filename;
            auto final_path = oss.str();
            auto policy = MetaPolicy(&gr, step * k);
            try
            {
                policy.write(final_path);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                return EXIT_FAILURE;
            }
            
            
        }
    }
    else{
            std::ostringstream oss;
            std::string filename = std::format("{}.yaml",args.filename);
            if (args.filename == "NONE")
            {
                filename = std::format("meta_policy_t{}.yaml", args.t);
            }
            oss << args.filepath << filename;
            auto final_path = oss.str();
            auto policy = MetaPolicy(&gr, args.max);

            try
            {
                policy.write(final_path);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                return EXIT_FAILURE;
            }
    }
    return 0;
}
