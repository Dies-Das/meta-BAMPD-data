#include "argparse.hpp"
#include "knowledge_policy.hpp"
#include <iostream>
struct MyArgs : public argparse::Args {
        std::string &filepath = kwarg("o,output", "Output path for meta policy")
                                    .set_default("../temp/");
        std::string &filename =
            kwarg("f,filename", "Name of the output file").set_default("NONE");
        int &t = kwarg("t,time", "Time horizon of the game").set_default(10);
        int &arms = kwarg("a,arms", "Number of arms").set_default(2);
        int &k = kwarg("k", "Bound size of knowledge gradient").set_default(2);
};

int main(int argc, char *argv[]) {

    auto args = argparse::parse<MyArgs>(argc, argv);
    KnowledgeGradientPolicy grad =
        KnowledgeGradientPolicy(args.arms, args.t, args.k);
    OptimalPolicy optimal = OptimalPolicy(args.arms, args.t);
    auto root = State();
    root.data.resize(args.arms * 2);
    std::cout << (grad.value(root)) << std::endl;
    // std::cout << "optimal: " << optimal.policy[root].reward();
}
