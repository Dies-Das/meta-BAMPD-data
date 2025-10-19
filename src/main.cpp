#include "argparse.hpp"
#include "bandit.hpp"
#include "greedy_policy.hpp"
#include "meta_graph.hpp"
#include "meta_policy.hpp"
#include "optimal_policy.hpp"
#include <chrono>
#include <iostream>
struct MyArgs : public argparse::Args {
  std::string &filepath =
      kwarg("o,output", "Output path for meta policy").set_default("../temp/");
  std::string &filename =
      kwarg("f,filename", "Name of the output file").set_default("NONE");
  int &bound_type =
      kwarg("b,bound", "Bound type for meta graph computation. 0 is none, 1 is "
                       "depth, 2 is size, 3 is computations")
          .set_default(3);
  int &t = kwarg("t,time", "Time horizon of the game").set_default(3);
  int &arms = kwarg("a,arms", "Number of arms").set_default(2);
  int &bound =
      kwarg("n,number", "Bound size of specified bound").set_default(3);
  double &min = kwarg("min", "Minimum cost of computation").set_default(0.0);
  double &max = kwarg("max", "Maximum cost of computation").set_default(0.001);
  int &samples = kwarg("samples", "Number of samples").set_default(20);
  bool &series = flag("s,series", "Create metapolicies for a range of costs");
  std::string &b_state = kwarg("state", "Current b state").set_default("");
};
int main(int argc, char *argv[]) {
  auto args = argparse::parse<MyArgs>(argc, argv);
  BoundingParameters param = BoundingParameters(
      static_cast<BoundingCondition>(args.bound_type), args.bound);

  if (args.series) {
    double step = (args.max - args.min) / (args.samples - 1);

    auto gr = MetaGraph(param, args.t, 0, 0, args.arms);
    for (int k = 0; k < args.samples; k++) {
      std::ostringstream oss;
      std::string filename = std::format("{}{}.json", args.filename, k);

      if (args.filename == "NONE") {
        filename = std::format("meta_policy_t{}_sample{}.json", args.t, k);
      }
      oss << args.filepath << filename;
      auto final_path = oss.str();
      auto policy = MetaPolicy(&gr, step * k);
      try {
        policy.write(final_path);
      } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
      }
    }
  } else {
    State initial;
    if (args.b_state.empty()) {
      initial.data.resize(2 * args.arms);
    } else {
      std::stringstream ss;
      std::string token;
      while (std::getline(ss, token, ',')) {
        initial.data.push_back(std::stoi(token));
      }
    }
    auto gr = MetaGraph(initial, param, args.t, 0, 0, args.arms);
    std::ostringstream oss;
    std::string filename = std::format("{}json", args.filename);
    if (args.filename == "NONE") {
      filename = std::format("meta_policy_t{}.json", args.t);
    }
    oss << args.filepath << "/" << filename;
    auto final_path = oss.str();
    auto policy = MetaPolicy(&gr, args.max);

    try {
      policy.write(final_path);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      return EXIT_FAILURE;
    }
  }
  return 0;
}
