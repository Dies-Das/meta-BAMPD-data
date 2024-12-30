#include "meta_policy.hpp"

MetaPolicy::MetaPolicy(MetaGraph* _meta, const double _base_cost):meta(_meta), base_cost(_base_cost){
  
}
MetaPolicyItem MetaPolicy::expand(const Belief& belief){
  auto node_iterator = this->data.find(belief);
  if(node_iterator==this->data.end()){
    return node_iterator->second;
  }
  MetaPolicyItem result;
return result;
}
