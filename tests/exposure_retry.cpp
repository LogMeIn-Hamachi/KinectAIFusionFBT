#include "exposure_startup.hpp"
#include <stdexcept>
#include <iostream>
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    kf::ExposureStartup state;state.reset(true);
    require(state.due(10),"Initial live frame should allow a command");
    state.begin();state.failed(10);
    require(!state.due(10.9),"Failed commands must back off");
    require(state.due(11),"Transient failure must allow a retry");
    state.begin();state.succeeded();
    require(!state.due(100),"Success must stop control work");
    state.reset(false);
    require(!state.prefer30 && state.due(0),"Reopening must apply the requested mode");
    for(unsigned i=0;i<kf::ExposureStartup::maxAttempts;++i){state.begin();state.failed(i);}
    require(state.exhausted() && !state.due(100),"Persistent failure must stop retries");
    state.reset(true);
    require(state.attempts==0 && state.due(0),"New capture session must get a fresh retry budget");
    std::cout<<"Exposure startup retry, recovery, exhaustion and reset checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
