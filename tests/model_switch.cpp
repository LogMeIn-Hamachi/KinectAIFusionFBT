#include "engine.hpp"
#include <fstream>
#include <iostream>
using namespace kf;
int main(int argc,char **argv) {
    if(argc!=3)return 2;
    auto root=std::filesystem::absolute(argv[1]);auto file=root/"tracking-model.txt";
    bool existed=std::filesystem::exists(file);std::string previous;
    if(existed){std::ifstream in(file);previous.assign(std::istreambuf_iterator<char>(in),{});}
    auto restore=[&]{if(existed){std::ofstream out(file);out<<previous;}else std::filesystem::remove(file);};
    try {
        std::filesystem::remove(file);
        Engine engine(root);
        if(engine.view().modelChoice!=3)throw std::runtime_error("Installed optimized SAM is not the default");
        for(int choice:{2,3,1,3,0}) {
            engine.chooseModel(choice);
            if(engine.view().modelChoice!=choice)throw std::runtime_error("Model choice not applied");
            if(Engine(root).view().modelChoice!=choice)throw std::runtime_error("Model choice not persisted");
            engine.start(std::filesystem::absolute(argv[2]));
            engine.chooseModel((choice+1)%4);
            if(engine.view().modelChoice!=choice)throw std::runtime_error("Changed active model unsafely");
            double start=now();View v;
            do {std::this_thread::sleep_for(std::chrono::milliseconds(20));v=engine.view();}while(v.frames<4 && now()-start<60);
            if(v.frames<4 || v.sent || v.output || !v.replay || v.modelHash.empty())throw std::runtime_error("Replay model failed to initialize: "+v.inference+" / "+v.notice);
            if(v.poseSource.find(choice==1?"NLF-S pose":"SAM pose")==std::string::npos)throw std::runtime_error("Wrong learned pose source: "+v.poseSource);
            if(choice>=2 && v.inference.find("selective FP8")==std::string::npos)throw std::runtime_error("FP8 model was not selected");
            if(choice==3 && v.inference.find("optimized sampling")==std::string::npos)throw std::runtime_error("Optimized sampling was not selected");
            engine.stop();
            if(engine.view().running || engine.view().sent)throw std::runtime_error("Model did not stop cleanly");
            std::cout<<(choice==3?"SAM optimized":choice==2?"SAM FP8":choice==1?"NLF-S":"SAM FP16")<<" replay initialized, inferred and stopped; zero OSC.\n"<<std::flush;
        }
        restore();return 0;
    }catch(const std::exception &e){restore();std::cerr<<e.what()<<'\n';return 1;}
}
