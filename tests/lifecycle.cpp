#include "engine.hpp"
#include <iostream>
using namespace kf;
int main() {
    const auto root=std::filesystem::temp_directory_path()/("kf-offline-lifecycle-"+std::to_string(now()));
    const auto broken=root/"broken.kfr",valid=root/"synthetic.kfr";
    try {
        std::filesystem::create_directory(root);
        {std::ofstream file(broken);file<<"invalid";}
        {
            Engine engine(root); // Empty isolated root: no models, camera, preferences or GPU sessions.
            engine.chooseLowEndCalibration(true);
            if(!engine.view().lowEndCalibration)throw std::runtime_error("Low-end calibration choice was not applied");
            for(int attempt=0;attempt<2;++attempt) {
                engine.start(broken); // Nonempty replay path never enters camera/VR-input capture.
                const double deadline=now()+3;
                while(engine.view().running && now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(5));
                auto state=engine.view();
                if(state.running || state.output || state.recording || state.sent || state.notice.find("Press Start")==std::string::npos)
                    throw std::runtime_error("Fatal capture error did not stop safely and allow restart");
                engine.output(true);
                if(engine.view().output)throw std::runtime_error("Stopped engine accepted output");
                engine.stop();
            }
            auto config=std::make_shared<ReplayConfig>();config->settings.contacts=false;config->settings.constraints=false;
            {RecordingWriter writer;writer.open(valid,"Synthetic empty frames; no camera data");
                for(int i=0;i<30;++i){Frame frame;frame.host=frame.arrival=10+i/30.;frame.depthId=i;frame.runConfig=config;writer.write(frame);}}
            engine.start(valid);
            const double deadline=now()+3;
            while(engine.view().frames==0 && engine.view().running && now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if(!engine.view().frames)throw std::runtime_error("Synthetic replay did not process");
            auto settings=engine.view().settings;settings.contacts=true;settings.constraints=true;settings.baseline=1;
            engine.settings(settings);
            auto state=engine.view();
            if(state.settings.contacts || state.settings.constraints || state.settings.baseline!=1)
                throw std::runtime_error("Replay UI does not reflect recorded controls and explicit comparison override");
            engine.output(true);
            if(engine.view().output || engine.view().sent)throw std::runtime_error("Replay enabled tracking output");
            engine.stop();
            if(engine.view().running || engine.view().replay)throw std::runtime_error("Stop retained running/replay state");
        }
        {
            Engine restored(root);
            if(!restored.view().lowEndCalibration)throw std::runtime_error("Low-end calibration choice was not restored after restart");
            restored.chooseLowEndCalibration(false);
        }
        std::filesystem::remove(broken);std::filesystem::remove(valid);
        std::filesystem::remove(root/"calibration-low-end.txt");std::filesystem::remove(root);
        std::cout<<"Offline engine checks passed: fatal error, restart, output refusal and replay controls. No camera, GPU inference or tracker output.\n";
        return 0;
    } catch(const std::exception& e) {
        std::filesystem::remove(broken);std::filesystem::remove(valid);
        std::filesystem::remove(root/"calibration-low-end.txt");std::filesystem::remove(root);
        std::cerr<<e.what()<<'\n';return 1;
    }
}
