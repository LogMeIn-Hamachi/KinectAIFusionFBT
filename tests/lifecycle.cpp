#include "engine.hpp"
#include <iostream>
using namespace kf;
int main(int argc, char **argv) {
    try {
        auto root = std::filesystem::absolute(
            argc > 1 ? argv[1] : std::filesystem::path(argv[0]).parent_path().string());
        Engine engine(root);
        for (int cycle = 0; cycle < 2; ++cycle) {
            engine.start();
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            engine.stop();
            auto s = engine.view();
            if (s.running || s.output || s.recording)
                throw std::runtime_error("Stop did not clear lifecycle flags");
            if (s.sent)
                throw std::runtime_error("Unexpected UDP output without alignment");
        }
        auto broken = root / "lifecycle-invalid-recording.kfr";
        {
            std::ofstream f(broken);
            f << "invalid";
        }
        engine.start(broken);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.stop();
        auto s = engine.view();
        if (s.sent || s.running)
            throw std::runtime_error("Corrupt replay escaped lifecycle handling");
        std::filesystem::remove(broken);
        std::cout << "3 native lifecycle scenarios passed: cancel initialization, restart, corrupt replay. "
                     "Zero OSC packets.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
