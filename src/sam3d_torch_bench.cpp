#include "core.hpp"
#include <Windows.h>
#include <fstream>
#include <iostream>
int main(int argc,char **argv) {
    try {
        if(argc<5 || argc>7)throw std::runtime_error("kf_torch_bench <decoder-runtime-dir> <decoder.pt> <inputs-dir> <new-output-dir> [fp32|tf32] [decoder-dll]");
        std::string precision=argc>5?argv[5]:"fp32";
        if(precision!="fp32" && precision!="tf32")throw std::runtime_error("Invalid precision");
        auto libs=std::filesystem::absolute(argv[1]),out=std::filesystem::absolute(argv[4]);
        if(std::filesystem::exists(out))throw std::runtime_error("Output exists");
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);AddDllDirectory(libs.c_str());
        auto dllPath=argc>6?std::filesystem::absolute(argv[6]):libs/L"kf_sam3d_decoder.dll";
        auto dll=LoadLibraryExW(dllPath.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if(!dll)throw std::runtime_error("Cannot load native decoder DLL: "+std::to_string(GetLastError()));
        auto create=reinterpret_cast<void*(*)(const char*,int)>(GetProcAddress(dll,"kf_decoder_create_precision"));
        auto run=reinterpret_cast<int(*)(void*,const float*,const float*,const float*,const float*,float*,float*,float*)>(GetProcAddress(dll,"kf_decoder_run"));
        auto destroy=reinterpret_cast<void(*)(void*)>(GetProcAddress(dll,"kf_decoder_destroy"));
        auto error=reinterpret_cast<const char*(*)()>(GetProcAddress(dll,"kf_decoder_error"));
        if(!create||!run||!destroy||!error)throw std::runtime_error("Wrong decoder DLL interface");
        auto model=std::filesystem::absolute(argv[2]).u8string();std::string modelPath(model.begin(),model.end());
        void *decoder=create(modelPath.c_str(),precision=="tf32"?1:0);if(!decoder)throw std::runtime_error(error());
        std::vector<float> input[4];size_t counts[]{1280*32*32,2,2,9};
        for(int i=0;i<4;++i){input[i].resize(counts[i]);std::ifstream f(std::filesystem::path(argv[3])/("input_"+std::to_string(i)+".bin"),std::ios::binary);if(!f.read(reinterpret_cast<char*>(input[i].data()),counts[i]*4))throw std::runtime_error("Invalid input");}
        std::vector<float> result[3];result[0].resize(210);result[1].resize(140);result[2].resize(1143);
        auto execute=[&]{if(!run(decoder,input[0].data(),input[1].data(),input[2].data(),input[3].data(),result[0].data(),result[1].data(),result[2].data()))throw std::runtime_error(error());};
        for(int i=0;i<3;++i)execute();
        std::vector<double> times;
        for(int i=0;i<30;++i){double start=kf::now();execute();times.push_back((kf::now()-start)*1000);}
        std::filesystem::create_directories(out);
        for(int i=0;i<3;++i){for(float x:result[i])if(!std::isfinite(x))throw std::runtime_error("Nonfinite output");std::ofstream f(out/("output_"+std::to_string(i)+".bin"),std::ios::binary);f.write(reinterpret_cast<char*>(result[i].data()),result[i].size()*4);}
        std::sort(times.begin(),times.end());
        std::ofstream report(out/"result.json");report<<"{\"backend\":\"Native LibTorch CUDA graph, no Python\",\"precision\":\""<<precision<<"\",\"median_ms\":"<<times[14]<<",\"p95_ms\":"<<times[28]<<"}\n";
        report.close();
        std::cout<<"Native CUDA decoder: "<<times[14]<<" ms median, "<<times[28]<<" ms p95"<<std::endl;
        destroy(decoder);
        // Keep LibTorch loaded until process exit. Unloading its dependency
        // chain under the Windows loader lock can deadlock worker cleanup.
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
