#pragma once
#include <algorithm>
namespace kf {
// SDK 1.8: at most one command per second; 20 seconds rest after 15
// consecutive changes. Two seconds also allows mechanical settling.
struct TiltLimiter {
    double nextAllowed{},lastCommand{-100};
    unsigned consecutive{};
    bool ready(double time) const {return time>=nextAllowed;}
    bool claim(double time) {
        if(!ready(time))return false;
        if(time-lastCommand>=20)consecutive=0;
        lastCommand=time;
        nextAllowed=time+(++consecutive>=15?20:2);
        return true;
    }
    static int target(int current,int direction) {return std::clamp(current+(direction>0?2:-2),-27,27);}
};
}
