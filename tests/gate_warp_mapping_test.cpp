#include "../src/gate_warp_mapping.hpp"
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <random>
#include <set>
using namespace qcs::gate_warp;
static uint64_t deposit(uint64_t v, const int* p, int n) {
    uint64_t x=0; for(int b=0;b<n;++b) x|=((v>>b)&1)<<p[b]; return x;
}
static void check(int n, std::vector<int> t, std::vector<int> neg, std::vector<int> pos) {
    mapping m;
    if (!build(n,t,neg,pos,m)) {
        assert(n-int(neg.size()+pos.size())<5); return;
    }
    std::vector<int> visits(size_t(1)<<n);
    for(uint64_t w=0;w<m.warps;++w) for(int l=0;l<32;++l)
    for(int r=0;r<(1<<m.register_count);++r) {
        auto index=m.positive|deposit(w,m.remaining,m.remaining_count)|deposit(l,m.lane,5)|deposit(r,m.registers,m.register_count);
        assert(index<visits.size()); assert(++visits[index]==1);
        int row=0;
        for(unsigned j=0;j<t.size();++j) row|=((index>>t[j])&1)<<j;
        for(int c=0;c<(1<<t.size());++c) {
            unsigned mask=0; int reg=0;
            for(unsigned j=0;j<t.size();++j) {
                if(m.target_lane[j]>=0) mask|=(((l>>m.target_lane[j])^(c>>j))&1)<<m.target_lane[j];
                else reg|=((c>>j)&1)<<m.target_register[j];
            }
            auto source=m.positive|deposit(w,m.remaining,m.remaining_count)|deposit(l^mask,m.lane,5)|deposit(reg,m.registers,m.register_count);
            auto expected=index;
            for(unsigned j=0;j<t.size();++j) expected=(expected&~(uint64_t(1)<<t[j]))|uint64_t((c>>j)&1)<<t[j];
            assert(source==expected);
        }
        int mapped=0;
        for(unsigned j=0;j<t.size();++j) mapped|=((m.target_lane[j]>=0?(l>>m.target_lane[j]):(r>>m.target_register[j]))&1)<<j;
        assert(mapped==row);
    }
    for(size_t i=0;i<visits.size();++i)
        assert(visits[i]==int((i&m.positive)==m.positive && (i&m.negative)==0));
}
int main() {
    check(9,{0,7},{},{}); check(9,{0,7},{},{1}); check(9,{1,5},{},{0});
    check(12,{1,6,10},{},{3}); check(9,{7,0},{2},{});
    std::mt19937 gen(42);
    for(int trial=0;trial<300;++trial) {
        int n=3+gen()%10, k=1+gen()%3;
        std::vector<int> p(n); for(int i=0;i<n;++i)p[i]=i;
        std::shuffle(p.begin(),p.end(),gen);
        std::vector<int> t(p.begin(),p.begin()+k), neg,pos;
        for(int j=k;j<n;++j) {auto v=gen()%4; if(v==0)neg.push_back(p[j]); if(v==1)pos.push_back(p[j]);}
        check(n,t,neg,pos);
    }
    mapping m; assert(!build(8,{1,1},{},{},m)); assert(!build(8,{1},{1},{},m));
    std::cout<<"warp mapping checks passed\n";
}
