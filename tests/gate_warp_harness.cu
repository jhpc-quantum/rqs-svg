// Standalone CUDA regression and CUDA-event benchmark. No MPI initialization
// is needed: these tests launch the local production kernels directly.
#include "../src/qcs.cu"
#include <iostream>
#include <complex>

template<int K> struct dense {
    static constexpr int num_target_qubits=K, N=1<<K;
    qcs::complex_t matrix[N*N];
    dense() {
        // Fourier matrix with unequal row/column phases: complex, nonsymmetric unitary.
        for(int r=0;r<N;++r)for(int c=0;c<N;++c) {
            double angle=2*3.141592653589793*r*c/N+0.17*r+0.31*c;
            matrix[r*N+c]=qcs::complex_t(cos(angle),sin(angle))/sqrt(double(N));
        }
    }
    __device__ void apply(const qcs::complex_t* const in[], qcs::complex_t* const out[]) const {
        qcs::complex_t result[N];
        for(int r=0;r<N;++r) { result[r]=0;for(int c=0;c<N;++c)result[r]+=matrix[r*N+c]* *in[c]; }
        for(int r=0;r<N;++r)*out[r]=result[r];
    }
};
template<class Gate> void cpu_check(const Gate&, const std::vector<qcs::complex_t>&,
    const std::vector<qcs::complex_t>&, const std::vector<int>&,
    const std::vector<int>&, const std::vector<int>&) {}
template<int K> void cpu_check(const dense<K>& gate, const std::vector<qcs::complex_t>& initial,
    const std::vector<qcs::complex_t>& actual, const std::vector<int>& targets,
    const std::vector<int>& neg, const std::vector<int>& pos) {
    for(size_t i=0;i<initial.size();++i) {
        bool active=true;for(int b:neg)active &= !((i>>b)&1);for(int b:pos)active &= ((i>>b)&1);
        if(!active)continue;
        int row=0;size_t base=i;
        for(int j=0;j<K;++j) {row|=((i>>targets[j])&1)<<j;base&=~(size_t(1)<<targets[j]);}
        std::complex<double> expected=0;
        for(int c=0;c<(1<<K);++c) {
            size_t source=base;for(int j=0;j<K;++j)source|=size_t((c>>j)&1)<<targets[j];
            auto a=gate.matrix[row*(1<<K)+c], b=initial[source];
            expected+=std::complex<double>(a.real(),a.imag())*std::complex<double>(b.real(),b.imag());
        }
        if(std::abs(expected-std::complex<double>(actual[i].real(),actual[i].imag()))>1e-11)
            throw std::runtime_error("CPU reference mismatch");
    }
}
static void checked(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
template<class Gate> void run(Gate gate,int n,std::vector<int> targets,std::vector<int> neg,std::vector<int> pos) {
    using C=qcs::complex_t;
    size_t count=size_t(1)<<n; C* device=nullptr; checked(cudaMalloc(&device,count*sizeof(C)));
    std::vector<C> initial(count), reference(count), actual(count);
    for(size_t i=0;i<count;++i)initial[i]=C(sin(i*0.17),cos(i*0.23));
    qcs::kernel_common_struct common{n,device};
    checked(cudaMemcpyToSymbol(qcs::kernel_common_constant,&common,sizeof(common)));
    size_t bytes=qcs::kernel_input_qnlist_struct::needed_size(targets.size(),neg.size(),pos.size());
    std::vector<uint64_t> buffer((bytes+7)/8,0);
    auto args=reinterpret_cast<qcs::kernel_input_qnlist_struct*>(buffer.data());
    args->num_target_qubits=targets.size();args->num_negative_control_qubits=neg.size();args->num_positive_control_qubits=pos.size();
    std::vector<int> operands=targets;operands.insert(operands.end(),neg.begin(),neg.end());operands.insert(operands.end(),pos.begin(),pos.end());std::sort(operands.begin(),operands.end());
    std::copy(operands.begin(),operands.end(),args->get_operand_qubit_num_list_sorted());
    std::copy(pos.begin(),pos.end(),args->get_positive_control_qubit_num_list());
    std::copy(targets.begin(),targets.end(),args->get_target_qubit_num_list());
    checked(cudaMemcpyToSymbol(qcs::kernel_input_constant,buffer.data(),bytes));
    uint64_t threads=count>>operands.size(); unsigned block=unsigned(std::min(uint64_t(128),threads));
    qcs::gate_warp::mapping map;bool supported=qcs::gate_warp::build(n,targets,neg,pos,map);
    auto launch=[&](bool warp) {
        if(!warp || !supported) qcs::cuda_gate<<<unsigned(threads/block),block>>>(gate);
        else {
            unsigned blocks=unsigned((map.warps+3)/4);
            switch(map.register_count) {
#define LAUNCH(R) case R:qcs::gate_warp::execute<Gate,R><<<blocks,128>>>(gate,map,device);break
                LAUNCH(0);LAUNCH(1);LAUNCH(2);LAUNCH(3);
#undef LAUNCH
            }
        }
        checked(cudaGetLastError());
    };
    for(int mode=0;mode<2;++mode) {
        checked(cudaMemcpy(device,initial.data(),count*sizeof(C),cudaMemcpyHostToDevice));launch(mode);
        checked(cudaMemcpy(actual.data(),device,count*sizeof(C),cudaMemcpyDeviceToHost));
        cpu_check(gate,initial,actual,targets,neg,pos);
        if(!mode)reference=actual;
        else for(size_t i=0;i<count;++i)if(cuda::std::abs(reference[i]-actual[i])>1e-11)throw std::runtime_error("backend mismatch");
        // Outside controls must be bitwise unchanged.
        for(size_t i=0;i<count;++i) {
            bool active=true;for(int b:neg)active &= !((i>>b)&1);for(int b:pos)active &= ((i>>b)&1);
            if(!active && actual[i]!=initial[i])throw std::runtime_error("inactive amplitude modified");
        }
        for(int j=0;j<5;++j)launch(mode);
        cudaEvent_t start,stop;checked(cudaEventCreate(&start));checked(cudaEventCreate(&stop));
        std::vector<float> times;
        for(int j=0;j<31;++j) {
            checked(cudaEventRecord(start));launch(mode);checked(cudaEventRecord(stop));checked(cudaEventSynchronize(stop));
            float ms;checked(cudaEventElapsedTime(&ms,start,stop));times.push_back(ms);
        }
        std::sort(times.begin(),times.end());
        std::cout<<n<<","<<Gate::num_target_qubits<<","<<targets[0]<<","<<neg.size()<<","<<pos.size()<<","<<(mode&&supported?"warp":"original")<<","<<times[15]<<","<<times[3]<<","<<times[27]<<"\n";
        checked(cudaEventDestroy(start));checked(cudaEventDestroy(stop));
    }
    checked(cudaFree(device));
}
int main(int argc,char** argv) {
    try {
        int n=argc>1?std::stoi(argv[1]):12;if(n<11||n>28)throw std::runtime_error("qubits must be 11..28");
        std::cout<<"qubits,targets,first_target,negative_controls,positive_controls,backend,median_ms,p10_ms,p90_ms\n";
        run(dense<2>{},n,{0,7},{},{});run(dense<2>{},n,{0,7},{},{1});
        run(dense<2>{},n,{1,5},{},{0});run(dense<3>{},n,{1,6,10},{},{3});
        run(dense<2>{},n,{7,0},{2},{});run(dense<3>{},n,{10,7,6},{2},{0,3});
        for(int bit=0;bit<n;++bit) {
            run(dense<1>{},n,{bit},{},{});run(qcs::gate::hadamard{},n,{bit},{},{});
            run(qcs::gate::x{},n,{bit},{},{});run(qcs::gate::z{},n,{bit},{},{});
        }
        std::mt19937 random(42);
        for(int trial=0;trial<20;++trial) {
            std::vector<int> bits(12);for(int b=0;b<12;++b)bits[b]=b;
            std::shuffle(bits.begin(),bits.end(),random);
            run(dense<3>{},12,{bits[0],bits[1],bits[2]},{bits[3]},{bits[4],bits[5]});
        }
        run(dense<2>{},4,{3,0},{1},{}); // fallback, subwarp state space
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
