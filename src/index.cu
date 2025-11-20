#include "hybrid_search.cuh"
#include "utils.cuh"

#define MAX_P 512 // number of samples for each node during NN-Descent

#define K 64 // the degree of kNN graph

#define INF_DIS 1000000.0f

#define POINTS 1000000 // maximal number of vectors

#define MAX_SPARSE 570 
#define MAX_BM25 780 
#define KEYEDGE 32 
#define KEYEDGEFINAL 8 

using namespace std;

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

// struct used to calculate the groundtruth
struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}
    Neighbor(const Neighbor& nei){
        id = nei.id;
        distance = nei.distance;
        flag = nei.flag;
    }

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance;
    }
};

// hybrid distance function on CPU
float compare(float* a, float* b, unsigned dim, unsigned* idx1, float* val1, unsigned len1, unsigned* idx2, float* val2, unsigned len2, unsigned* bm25_idx1, float* bm25_val1, unsigned bm25_len1, unsigned* bm25_idx2, float* bm25_val2, unsigned bm25_len2){
    float res = 0.0;
    // inner product metric
    for(unsigned i = 0; i < dim; i++){
        // res += ((a[i] - b[i]) * (a[i] - b[i]));
        res += (a[i] * b[i]);
    }
    // cout << res << ", ";
    unsigned p1 = 0, p2 = 0;
    while(p1 < len1 && p2 < len2){
        if(idx1[p1] < idx2[p2]) p1++;
        else if(idx1[p1] > idx2[p2]) p2++;
        else{
            res += val1[p1] * val2[p2];
            p1++;
            p2++;
        }
    }
    unsigned bm25_p1 = 0, bm25_p2 = 0;
    while(bm25_p1 < bm25_len1 && bm25_p2 < bm25_len2){
        if(bm25_idx1[bm25_p1] < bm25_idx2[bm25_p2]) bm25_p1++;
        else if(bm25_idx1[bm25_p1] > bm25_idx2[bm25_p2]) bm25_p2++;
        else{
            res += bm25_val1[bm25_p1] * bm25_val2[bm25_p2];
            bm25_p1++;
            bm25_p2++;
        }
    }
    return 1-res;
}

// randomly initialize the knn graph
__device__ inline unsigned cinn_nvgpu_uniform_random(unsigned long long seed, unsigned node_num){
    curandStatePhilox4_32_10_t state;
    int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    curand_init(seed, idx, 1, &state);
    return (unsigned)(((float)node_num) * curand_uniform(&state)) % node_num;
}

__device__ float nei_distance[POINTS][K];

__device__ float reverse_distance[POINTS][K];

__device__ bool nei_visit[POINTS][K];

__device__ unsigned reverse_num[POINTS] = {0};

__device__ unsigned new_list[POINTS][K];

__global__ void initialize_graph(unsigned* graph, unsigned node_num, float* data){
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        graph[bid * K + i] = cinn_nvgpu_uniform_random(clock()*(unsigned long long)bid, node_num); // (K = 32 * n, is this assumption correct?)
        nei_distance[bid][i] = INF_DIS;
        nei_visit[bid][i] = false;
    }
}

// check if an item is in the hash table: used to recycle the keyword edges
__device__ unsigned hash_check(unsigned *hash_table, unsigned key){
    const unsigned bit_mask = HASHLEN - 1;
    unsigned index = ((key ^ (key >> HASHSIZE)) & bit_mask);
    const unsigned stride = 1;
    for(unsigned i = 0; i < HASHLEN; i++){
        // const unsigned old = atomicCAS(&hash_table[index], 0xFFFFFFFF, key);
        unsigned old = hash_table[index];
        if(old == 0xFFFFFFFF){
            return 1;
        }
        else if(old == key){
            return 0;
        }
        index = (index + stride) & bit_mask;
    }
    return 0;
}

// bm25 distance calculation kernel
__device__ float cal_bm25_dist(unsigned* node_idx1, float* node_val1, unsigned num1, unsigned* node_idx2, float* node_val2, unsigned num2, unsigned laneid, unsigned* cur_nei2, unsigned* hash_table, bool* flag_per_warp){
    float res_dist = 0.0;
    for(unsigned j = laneid; j < num1; j += blockDim.x){
        unsigned val = node_idx1[j];
        unsigned tmp = num2, res_id = 0;
        // binary search
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            unsigned cand = node_idx2[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (node_idx2[res_id] < val);
        if(res_id <= (num2 - 1) && node_idx2[res_id] == val){
            res_dist += (node_val2[res_id] * node_val1[j]);
        }
        else{
            // if the query and the document have common keyword, record it into flag_per_warp
            if(!hash_check(hash_table, val)){
                flag_per_warp[threadIdx.y] = 1;
            }
        }
    }

    return res_dist;
}

// performing NN-Descent on the GPU
__global__ void nn_descent_gpu(unsigned* graph, unsigned* reverse_graph, unsigned it, const half* __restrict__ values, unsigned node_num, unsigned all_it, unsigned* sparse_off, unsigned* sparse_idx, float* sparse_val, unsigned* bm25_off, unsigned* bm25_idx, float* bm25_val, float* data_power){
    unsigned tmp_it = threadIdx.y/4;
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ unsigned fetch_ids[4];
    __shared__ int lock;
    __shared__ unsigned new_list_shared[MAX_P], nei_list_shared[K], new_list2[MAX_P];
    __shared__ float new_dist_shared[MAX_P], nei_dist_shared[K], new_dist2[MAX_P];
    __shared__ bool nei_visit_shared[K];
    __shared__ half4 tmp_val_sha[DIM/4];
    __shared__ unsigned node_idx[MAX_SPARSE];
    __shared__ float node_val[MAX_SPARSE];
    __shared__ unsigned node_idx_bm25[MAX_BM25];
    __shared__ float node_val_bm25[MAX_BM25];
    if(tid == 0) lock = 1;
    __syncthreads();
    if(threadIdx.y % 4 == 0 && laneid == 0){
        while (atomicExch(&lock, 0) == 0);
        while(nei_visit[bid][tmp_it] == true && tmp_it < K){
            tmp_it++;
        }
        nei_visit[bid][tmp_it] = true;
        lock = 1;
        fetch_ids[threadIdx.y / 4] = tmp_it % K;
    }
    __syncthreads();
    tmp_it = fetch_ids[threadIdx.y / 4];
    unsigned samp_id = graph[bid * K + tmp_it];
    if(threadIdx.y % 4 < 2){
        new_list_shared[tid] = graph[samp_id * K + laneid + (threadIdx.y%4)*32];
    }
    else{
        // new_list_shared[tid] = reverse_graph[samp_id * K + cinn_nvgpu_uniform_random(clock(), K)];
        new_list_shared[tid] = reverse_graph[samp_id * K + laneid + (threadIdx.y%4-2)*32];
    }


    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        new_list2[i] = graph[bid * K + i];
    }
    __syncthreads();

    bitonic_sort_id_new2(new_list_shared, MAX_P);
    bitonic_sort_id_new2(new_list2, K);

    for(unsigned i = tid + 1; i < MAX_P; i += blockDim.x * blockDim.y){
        new_dist_shared[i] = (new_list_shared[i] == new_list_shared[i - 1] ? INF_DIS : (new_list_shared[i] == bid ? INF_DIS : 0));
    }

    if(tid == 0){
        if(new_list_shared[0] != bid){
            new_dist_shared[0] = 0;
        }
        else{
            new_dist_shared[0] = INF_DIS;
        }
    }
    __syncthreads();
    for(unsigned i = tid; i < MAX_P; i += blockDim.x * blockDim.y){
        if(new_dist_shared[i] > 1.0f) continue;
        unsigned tmp = K, res_id = 0;
        unsigned val = new_list_shared[i];
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            unsigned cand = new_list2[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (new_list2[res_id] < val);
        unsigned tmp_count = 0;
        if(new_list2[res_id] == new_list_shared[i]){
            new_dist_shared[i] = INF_DIS;
        }
    }
    __syncthreads();
    // cal_distance
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
        tmp_val_sha[i] = Load(&values[bid * DIM + 4 * i]);
    }
    for(unsigned i = sparse_off[bid] + tid; i < sparse_off[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx[i - sparse_off[bid]] = sparse_idx[i];
        node_val[i - sparse_off[bid]] = sparse_val[i];
    }
    for(unsigned i = bm25_off[bid] + tid; i < bm25_off[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx_bm25[i - bm25_off[bid]] = bm25_idx[i];
        node_val_bm25[i - bm25_off[bid]] = bm25_val[i];
    }
    __syncthreads();
    half2 const_one;
    const_one.x = 1.0; const_one.y = 1.0;
    
    // hybrid distance optimization
    #pragma unroll
    for(unsigned i = threadIdx.y; i < MAX_P; i += blockDim.y){
        if(new_dist_shared[i] < 1.0){
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                half4 val2 = Load(&values[new_list_shared[i] * DIM + j * 4]);
                val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
                val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
            }
            float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[new_list_shared[i]], sparse_val+sparse_off[new_list_shared[i]], sparse_off[new_list_shared[i]+1]-sparse_off[new_list_shared[i]], node_idx, node_val, sparse_off[bid+1]-sparse_off[bid], laneid)
                                + cal_sparse_dist(bm25_idx+bm25_off[new_list_shared[i]], bm25_val+bm25_off[new_list_shared[i]], bm25_off[new_list_shared[i]+1]-bm25_off[new_list_shared[i]], node_idx_bm25, node_val_bm25, bm25_off[bid+1]-bm25_off[bid], laneid);
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
            }

            if(laneid == 0){
                new_dist_shared[i] = 1 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
            }
        }
    }
    
    __syncthreads();
    for(unsigned i = tid; i < MAX_P; i += blockDim.x * blockDim.y){
        new_list2[i] = new_list_shared[i];
        new_dist2[i] = new_dist_shared[i];
    }

    // merge
    // sort (and remove duplicates)
    
    float min_ele = nei_distance[bid][K - 1];
    __shared__ unsigned dup_num;
    if(tid == 0) dup_num = 0;
    __syncthreads();

    for(unsigned i = tid; i < MAX_P; i += blockDim.x * blockDim.y){
        if(new_dist2[i] < min_ele){
            unsigned id_tmp = atomicAdd(&dup_num, 1);
            new_list_shared[id_tmp] = new_list2[i];
            new_dist_shared[id_tmp] = new_dist2[i];
        }
    }
    
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        nei_visit_shared[i] = nei_visit[bid][i];
        nei_dist_shared[i] = nei_distance[bid][i];
        nei_list_shared[i] = graph[bid * K + i];
    }
    __syncthreads();
    

    for(unsigned i = tid; i < dup_num; i += blockDim.x * blockDim.y){
        float val = new_dist_shared[i];
        // binary search
        unsigned tmp = K, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            float cand = nei_dist_shared[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (nei_dist_shared[res_id] < val);
        unsigned tmp_count = 0;
        while(res_id + tmp_count < K && nei_dist_shared[res_id + tmp_count] == new_dist_shared[i]){
            if(nei_list_shared[res_id + tmp_count] == new_list_shared[i]){
                new_dist_shared[i] = INF_DIS;
                res_id = K;
                break;
            }
            tmp_count++;
        }
        new_list2[i] = res_id;
    }
    __syncthreads();

    bitonic_sort_new2(new_dist_shared, new_list_shared, new_list2, dup_num);

    // merge
    if(dup_num > 0){
    for(unsigned i = threadIdx.y; i < (K + dup_num + blockDim.x - 1) / blockDim.x; i += blockDim.y){
        unsigned res_id = 0, id_reg;
        bool visit_reg = false;
        float val;
        if(i < K / blockDim.x){
            val = nei_dist_shared[laneid + i * blockDim.x];
            id_reg = nei_list_shared[laneid + i * blockDim.x];
            visit_reg = nei_visit_shared[laneid + i * blockDim.x];
            unsigned tmp = dup_num;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = new_dist_shared[res_id + halfsize];
                res_id += ((cand <= val) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id += (new_dist_shared[res_id] <= val);
            res_id += (laneid + i * blockDim.x);
        }
        else{
            if(laneid + i * blockDim.x - K < dup_num){
                val = new_dist_shared[laneid + i * blockDim.x - K];
                id_reg = new_list_shared[laneid + i * blockDim.x - K];
                res_id = (new_list2[laneid + i * blockDim.x - K] + laneid + i * blockDim.x - K);
            }
            else{
                res_id = K;
            }
        }
        __syncthreads();
        if(res_id < K){
            nei_distance[bid][res_id] = val;
            graph[bid * K + res_id] = id_reg;
            nei_visit[bid][res_id] = visit_reg;
            unsigned tmp_id = atomicAdd(&reverse_num[id_reg], 1);
            if(tmp_id < K) {
                reverse_graph[id_reg * K + tmp_id] = bid;
                reverse_distance[id_reg][tmp_id] = val;
            }
            else{
                if(val < reverse_distance[id_reg][tmp_id % K]){
                    reverse_graph[id_reg * K + tmp_id % K] = bid;
                    reverse_distance[id_reg][tmp_id % K] = val;
                }
            }
        }
    }
    }
}

__global__ void reset_reverse_num(unsigned points_num){
    unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(unsigned i = tid ; i < points_num; i += gridDim.x * blockDim.x){
        reverse_num[i] = 0;
    }
}

// merge the neighbor list and the reverse neighbor list on the GPU
__global__ void merge_reverse_plus(unsigned* graph, unsigned* reverse_graph){
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ unsigned graph_nei[K], reverse_nei[K], new_list2[K];
    __shared__ float graph_nei_dist[K], reverse_nei_dist[K];
    unsigned reverse_act_num = min(K, reverse_num[bid]);
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        graph_nei[i] = graph[bid * K + i];
        graph_nei_dist[i] = nei_distance[bid][i];
    }
    for(unsigned i = tid; i < reverse_act_num; i += blockDim.x * blockDim.y){
        reverse_nei[i] = reverse_graph[bid * K + i];
        reverse_nei_dist[i] = reverse_distance[bid][i];
    }
    __syncthreads();

    bitonic_sort_by_id(reverse_nei_dist, reverse_nei, reverse_act_num);
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        unsigned val = graph_nei[i];
        // binary search
        unsigned tmp = reverse_act_num, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            unsigned cand = reverse_nei[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (reverse_nei[res_id] < val);
        if(reverse_nei[res_id] == graph_nei[i]){
            new_list2[res_id] = K;
            reverse_nei_dist[res_id] = INF_DIS;
        }
    }
    __syncthreads();
    for(unsigned i = tid; i < reverse_act_num; i += blockDim.x * blockDim.y){
        float val = reverse_nei_dist[i];
        // binary search
        unsigned tmp = K, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            float cand = graph_nei_dist[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (graph_nei_dist[res_id] < val);
        unsigned tmp_count = 0;
        while(res_id + tmp_count < K && abs(graph_nei_dist[res_id + tmp_count] - reverse_nei_dist[i]) < 0.1){
            if(graph_nei[res_id + tmp_count] == reverse_nei[i]){
                reverse_nei_dist[i] = INF_DIS;
                res_id = K;
                break;
            }
            tmp_count++;
        }
        new_list2[i] = res_id;
    }
    __syncthreads();
    bitonic_sort_new2(reverse_nei_dist, reverse_nei, new_list2, reverse_act_num);

    if(reverse_act_num > 0){
    for(unsigned i = threadIdx.y; i < (K + reverse_act_num + blockDim.x - 1) / blockDim.x; i += blockDim.y){
        unsigned res_id = 0, id_reg;
        float val;
        if(i < K / blockDim.x){
            val = graph_nei_dist[laneid + i * blockDim.x];
            id_reg = graph_nei[laneid + i * blockDim.x];
            unsigned tmp = reverse_act_num;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = reverse_nei_dist[res_id + halfsize];
                res_id += ((cand <= val) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id += (reverse_nei_dist[res_id] <= val);
            res_id += (laneid + i * blockDim.x);
        }
        else{
            if(laneid + i * blockDim.x - K < reverse_act_num){
                val = reverse_nei_dist[laneid + i * blockDim.x - K];
                id_reg = reverse_nei[laneid + i * blockDim.x - K];
                res_id = (new_list2[laneid + i * blockDim.x - K] + laneid + i * blockDim.x - K);
            }
            else{
                res_id = K;
            }
        }
        __syncthreads();
        if(res_id < K){
            nei_distance[bid][res_id] = val;
            graph[bid * K + res_id] = id_reg;
        }
    }
    }
}

unsigned res_graph[K * POINTS];

// compute the norm of each hybrid vector
__global__ void cal_power(half* data_half, float* data_power, unsigned dim, unsigned* sparse_off, unsigned* sparse_idx, float* sparse_val, unsigned* bm25_off, unsigned* bm25_idx, float* bm25_val, unsigned points_num){
    unsigned tid = threadIdx.x;
    __shared__ float shared_val[(DIM + 31) / 32];
    for(unsigned bid = blockIdx.x; bid < points_num; bid += gridDim.x){
    for(unsigned i = tid; i < DIM; i += blockDim.x){
        half val_res = data_half[bid * dim + i];
        val_res = __hmul(val_res, val_res);
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }
        if(tid % 32 == 0) shared_val[i / 32] = __half2float(val_res);
    }
    __syncthreads();
    for(unsigned i = tid; i < ((sparse_off[bid + 1] - sparse_off[bid] + 31) / 32) * 32; i+= blockDim.x){
        float sparse_dis;
        if(sparse_off[bid] + i < sparse_off[bid + 1]){
            sparse_dis = sparse_val[sparse_off[bid] + i];
            sparse_dis *= sparse_dis;
        }
        else sparse_dis = 0.0;
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
        }
        if(tid % 32 == 0) shared_val[tid / 32] += sparse_dis;
    }
    __syncthreads();
    for(unsigned i = tid; i < ((bm25_off[bid + 1] - bm25_off[bid] + 31) / 32) * 32; i+= blockDim.x){
        float bm25_dis;
        if(bm25_off[bid] + i < bm25_off[bid + 1]){
            bm25_dis = bm25_val[bm25_off[bid] + i];
            bm25_dis *= bm25_dis;
        }
        else bm25_dis = 0.0;
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            bm25_dis += __shfl_down_sync(0xffffffff, bm25_dis, lane_mask);
        }
        if(tid % 32 == 0) shared_val[tid / 32] += bm25_dis;
    }
    __syncthreads();
    if(tid == 0){
        float val_r = 0.0;
        for(unsigned i = 0; i < (DIM + 31) / 32; i++){
            val_r += shared_val[i];
            shared_val[i] = 0.0;
        }
        data_power[bid] = val_r;
    }
    }
}

__device__ unsigned Keyword_edge[POINTS][KEYEDGE+1] = {0};

// the pruning kernel in Allan-Poe
__global__ void allan_poe(unsigned* graph, unsigned d, unsigned* reverse_graph, const half* __restrict__ values, float* data_power, unsigned* sparse_off, unsigned* sparse_idx, float* sparse_val, unsigned* bm25_off, unsigned* bm25_idx, float* bm25_val){
    __shared__ unsigned nei1[K];
    __shared__ unsigned nei2[K];
    __shared__ unsigned nei0[K];
    __shared__ unsigned detour_num[K];
    __shared__ half4 tmp_val_sha[DIM/4];
    __shared__ unsigned node_idx[MAX_SPARSE];
    __shared__ float node_val[MAX_SPARSE];
    __shared__ unsigned node_idx_bm25[MAX_BM25];
    __shared__ float node_val_bm25[MAX_BM25];
    __shared__ unsigned tmp_id, ip_neighbors, cur_nei, cur_nei2;
    __shared__ bool flag_per_warp[16];
    __shared__ unsigned hash_table[HASHLEN];

    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;

    for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
        nei0[j] = graph[bid * K + j];
        detour_num[j] = 0;
    }
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    __syncthreads();
    for(unsigned i = bm25_off[bid] + tid; i < bm25_off[bid + 1]; i += blockDim.x * blockDim.y){
        hash_insert(hash_table, bm25_idx[i]);
    }
    if(laneid == 0) {
        flag_per_warp[threadIdx.y] = 0;
    }
    __syncthreads();

    // optimized cagra pruning strategy
    for(unsigned i = 0; i < K / 2; i++){
        unsigned nei_id1 = graph[bid * K + i], nei_id2 = graph[bid * K + K - 1 - i];
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            nei1[j] = graph[nei_id1 * K + j];
            nei2[j] = graph[nei_id2 * K + j];
        }
        __syncthreads();
        unsigned tmp = K, res_id = 0;
        if(tid < K - 1){
            unsigned find_key = ((tid < K - 1 - i) ? nei0[i + 1 + tid] : nei0[tid + 1]);
            unsigned* shared_nei = ((tid < K - 1 - i) ? &nei1[0] : &nei2[0]);
            for(unsigned j = 0; j < K; j++){
                if(shared_nei[j] == find_key) {
                    res_id = j;
                    break;
                }
            }
            if(res_id < K && shared_nei[res_id] == find_key){
                unsigned ori_dis = ((tid < K - 1 - i) ? i : (K - 1 - i)), my_dis = ((tid < K - 1 - i) ? (i + 1 + tid) : (tid + 1));
                if(nei_distance[nei0[ori_dis]][res_id] < nei_distance[bid][my_dis]){
                    atomicAdd(&detour_num[my_dis], 1);
                }
            }
        }
        __syncthreads();
    }
    bitonic_sort_id_by_detour(detour_num, nei0, K);

    if(tid== 0){
        nei1[0] = nei0[0];
        tmp_id = 1;
        ip_neighbors = 1;
        cur_nei = 0;
        cur_nei2 = 0;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        nei2[i] = 0;
    }
    __syncthreads();

    // pruning with inner product neighbor strategy; parallelized on the GPU
    while(ip_neighbors < K){
        for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
            tmp_val_sha[i] = Load(&values[(nei0[tmp_id]) * DIM + 4 * i]);
        }
        for(unsigned i = sparse_off[nei0[tmp_id]] + tid; i < sparse_off[nei0[tmp_id] + 1]; i += blockDim.x * blockDim.y){
            node_idx[i - sparse_off[nei0[tmp_id]]] = sparse_idx[i];
            node_val[i - sparse_off[nei0[tmp_id]]] = sparse_val[i];
        }
        for(unsigned i = bm25_off[nei0[tmp_id]] + tid; i < bm25_off[nei0[tmp_id] + 1]; i += blockDim.x * blockDim.y){
            node_idx_bm25[i - bm25_off[nei0[tmp_id]]] = bm25_idx[i];
            node_val_bm25[i - bm25_off[nei0[tmp_id]]] = bm25_val[i];
        }
        __syncthreads();
        unsigned i;
        for(i = threadIdx.y; i < ip_neighbors; i += blockDim.y){
            if(nei1[i] == 0xFFFFFFFF) continue;
            half2 val_res_ip;
            val_res_ip.x = 0.0; val_res_ip.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                half4 val2 = Load(&values[(nei1[i]) * DIM + j * 4]);
                val_res_ip = __hadd2(val_res_ip, __hmul2(tmp_val_sha[j].x, val2.x));
                val_res_ip = __hadd2(val_res_ip, __hmul2(tmp_val_sha[j].y, val2.y));
            }
            float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[nei1[i]], sparse_val+sparse_off[nei1[i]], sparse_off[nei1[i]+1]-sparse_off[nei1[i]], node_idx, node_val, sparse_off[nei0[tmp_id]+1]-sparse_off[nei0[tmp_id]], laneid)
                            + cal_bm25_dist(bm25_idx+bm25_off[nei1[i]], bm25_val+bm25_off[nei1[i]], bm25_off[nei1[i]+1]-bm25_off[nei1[i]], node_idx_bm25, node_val_bm25, bm25_off[nei0[tmp_id]+1]-bm25_off[nei0[tmp_id]], laneid, &cur_nei2, hash_table, flag_per_warp);
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){
                val_res_ip = __hadd2(val_res_ip, __shfl_down_sync(0xffffffff, val_res_ip, lane_mask));
                sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
            }
            if(laneid == 0){
                float res_ip = __half2float(__hadd(val_res_ip.x, val_res_ip.y))+sparse_dis;
                if(data_power[nei0[tmp_id]] < res_ip){
                    cur_nei = 1;
                }
                if(data_power[nei1[i]] < res_ip){
                    nei1[i] = 0xFFFFFFFF;
                }
                if(flag_per_warp[threadIdx.y] != 1){
                    cur_nei2 = 1;
                    flag_per_warp[threadIdx.y] = 0;
                }
            }
        }
        __syncthreads();
        if(tid == 0){
            if(cur_nei == 0){
                nei1[ip_neighbors] = nei0[tmp_id];
                ip_neighbors++;
            }
            // nei2 is reused store the keyword flag
            if(cur_nei2 == 0){
                nei2[ip_neighbors] = 1;
            }
            cur_nei = 0;
            cur_nei2 = 0;
            tmp_id++;
        }
        if(laneid == 0) {
            flag_per_warp[threadIdx.y] = 0;
        }
        __syncthreads();
        if(tmp_id >= K) break;
    }
    __syncthreads();
    
    __shared__ unsigned i_neis;
    if(tid == 0){
        unsigned count = 0;
        i_neis = 0;
        for(i_neis = 0; i_neis < ip_neighbors && count < d; i_neis++){
            if(nei1[i_neis] != 0xFFFFFFFF){
                nei0[count] = nei1[i_neis];
                count++;
            }
        }

        for(unsigned i = i_neis; i < ip_neighbors; i++){
            if(nei2[i] == 1 && nei1[i] != 0xFFFFFFFF && Keyword_edge[bid][0] < KEYEDGE){
                Keyword_edge[bid][Keyword_edge[bid][0] + 1] = nei1[i];
                Keyword_edge[bid][0]++;
            }
        }
    }
    __syncthreads();
    
    for(unsigned i = tid; i < d; i += blockDim.x * blockDim.y){
        unsigned tmp = atomicAdd(&reverse_num[nei0[i]], 1);
        if (tmp < d) {
            reverse_graph[nei0[i] * d * 2 + tmp * 2] = bid;
            reverse_graph[nei0[i] * d * 2 + tmp * 2 + 1] = detour_num[i];
        }
        else{
            if(reverse_graph[nei0[i] * d * 2 + (tmp%d) * 2 + 1] > detour_num[i]){
                reverse_graph[nei0[i] * d * 2 + (tmp%d) * 2] = bid;
                reverse_graph[nei0[i] * d * 2 + (tmp%d) * 2 + 1] = detour_num[i];
            }
        }
        new_list[bid][i] = nei0[i];
    }

}

// extend the neighbor lists with the reverse neighbors
__global__ void allan_poe_final_graph(unsigned* graph, unsigned* reverse_graph, unsigned d, unsigned final_degree){
    __shared__ unsigned nei_id[K];
    __shared__ unsigned reverse_nei[K];
    __shared__ unsigned reverse_nei_detour[K];
    __shared__ unsigned cur_num;
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;
    unsigned rever_num_cur = min(d, reverse_num[bid]);
    for(unsigned i = tid; i < rever_num_cur; i += blockDim.x * blockDim.y){
        reverse_nei[i] = reverse_graph[bid * d * 2 + i * 2];
        reverse_nei_detour[i] = reverse_graph[bid * d * 2 + i * 2 + 1];
    }
    for(unsigned i = tid; i < d; i += blockDim.x * blockDim.y){
        nei_id[i] = new_list[bid][i];
    }
    if(tid == 0) cur_num = 0;
    __syncthreads();
    bitonic_sort_id_by_detour(reverse_nei, reverse_nei_detour, rever_num_cur);
    for(unsigned i = tid + 1; i < rever_num_cur; i += blockDim.x * blockDim.y){
        if(reverse_nei[i] == reverse_nei[i - 1]){
            reverse_nei_detour[i] = 0xFFFFFFFF;
        }
    }
    for(unsigned i = tid; i < d/2; i += blockDim.x * blockDim.y){
        graph[bid * final_degree + i] = nei_id[i];
    }
    __syncthreads();
    bitonic_sort_id_by_detour(reverse_nei_detour, reverse_nei, rever_num_cur);
    if(rever_num_cur < d / 2){
        for(unsigned i = tid; i < rever_num_cur; i += blockDim.x * blockDim.y){
            bool flag = false;
            for(unsigned j = 0; j < d - rever_num_cur; j++){
                if(reverse_nei[i] == nei_id[j]) {
                    flag = true;
                    break;
                }
            }
            if(!flag){
                unsigned tmp_id = atomicAdd(&cur_num, 1);
                if(tmp_id < d/2)
                    graph[bid * final_degree + d/2 + tmp_id] = reverse_nei[i];
            }
        }
        __syncthreads();

        if(cur_num < d/2){
            for(unsigned i = tid; i < d/2 - cur_num; i += blockDim.x * blockDim.y){
                graph[bid * final_degree + d/2 + cur_num + i] = nei_id[i + d/2];
            }
        }
    }
    else{
        for(unsigned i = tid; cur_num < d/2 && i < rever_num_cur; i += blockDim.x * blockDim.y){
            bool flag = false;
            for(unsigned j = 0; j < d/2; j++){
                if(reverse_nei[i] == nei_id[j]) {
                    flag = true;
                    break;
                }
            }
            if(!flag){
                unsigned tmp_id = atomicAdd(&cur_num, 1);
                if(tmp_id < d/2)
                    graph[bid * final_degree + d/2 + tmp_id] = reverse_nei[i];
            }
        }
        __syncthreads();
        for(unsigned i = tid; i < d/2 - min(cur_num, d/2); i += blockDim.x * blockDim.y){
            graph[bid * final_degree + min(cur_num, d/2) + d/2 + i] = nei_id[i + d/2];
        }
    }
    __syncthreads();
    if(tid == 0) cur_num = 0;
    __syncthreads();
    for(unsigned i = tid; i < Keyword_edge[bid][0] && cur_num < KEYEDGEFINAL; i += blockDim.x * blockDim.y){
        bool flag = false;
        for(unsigned j = d/2; j < d; j++){
            if(graph[bid * final_degree + j] == Keyword_edge[bid][1 + i]) {
                flag = true;
                break;
            }
        }
        if(!flag){
            unsigned tmp_id = atomicAdd(&cur_num, 1);
            if(tmp_id < KEYEDGEFINAL - 1) graph[bid * final_degree + d + tmp_id + 1] = Keyword_edge[bid][1 + i];
        }
    }
    __syncthreads();
    if(tid == 0) {
        graph[bid * final_degree + d] = min(cur_num, KEYEDGEFINAL - 1);
    }
}


void build_index_impl(
    const std::string& dense_data_path,
    const std::string& sparse_data_path,
    const std::string& bm25_data_path,
    const std::string& output_graph_path
) {
    float* data_load = NULL;
    unsigned points_num, dim, non_zero_num, bm25_non_zero;
    unsigned* sparse_off = NULL, *sparse_idx = NULL, *bm25_off = NULL, *bm25_idx = NULL;
    float* sparse_val = NULL, *bm25_val = NULL;

    std::cout << "Loading dense data from " << dense_data_path << std::endl;
    load_data((char*)dense_data_path.c_str(), data_load, points_num, dim);
    
    load_sparse_data((char*)sparse_data_path.c_str(), sparse_off, sparse_idx, sparse_val, points_num+1, non_zero_num);
    load_sparse_data((char*)bm25_data_path.c_str(), bm25_off, bm25_idx, bm25_val, points_num+1, bm25_non_zero);

    std::cout << "Points: " << points_num << ", Dim: " << dim << std::endl;

    unsigned max_sparse_dim = 0, max_bm25_dim = 0;
    for(unsigned i = 0; i < points_num; i++){
        if(max_sparse_dim < sparse_off[i + 1] - sparse_off[i]) max_sparse_dim = sparse_off[i + 1] - sparse_off[i];
        if(max_bm25_dim < bm25_off[i + 1] - bm25_off[i]) max_bm25_dim = bm25_off[i + 1] - bm25_off[i];
    }
    
    cout << "Max sparse dim: " << max_sparse_dim << ", Max bm25 dim: " << max_bm25_dim << endl;
    for(unsigned i = 0; i < points_num; i++){
        for(unsigned j = 0; j < dim; j++){
            data_load[i * dim + j] *= 7;
        }
    }

    unsigned* graph_dev;
    cudaMalloc((void**)&graph_dev, points_num * K * sizeof(unsigned));
    
    unsigned* reverse_graph_dev;
    cudaMalloc((void**)&reverse_graph_dev, points_num * K * sizeof(unsigned));
    float* data_dev;
    cudaMalloc((void**)&data_dev, points_num * dim * sizeof(float));
    cudaMemcpy(data_dev, data_load, points_num * dim * sizeof(float), cudaMemcpyHostToDevice);

    half* data_half_dev;
    cudaMalloc((void**)&data_half_dev, points_num * dim * sizeof(half));

    unsigned* sparse_off_dev, * sparse_idx_dev, *bm25_off_dev, *bm25_idx_dev;
    float* sparse_val_dev, *bm25_val_dev;
    cudaMalloc((void**)&sparse_off_dev, (points_num+1) * sizeof(unsigned));
    cudaMemcpy(sparse_off_dev, sparse_off, (points_num+1) * sizeof(unsigned), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&sparse_idx_dev, non_zero_num * sizeof(unsigned));
    cudaMemcpy(sparse_idx_dev, sparse_idx, non_zero_num * sizeof(unsigned), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&sparse_val_dev, non_zero_num * sizeof(float));
    cudaMemcpy(sparse_val_dev, sparse_val, non_zero_num * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&bm25_off_dev, (points_num+1) * sizeof(unsigned));
    cudaMemcpy(bm25_off_dev, bm25_off, (points_num+1) * sizeof(unsigned), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&bm25_idx_dev, bm25_non_zero * sizeof(unsigned));
    cudaMemcpy(bm25_idx_dev, bm25_idx, bm25_non_zero * sizeof(unsigned), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&bm25_val_dev, bm25_non_zero * sizeof(float));
    cudaMemcpy(bm25_val_dev, bm25_val, bm25_non_zero * sizeof(float), cudaMemcpyHostToDevice);

    float* data_power_dev;
    cudaMalloc((void**)&data_power_dev, points_num * sizeof(float));
    cudaMemset(data_power_dev, 0, points_num * sizeof(float));
    checkCudaErrors(cudaGetLastError());

    dim3 grid(points_num, 1, 1);
    dim3 block(32, MAX_P / 32, 1);
    dim3 block2(32, 16, 1);
    dim3 block3(32, 3, 1);
    dim3 block4(32, 3, 1);
    unsigned all_it = 25; // #iterations of NN-Descent
    dim3 grid_s(points_num, 1, 1);
    dim3 block_s(32, 3, 1);
    dim3 block_m(32, 16, 1);
    unsigned degree = 32; // the degree of the graph index is set to 32
    auto start = std::chrono::high_resolution_clock::now();
    initialize_graph<<<points_num, 32>>>(graph_dev, points_num, data_dev);
    f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num);
    cal_power<<<points_num, 32>>>(data_half_dev, data_power_dev, dim, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev, points_num);

    for(unsigned it = 0; it < all_it; it++){
        nn_descent_gpu<<<grid, block>>>(graph_dev, reverse_graph_dev, it, data_half_dev, points_num, all_it, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev, data_power_dev);
        if(it < all_it - 1) reset_reverse_num<<<1000, 1024>>>(points_num);
    }
    merge_reverse_plus<<<grid, block3>>>(graph_dev, reverse_graph_dev);
    reset_reverse_num<<<1000, 1024>>>(points_num);

    // pruning using our proposed strategy
    allan_poe<<<grid_s, block_m>>>(graph_dev, degree, reverse_graph_dev, data_half_dev, data_power_dev, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev);
    allan_poe_final_graph<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, degree, degree + KEYEDGEFINAL);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Indexing time: " << duration.count() << "s" << std::endl;

    cudaMemcpy(res_graph, graph_dev, points_num * K * sizeof(unsigned), cudaMemcpyDeviceToHost);
    vector<float> data_p(points_num);
    cudaMemcpy(data_p.data(), data_power_dev, points_num * sizeof(float), cudaMemcpyDeviceToHost);
    
    // select entry points
    vector<pair<float, unsigned>> dp;
    for(unsigned i = 0; i < points_num; i++){
        dp.push_back(pair<float, unsigned>(data_p[i], i));
    }
    sort(dp.begin(), dp.end());
    vector<unsigned> eps_vec;
    for(unsigned i = 0; i < degree; i++){
        eps_vec.push_back(dp[i].second);
    }

    //save allan_poe
    unsigned tmp_deg = degree + KEYEDGEFINAL;
    std::ofstream out((char*)output_graph_path.c_str(), std::ios::binary | std::ios::out);
    out.write((char *)&(tmp_deg), sizeof(unsigned));
    unsigned n_ep=eps_vec.size();
    out.write((char *)&n_ep, sizeof(unsigned));
    out.write((char *)eps_vec.data(), n_ep*sizeof(unsigned));
    for (unsigned i = 0; i < points_num; i++) {
        unsigned GK = (unsigned) (degree+KEYEDGEFINAL);
        out.write((char*)&GK, sizeof(unsigned));
        out.write((char*)&res_graph[i * (degree+KEYEDGEFINAL)], GK * sizeof(unsigned));
    }
    out.close();

}