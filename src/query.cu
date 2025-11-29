#include <float.h>

#include "hybrid_search.cuh"
#include "utils.cuh"

#define K 32 // graph index degree
#define MAX_ITER 1000 // maximal number of iterations during greedy search

#define MAX_CAND_POOL 512

#define INF_DIS 1000000.0f

#define MAX_SPARSE 550
#define MAX_BM25 780

#define HASH_RESET 4

#define KEYWORD 32
#define KEYEDGE 8

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

// load the graph index
void load_graph(const char *filename, std::vector<unsigned>& final_graph_, vector<unsigned>& ent_pts) {
    // std::vector<std::vector<unsigned>> final_graph_;
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    unsigned k;
    in.read((char*)&k,4);
    unsigned ent_num;
    in.read((char*)&ent_num, 4);
    in.seekg(0,std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    size_t num = (fsize - (ent_num + 1) * 4) / ((size_t)k + 1) / 4;
    in.seekg(0,std::ios::beg);
    cout << "Degree: " << k << "," << "num: " << num << endl;
    
    ent_pts.resize(ent_num);
    in.seekg(8,std::ios::cur);
    in.read((char*)(ent_pts.data()), ent_num * sizeof(unsigned));
    
    final_graph_.resize(num * k);
    for(size_t i = 0; i < num; i++){
        in.seekg(4,std::ios::cur);
        // final_graph_[i].resize(k);
        // final_graph_[i].reserve(k);
        in.read((char*)(&final_graph_[i * k]), k * sizeof(unsigned));
    }
    in.close();
}

// load the groundtruth file
void load_txt(char* filename, std::vector<std::vector<unsigned>>& results, unsigned query_num){
    std::ifstream file(filename); 
    std::string line;
    unsigned count = 0;
    unsigned all_num = 0;
    if (file.is_open()) {
        while (getline(file, line)) {
            std::istringstream iss(line);
            unsigned number;
            while (iss >> number) {
                results[count].push_back(number);
                all_num += 1;
            }
            iss.clear(); 
            count++;
            if(count >= query_num) break;
        }
        file.close();
    } else {
        std::cout << "open file error" << std::endl;
    }
}

// load the knowledge graph
void load_kg_graph(char* filename, unsigned* &graph_off, unsigned* &graph_idx, unsigned& num){
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    unsigned non_zero_num;
    in.read((char*)(&num),4);
    in.read((char*)(&non_zero_num),4);

    cout << "Entity num: " << num << ", Non zero num: " << non_zero_num << endl;

    graph_off = new unsigned[num];
    in.read((char*)(graph_off),num*4);

    graph_idx = new unsigned[non_zero_num];
    // sparse_val = new float[non_zero_num];

    in.read((char*)(graph_idx),non_zero_num*4);
    // in.read((char*)(sparse_val),non_zero_num*4);

    in.close();
}

// calculate the nDCG value
void cal_precision(vector<vector<unsigned>>& v1,vector<vector<unsigned>>& v2){
    vector<unsigned> v;
    float all = 0.0;
    float nDCG = 0.0;
    for(int i = 0; i < v1.size(); i++){
        float IDCG = 0.0;
        for(unsigned j = 0; j < v2[i].size(); j++){
            IDCG += (1 / log2((float(j + 2))));
        }
        vector<unsigned> tmp_v;
        for(unsigned j = 0; j < v1[i].size(); j++){
            tmp_v.push_back(v1[i][j]);
        }
        sort(tmp_v.begin(),tmp_v.end());   
        sort(v2[i].begin(),v2[i].end());   
        set_intersection(tmp_v.begin(),tmp_v.end(),v2[i].begin(),v2[i].end(),back_inserter(v));//求交集 
        all += ((float)v.size() / v2[i].size());
        // cal dcg
        float DCG = 0.0;
        for(auto x : v){
            unsigned it = 0;
            while(it < v1[i].size()){
                if(x == v1[i][it]){
                    DCG += (1 / log2(float(it + 2)));
                    break; 
                }
                it++;
            }
        }
        nDCG += (DCG/IDCG);
        v.clear();
    }
    
    cout << "nDCG@10: " << nDCG / v1.size() << endl;
}

// compute hybrid distance on CPU
void compute_hybrid_dis(float* a, float* b, unsigned dim, unsigned* idx1, float* val1, unsigned len1, unsigned* idx2, float* val2, unsigned len2, unsigned* bm25_idx1, float* bm25_val1, unsigned bm25_len1, unsigned* bm25_idx2, float* bm25_val2, unsigned bm25_len2, vector<float>&res_vec1, vector<float>&res_vec2, vector<float>&res_vec3){
    float res_dis = 0.0;
    for(unsigned i = 0; i < dim; i++){
        res_dis += (a[i] * b[i]);
    }
    res_vec1.push_back(-res_dis);
    res_dis = 0.0;
    unsigned p1 = 0, p2 = 0;
    while(p1 < len1 && p2 < len2){
        if(idx1[p1] < idx2[p2]) p1++;
        else if(idx1[p1] > idx2[p2]) p2++;
        else{
            res_dis += val1[p1] * val2[p2];
            p1++;
            p2++;
        }
    }
    res_vec2.push_back(-res_dis);
    res_dis = 0.0;
    unsigned bm25_p1 = 0, bm25_p2 = 0;
    while(bm25_p1 < bm25_len1 && bm25_p2 < bm25_len2){
        if(bm25_idx1[bm25_p1] < bm25_idx2[bm25_p2]) bm25_p1++;
        else if(bm25_idx1[bm25_p1] > bm25_idx2[bm25_p2]) bm25_p2++;
        else{
            res_dis += bm25_val1[bm25_p1] * bm25_val2[bm25_p2];
            bm25_p1++;
            bm25_p2++;
        }
    }
    res_vec3.push_back(-res_dis);
}

// merge two neighbor lists according to the distances
__device__ void merge_top(unsigned* arr1, unsigned* arr2, float* arr1_val, float* arr2_val, unsigned tid, unsigned TOPM){
    unsigned res_id_vec[(MAX_CAND_POOL + K+6*32-1)/ (6*32)] = {0};
    float val_vec[(MAX_CAND_POOL + K+6*32-1)/ (6*32)];
    unsigned id_reg_vec[(MAX_CAND_POOL + K+6*32-1)/ (6*32)];
    for(unsigned i = 0; i < (TOPM + K+6*32-1)/ (6*32); i ++){
        if(i * blockDim.x * blockDim.y + tid < TOPM){
            val_vec[i] = arr1_val[i * blockDim.x * blockDim.y + tid];
            id_reg_vec[i] = arr1[i * blockDim.x * blockDim.y + tid];
            unsigned tmp = K;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr2_val[res_id_vec[i] + halfsize];
                res_id_vec[i] += ((cand <= val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id_vec[i] += (arr2_val[res_id_vec[i]] <= val_vec[i]);
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid);
        }
        else if(i * blockDim.x * blockDim.y + tid < TOPM + K){
            val_vec[i] = arr2_val[i * blockDim.x * blockDim.y + tid - TOPM];
            id_reg_vec[i] = arr2[i * blockDim.x * blockDim.y + tid - TOPM];
            unsigned tmp = TOPM;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr1_val[res_id_vec[i] + halfsize];
                res_id_vec[i] += ((cand < val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id_vec[i] += (arr1_val[res_id_vec[i]] < val_vec[i]);
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid - TOPM);
        }
        else{
            res_id_vec[i] = TOPM;
        }
    }
    __syncthreads();
    for(unsigned i = 0; i < (TOPM + K + 6 * 32 - 1)/ (6 * 32); i ++){
        if(res_id_vec[i] < TOPM){
            arr1[res_id_vec[i]] = id_reg_vec[i];
            arr1_val[res_id_vec[i]] = val_vec[i];
        }
    }
    __syncthreads();
}

// merge two neighbor lists according to the distances while visiting the keywords
__device__ void merge_top_keyword(unsigned* arr1, unsigned* arr2, float* arr1_val, float* arr2_val, unsigned tid, unsigned* keyword_Cand, float* keyword_Cand_dis, unsigned* keyword_num, unsigned TOPM){
    unsigned res_id_vec[(MAX_CAND_POOL + K+6*32-1)/ (6*32)] = {0};
    float val_vec[(MAX_CAND_POOL + K+6*32-1)/ (6*32)];
    unsigned id_reg_vec[(MAX_CAND_POOL + K+6*32-1)/ (6*32)];
    for(unsigned i = 0; i < (TOPM + K+6*32-1)/ (6*32); i ++){
        if(i * blockDim.x * blockDim.y + tid < TOPM){
            val_vec[i] = arr1_val[i * blockDim.x * blockDim.y + tid];
            id_reg_vec[i] = arr1[i * blockDim.x * blockDim.y + tid];
            unsigned tmp = K;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr2_val[res_id_vec[i] + halfsize];
                res_id_vec[i] += ((cand <= val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id_vec[i] += (arr2_val[res_id_vec[i]] <= val_vec[i]);
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid);
            if(res_id_vec[i] >= TOPM && (id_reg_vec[i] & 0x40000000 != 0)) {
                unsigned tt_id = atomicAdd(keyword_num, 1);
                keyword_Cand[tt_id % KEYWORD] = (id_reg_vec[i] & 0x3FFFFFFF);
                keyword_Cand_dis[tt_id % KEYWORD] = val_vec[i];
            }
        }
        else if(i * blockDim.x * blockDim.y + tid < TOPM + K){
            val_vec[i] = arr2_val[i * blockDim.x * blockDim.y + tid - TOPM];
            id_reg_vec[i] = arr2[i * blockDim.x * blockDim.y + tid - TOPM];
            unsigned tmp = TOPM;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr1_val[res_id_vec[i] + halfsize];
                res_id_vec[i] += ((cand < val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id_vec[i] += (arr1_val[res_id_vec[i]] < val_vec[i]);
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid - TOPM);
        }
        else{
            res_id_vec[i] = TOPM;
        }
    }
    __syncthreads();
    for(unsigned i = 0; i < (TOPM + K + 6 * 32 - 1)/ (6 * 32); i ++){
        if(res_id_vec[i] < TOPM){
            arr1[res_id_vec[i]] = id_reg_vec[i];
            arr1_val[res_id_vec[i]] = val_vec[i];
        }
    }
    __syncthreads();
}

__global__ void search_with_dense(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts, unsigned TOPM){
    __shared__ unsigned top_M_Cand[MAX_CAND_POOL + K];
    __shared__ float top_M_Cand_dis[MAX_CAND_POOL + K];
    __shared__ half4 tmp_val_sha[DIM / 4];

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        // top_M_Cand[TOPM + tid] = graph[bid * K + i];
        top_M_Cand[TOPM + tid] = ent_pts[i];
        hash_insert(hash_table, top_M_Cand[TOPM + tid]);
    }
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
        }
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }
        if(laneid == 0){
            top_M_Cand_dis[TOPM + i] = 1.0 - __half2float(__hadd(val_res.x, val_res.y));
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
    for(unsigned i = tid; i < min(K, TOPM); i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
    }
    __syncthreads();

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + K; j += blockDim.x * blockDim.y){
                hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
            }
        }
        __syncthreads();
        
        if(tid < 32){
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
                unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                if(ballot_res > 0){
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = top_M_Cand[j];
                        top_M_Cand[j] |= 0x80000000;
                    }
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            break;
        }
        
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * (K + KEYEDGE) + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
        }
        __syncthreads();
        
        
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                }
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = 1.0 - __half2float(__hadd(val_res.x, val_res.y));
                }
            }
        }
        __syncthreads();
        
        bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
        merge_top(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid, TOPM);
        
    }
    __syncthreads();
    
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
}

__global__ void search_with_three_paths(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts, unsigned* sparse_off, unsigned *sparse_idx, float* sparse_val, unsigned *bm25_off, unsigned *bm25_idx, float *bm25_val, unsigned* sparse_off_query, unsigned *sparse_idx_query, float* sparse_val_query, unsigned *bm25_off_query, unsigned *bm25_idx_query, float *bm25_val_query, unsigned TOPM){
    __shared__ unsigned top_M_Cand[MAX_CAND_POOL + K];
    __shared__ float top_M_Cand_dis[MAX_CAND_POOL + K];

    __shared__ half4 tmp_val_sha[DIM / 4];

    __shared__ unsigned node_idx[MAX_SPARSE];
    __shared__ float node_val[MAX_SPARSE];
    __shared__ unsigned node_idx_bm25[MAX_BM25];
    __shared__ float node_val_bm25[MAX_BM25];

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];


    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        // top_M_Cand[TOPM + tid] = graph[bid * K + i];
        top_M_Cand[TOPM + tid] = ent_pts[i];
        hash_insert(hash_table, top_M_Cand[TOPM + tid]);
    }
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    for(unsigned i = sparse_off_query[bid] + tid; i < sparse_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx[i - sparse_off_query[bid]] = sparse_idx_query[i];
        node_val[i - sparse_off_query[bid]] = sparse_val_query[i];
    }
    for(unsigned i = bm25_off_query[bid] + tid; i < bm25_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx_bm25[i - bm25_off_query[bid]] = bm25_idx_query[i];
        node_val_bm25[i - bm25_off_query[bid]] = bm25_val_query[i];
    }
    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
        }
        float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + i]], sparse_val+sparse_off[top_M_Cand[TOPM + i]], sparse_off[top_M_Cand[TOPM + i]+1]-sparse_off[top_M_Cand[TOPM + i]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid)
                                + cal_sparse_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + i]], bm25_val+bm25_off[top_M_Cand[TOPM + i]], bm25_off[top_M_Cand[TOPM + i]+1]-bm25_off[top_M_Cand[TOPM + i]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid);
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
        }
        if(laneid == 0){
            // top_M_Cand_dis[TOPM + i] = __half2float(__hadd(val_res.x, val_res.y));
            top_M_Cand_dis[TOPM + i] = 1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
    for(unsigned i = tid; i < min(K, TOPM); i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
    }
    __syncthreads();

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + K; j += blockDim.x * blockDim.y){
                hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
            }
        }
        __syncthreads();

        if(tid < 32){
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
                unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                if(ballot_res > 0){
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = top_M_Cand[j];
                        top_M_Cand[j] |= 0x80000000;
                    }
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            break;
        }
        
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * (K + KEYEDGE) + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
        }
        __syncthreads();

        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                }
                float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + j]], sparse_val+sparse_off[top_M_Cand[TOPM + j]], sparse_off[top_M_Cand[TOPM + j]+1]-sparse_off[top_M_Cand[TOPM + j]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid)
                                + cal_sparse_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + j]], bm25_val+bm25_off[top_M_Cand[TOPM + j]], bm25_off[top_M_Cand[TOPM + j]+1]-bm25_off[top_M_Cand[TOPM + j]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid);
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = 1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
                }
            }
        }
        __syncthreads();
        
        // merge into top_M
        bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
        merge_top(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid, TOPM);
    }
    __syncthreads();
    
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
}

__global__ void search_with_dense_and_sparse(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts, unsigned* sparse_off, unsigned *sparse_idx, float* sparse_val, unsigned *bm25_off, unsigned *bm25_idx, float *bm25_val, unsigned* sparse_off_query, unsigned *sparse_idx_query, float* sparse_val_query, unsigned *bm25_off_query, unsigned *bm25_idx_query, float *bm25_val_query, unsigned TOPM){
    __shared__ unsigned top_M_Cand[MAX_CAND_POOL + K];
    __shared__ float top_M_Cand_dis[MAX_CAND_POOL + K];

    __shared__ half4 tmp_val_sha[DIM / 4];

    __shared__ unsigned node_idx[MAX_SPARSE];
    __shared__ float node_val[MAX_SPARSE];

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        // top_M_Cand[TOPM + tid] = graph[bid * K + i];
        top_M_Cand[TOPM + tid] = ent_pts[i];
        hash_insert(hash_table, top_M_Cand[TOPM + tid]);
    }
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    for(unsigned i = sparse_off_query[bid] + tid; i < sparse_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx[i - sparse_off_query[bid]] = sparse_idx_query[i];
        node_val[i - sparse_off_query[bid]] = sparse_val_query[i];
    }

    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
        }
        float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + i]], sparse_val+sparse_off[top_M_Cand[TOPM + i]], sparse_off[top_M_Cand[TOPM + i]+1]-sparse_off[top_M_Cand[TOPM + i]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid);
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
        }
        if(laneid == 0){
            top_M_Cand_dis[TOPM + i] = 1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
    for(unsigned i = tid; i < min(K, TOPM); i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
    }
    __syncthreads();

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + K; j += blockDim.x * blockDim.y){
                hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
            }
        }
        __syncthreads();

        if(tid < 32){
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
                unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                if(ballot_res > 0){
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = top_M_Cand[j];
                        top_M_Cand[j] |= 0x80000000;
                    }
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            break;
        }
        
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * (K + KEYEDGE) + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
        }
        __syncthreads();

        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                }
                float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + j]], sparse_val+sparse_off[top_M_Cand[TOPM + j]], sparse_off[top_M_Cand[TOPM + j]+1]-sparse_off[top_M_Cand[TOPM + j]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid);
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = 1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
                }
            }
        }
        __syncthreads();
        
        // merge into top_M
        bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
        merge_top(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid, TOPM);
    }
    __syncthreads();
    
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
}

__global__ void search_with_bm25_and_dense(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts, unsigned* sparse_off, unsigned *sparse_idx, float* sparse_val, unsigned *bm25_off, unsigned *bm25_idx, float *bm25_val, unsigned* sparse_off_query, unsigned *sparse_idx_query, float* sparse_val_query, unsigned *bm25_off_query, unsigned *bm25_idx_query, float *bm25_val_query, unsigned TOPM){
    __shared__ unsigned top_M_Cand[MAX_CAND_POOL + K];
    __shared__ float top_M_Cand_dis[MAX_CAND_POOL + K];

    __shared__ half4 tmp_val_sha[DIM / 4];

    __shared__ unsigned node_idx_bm25[MAX_BM25];
    __shared__ float node_val_bm25[MAX_BM25];

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        // top_M_Cand[TOPM + tid] = graph[bid * K + i];
        top_M_Cand[TOPM + tid] = ent_pts[i];
        hash_insert(hash_table, top_M_Cand[TOPM + tid]);
    }
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    for(unsigned i = bm25_off_query[bid] + tid; i < bm25_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx_bm25[i - bm25_off_query[bid]] = bm25_idx_query[i];
        node_val_bm25[i - bm25_off_query[bid]] = bm25_val_query[i];
    }
    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
        }
        float sparse_dis = cal_sparse_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + i]], bm25_val+bm25_off[top_M_Cand[TOPM + i]], bm25_off[top_M_Cand[TOPM + i]+1]-bm25_off[top_M_Cand[TOPM + i]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid);
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
        }
        if(laneid == 0){
            top_M_Cand_dis[TOPM + i] = 1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
    for(unsigned i = tid; i < min(K, TOPM); i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
    }
    __syncthreads();

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + K; j += blockDim.x * blockDim.y){
                hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
            }
        }
        __syncthreads();

        if(tid < 32){
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
                unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                if(ballot_res > 0){
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = top_M_Cand[j];
                        top_M_Cand[j] |= 0x80000000;
                    }
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            break;
        }
        
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * (K + KEYEDGE) + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
        }
        __syncthreads();
        
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                }
                float sparse_dis = cal_sparse_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + j]], bm25_val+bm25_off[top_M_Cand[TOPM + j]], bm25_off[top_M_Cand[TOPM + j]+1]-bm25_off[top_M_Cand[TOPM + j]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid);
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = 1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
                }
            }
        }
        __syncthreads();
        
        // merge into top_M
        bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
        merge_top(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid, TOPM);
    }
    __syncthreads();
    
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
}

__global__ void search_with_sparse_and_bm25(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts, unsigned* sparse_off, unsigned *sparse_idx, float* sparse_val, unsigned *bm25_off, unsigned *bm25_idx, float *bm25_val, unsigned* sparse_off_query, unsigned *sparse_idx_query, float* sparse_val_query, unsigned *bm25_off_query, unsigned *bm25_idx_query, float *bm25_val_query, unsigned TOPM){
    __shared__ unsigned top_M_Cand[MAX_CAND_POOL + K];
    __shared__ float top_M_Cand_dis[MAX_CAND_POOL + K];

    __shared__ unsigned node_idx[MAX_SPARSE];
    __shared__ float node_val[MAX_SPARSE];
    __shared__ unsigned node_idx_bm25[MAX_BM25];
    __shared__ float node_val_bm25[MAX_BM25];

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        // top_M_Cand[TOPM + tid] = graph[bid * K + i];
        top_M_Cand[TOPM + tid] = ent_pts[i];
        hash_insert(hash_table, top_M_Cand[TOPM + tid]);
    }
    for(unsigned i = sparse_off_query[bid] + tid; i < sparse_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx[i - sparse_off_query[bid]] = sparse_idx_query[i];
        node_val[i - sparse_off_query[bid]] = sparse_val_query[i];
    }
    for(unsigned i = bm25_off_query[bid] + tid; i < bm25_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx_bm25[i - bm25_off_query[bid]] = bm25_idx_query[i];
        node_val_bm25[i - bm25_off_query[bid]] = bm25_val_query[i];
    }
    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + i]], sparse_val+sparse_off[top_M_Cand[TOPM + i]], sparse_off[top_M_Cand[TOPM + i]+1]-sparse_off[top_M_Cand[TOPM + i]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid)
                                + cal_sparse_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + i]], bm25_val+bm25_off[top_M_Cand[TOPM + i]], bm25_off[top_M_Cand[TOPM + i]+1]-bm25_off[top_M_Cand[TOPM + i]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid);
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){ 
            sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
        }
        if(laneid == 0){
            top_M_Cand_dis[TOPM + i] = 1.0 - sparse_dis;
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
    for(unsigned i = tid; i < min(K, TOPM); i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
    }
    __syncthreads();

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + K; j += blockDim.x * blockDim.y){
                hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
            }
        }
        __syncthreads();

        if(tid < 32){
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
                unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                if(ballot_res > 0){
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = top_M_Cand[j];
                        top_M_Cand[j] |= 0x80000000;
                    }
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            break;
        }
        
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * (K + KEYEDGE) + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
        }
        __syncthreads();
        
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + j]], sparse_val+sparse_off[top_M_Cand[TOPM + j]], sparse_off[top_M_Cand[TOPM + j]+1]-sparse_off[top_M_Cand[TOPM + j]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid)
                                + cal_sparse_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + j]], bm25_val+bm25_off[top_M_Cand[TOPM + j]], bm25_off[top_M_Cand[TOPM + j]+1]-bm25_off[top_M_Cand[TOPM + j]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid);
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){
                    sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = 1.0 - sparse_dis;
                }
            }
        }
        __syncthreads();
        
        // merge into top_M
        bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
        merge_top(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid, TOPM);
    }
    __syncthreads();
    
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
}

__global__ void search_with_sparse(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts, unsigned* sparse_off, unsigned *sparse_idx, float* sparse_val, unsigned *bm25_off, unsigned *bm25_idx, float *bm25_val, unsigned* sparse_off_query, unsigned *sparse_idx_query, float* sparse_val_query, unsigned *bm25_off_query, unsigned *bm25_idx_query, float *bm25_val_query, unsigned TOPM){
    __shared__ unsigned top_M_Cand[MAX_CAND_POOL + K];
    __shared__ float top_M_Cand_dis[MAX_CAND_POOL + K];

    __shared__ unsigned node_idx[MAX_SPARSE];
    __shared__ float node_val[MAX_SPARSE];

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        top_M_Cand[TOPM + tid] = ent_pts[i];
        hash_insert(hash_table, top_M_Cand[TOPM + tid]);
    }

    for(unsigned i = sparse_off_query[bid] + tid; i < sparse_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx[i - sparse_off_query[bid]] = sparse_idx_query[i];
        node_val[i - sparse_off_query[bid]] = sparse_val_query[i];
    }

    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + i]], sparse_val+sparse_off[top_M_Cand[TOPM + i]], sparse_off[top_M_Cand[TOPM + i]+1]-sparse_off[top_M_Cand[TOPM + i]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid);
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){
            sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
        }
        if(laneid == 0){
            top_M_Cand_dis[TOPM + i] = 1.0 - sparse_dis;
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
    for(unsigned i = tid; i < min(K, TOPM); i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
    }
    __syncthreads();

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + K; j += blockDim.x * blockDim.y){
                hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
            }
        }
        __syncthreads();     

        if(tid < 32){
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
                unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                if(ballot_res > 0){
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = top_M_Cand[j];
                        top_M_Cand[j] |= 0x80000000;
                    }
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            break;
        }
        
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * (K + KEYEDGE) + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
        }
        __syncthreads();
        
        
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + j]], sparse_val+sparse_off[top_M_Cand[TOPM + j]], sparse_off[top_M_Cand[TOPM + j]+1]-sparse_off[top_M_Cand[TOPM + j]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid);
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){
                    sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = 1.0 - sparse_dis;
                }
            }
        }
        __syncthreads();
        
        // merge into top_M
        bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
        merge_top(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid, TOPM);
    }
    __syncthreads();
    
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
}

__global__ void search_with_bm25(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts, unsigned* sparse_off, unsigned *sparse_idx, float* sparse_val, unsigned *bm25_off, unsigned *bm25_idx, float *bm25_val, unsigned* sparse_off_query, unsigned *sparse_idx_query, float* sparse_val_query, unsigned *bm25_off_query, unsigned *bm25_idx_query, float *bm25_val_query, unsigned TOPM){
    __shared__ unsigned top_M_Cand[MAX_CAND_POOL + K];
    __shared__ float top_M_Cand_dis[MAX_CAND_POOL + K];

    __shared__ unsigned node_idx[MAX_SPARSE];
    __shared__ float node_val[MAX_SPARSE];
    __shared__ unsigned node_idx_bm25[MAX_BM25];
    __shared__ float node_val_bm25[MAX_BM25];

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        // top_M_Cand[TOPM + tid] = graph[bid * K + i];
        top_M_Cand[TOPM + tid] = ent_pts[i];
        hash_insert(hash_table, top_M_Cand[TOPM + tid]);
    }

    for(unsigned i = bm25_off_query[bid] + tid; i < bm25_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx_bm25[i - bm25_off_query[bid]] = bm25_idx_query[i];
        node_val_bm25[i - bm25_off_query[bid]] = bm25_val_query[i];
    }
    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        float sparse_dis = cal_sparse_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + i]], bm25_val+bm25_off[top_M_Cand[TOPM + i]], bm25_off[top_M_Cand[TOPM + i]+1]-bm25_off[top_M_Cand[TOPM + i]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid);
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){
            sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
        }
        if(laneid == 0){
            top_M_Cand_dis[TOPM + i] = 1.0 - sparse_dis;
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
    for(unsigned i = tid; i < min(K, TOPM); i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
    }
    __syncthreads();

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + K; j += blockDim.x * blockDim.y){
                hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
            }
        }
        __syncthreads();     

        if(tid < 32){
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
                unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                if(ballot_res > 0){
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = top_M_Cand[j];
                        top_M_Cand[j] |= 0x80000000;
                    }
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            // if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
            break;
        }
        
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * (K + KEYEDGE) + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
        }
        __syncthreads();
        
        
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                float sparse_dis = cal_sparse_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + j]], bm25_val+bm25_off[top_M_Cand[TOPM + j]], bm25_off[top_M_Cand[TOPM + j]+1]-bm25_off[top_M_Cand[TOPM + j]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid);
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){
                    sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = 1.0 - sparse_dis;
                }
            }
        }
        __syncthreads();
        
        // merge into top_M
        bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
        merge_top(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid, TOPM);
    }
    __syncthreads();
    
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }
}

// keywords as filters
__global__ void search_with_keyword(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts, unsigned* sparse_off, unsigned *sparse_idx, float* sparse_val, unsigned *bm25_off, unsigned *bm25_idx, float *bm25_val, unsigned* sparse_off_query, unsigned *sparse_idx_query, float* sparse_val_query, unsigned *bm25_off_query, unsigned *bm25_idx_query, float *bm25_val_query, unsigned TOPM){
    __shared__ unsigned top_M_Cand[MAX_CAND_POOL + K];
    __shared__ float top_M_Cand_dis[MAX_CAND_POOL + K];

    __shared__ unsigned keyword_Cand[KEYWORD];
    __shared__ float keyword_Cand_dis[KEYWORD];

    __shared__ half4 tmp_val_sha[DIM / 4];

    __shared__ unsigned node_idx[MAX_SPARSE];
    __shared__ float node_val[MAX_SPARSE];
    __shared__ unsigned node_idx_bm25[MAX_BM25];
    __shared__ float node_val_bm25[MAX_BM25];

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    __shared__ unsigned key_hit[6];
    __shared__ unsigned keyword_num;
    __shared__ unsigned keyword_tmp;

    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;
    if(tid == 0) {
        keyword_num = 0;
        keyword_tmp = 0;
    }
    for(unsigned i = tid; i < 6; i+= blockDim.x * blockDim.y){
        key_hit[i] = 0;
    }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        // top_M_Cand[TOPM + tid] = graph[bid * K + i];
        top_M_Cand[TOPM + tid] = ent_pts[i];
        hash_insert(hash_table, top_M_Cand[TOPM + tid]);
    }
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    for(unsigned i = sparse_off_query[bid] + tid; i < sparse_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx[i - sparse_off_query[bid]] = sparse_idx_query[i];
        node_val[i - sparse_off_query[bid]] = sparse_val_query[i];
    }
    for(unsigned i = bm25_off_query[bid] + tid; i < bm25_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx_bm25[i - bm25_off_query[bid]] = bm25_idx_query[i];
        node_val_bm25[i - bm25_off_query[bid]] = bm25_val_query[i];
    }
    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
        }
        float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + i]], sparse_val+sparse_off[top_M_Cand[TOPM + i]], sparse_off[top_M_Cand[TOPM + i]+1]-sparse_off[top_M_Cand[TOPM + i]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid)
                               + cal_bm25_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + i]], bm25_val+bm25_off[top_M_Cand[TOPM + i]], bm25_off[top_M_Cand[TOPM + i]+1]-bm25_off[top_M_Cand[TOPM + i]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid, &key_hit[threadIdx.y]);
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
        }
        if(laneid == 0){
            // top_M_Cand_dis[TOPM + i] = __half2float(__hadd(val_res.x, val_res.y));
            top_M_Cand_dis[TOPM + i] = 1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
            if(key_hit[threadIdx.y] > 0) {
                top_M_Cand[TOPM + i] |= 0x40000000;
                key_hit[threadIdx.y] = 0;
            }
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
    for(unsigned i = tid; i < min(K, TOPM); i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
    }
    __syncthreads();

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        // if(bid == 0 && tid == 0) printf("%d\n", i);
        if(tid == 0){
            keyword_tmp = 0;
        }
        
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + K; j += blockDim.x * blockDim.y){
                hash_insert(hash_table, (top_M_Cand[j] & 0x3FFFFFFF));
            }
        }
        __syncthreads();

        if(tid < 32){
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
                unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                if(ballot_res > 0){
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = (top_M_Cand[j]);
                        top_M_Cand[j] |= 0x80000000;
                    }
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            // if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
            break;
        }
        
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[(to_explore & 0x3FFFFFFF) * (K + KEYEDGE) + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
        }
        __syncthreads();
        // if((to_explore & 0x40000000) != 0){
            for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
                if(top_M_Cand_dis[TOPM + j] > 1.0){
                    unsigned tmp_id = atomicAdd(&keyword_tmp, 1);
                    if(tmp_id < graph[(to_explore & 0x3FFFFFFF) * (K + KEYEDGE) + K]){
                        unsigned to_append = graph[(to_explore & 0x3FFFFFFF) * (K + KEYEDGE) + K + tmp_id + 1];
                        if(hash_insert(hash_table, to_append) != 0){
                            top_M_Cand[TOPM + j] = to_append;
                            top_M_Cand_dis[TOPM + j] = 0.0;
                            // printf("A");
                        }
                    }
                }
            }
        // }
        
        __syncthreads();
        
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                }
                float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + j]], sparse_val+sparse_off[top_M_Cand[TOPM + j]], sparse_off[top_M_Cand[TOPM + j]+1]-sparse_off[top_M_Cand[TOPM + j]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid)
                                + cal_bm25_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + j]], bm25_val+bm25_off[top_M_Cand[TOPM + j]], bm25_off[top_M_Cand[TOPM + j]+1]-bm25_off[top_M_Cand[TOPM + j]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid, &key_hit[threadIdx.y]);
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = 1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
                    if(key_hit[threadIdx.y] > 0) {
                        top_M_Cand[TOPM + j] |= 0x40000000;
                        key_hit[threadIdx.y] = 0;
                    }
                }
            }
        }
        __syncthreads();
        
        // merge into top_M
        bitonic_sort_id_by_dis_no_explore(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], K);
        merge_top_keyword(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], tid, keyword_Cand, keyword_Cand_dis, &keyword_num, TOPM);
    }
    __syncthreads();
    bitonic_sort_id_by_dis_no_explore(keyword_Cand_dis, keyword_Cand, KEYWORD);

    if(tid == 0) {
        keyword_num = 0;
        for (unsigned i = 0; i < TOPM; i ++) {
            if((top_M_Cand[i] & 0x40000000) != 0){
                results_id[bid * TOPM + keyword_num] = (top_M_Cand[i] & 0x3FFFFFFF);
                results_dis[bid * TOPM + keyword_num] = top_M_Cand_dis[i];
                keyword_num++;
            }
        }
    }
    __syncthreads();
    for(unsigned i = keyword_num + tid; i < TOPM; i += blockDim.x * blockDim.y){
        results_id[bid * TOPM + i] = (keyword_Cand[i - keyword_num] & 0x3FFFFFFF);
        results_dis[bid * TOPM + i] = keyword_Cand_dis[i - keyword_num];
    }

}

__device__ void merge_top_kg(unsigned* arr1, unsigned* arr2, float* arr1_val, float* arr2_val, unsigned* arr_ent, unsigned* arr_ent_sub, unsigned* arr_hop, unsigned* arr_hop_sub, unsigned tid, unsigned TOPM){
    unsigned res_id_vec[(MAX_CAND_POOL + K+6*32-1)/ (6*32)] = {0};
    float val_vec[(MAX_CAND_POOL + K+6*32-1)/ (6*32)];
    unsigned id_reg_vec[(MAX_CAND_POOL + K+6*32-1)/ (6*32)];
    unsigned id_reg2[(MAX_CAND_POOL + K+6*32-1)/ (6*32)];
    unsigned id_reg3[(MAX_CAND_POOL + K+6*32-1)/ (6*32)];
    for(unsigned i = 0; i < (TOPM + K+6*32-1)/ (6*32); i ++){
        // unsigned res_id_vec[i] = 0;
        // float val;
        if(i * blockDim.x * blockDim.y + tid < TOPM){
            val_vec[i] = arr1_val[i * blockDim.x * blockDim.y + tid];
            id_reg_vec[i] = arr1[i * blockDim.x * blockDim.y + tid];
            id_reg2[i] = arr_ent[i * blockDim.x * blockDim.y + tid];
            id_reg3[i] = arr_hop[i * blockDim.x * blockDim.y + tid];
            unsigned tmp = K;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr2_val[res_id_vec[i] + halfsize];
                res_id_vec[i] += ((cand <= val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id_vec[i] += (arr2_val[res_id_vec[i]] <= val_vec[i]);
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid);
        }
        else if(i * blockDim.x * blockDim.y + tid < TOPM + K){
            val_vec[i] = arr2_val[i * blockDim.x * blockDim.y + tid - TOPM];
            id_reg_vec[i] = arr2[i * blockDim.x * blockDim.y + tid - TOPM];
            id_reg2[i] = arr_ent_sub[i * blockDim.x * blockDim.y + tid - TOPM];
            id_reg3[i] = arr_hop_sub[i * blockDim.x * blockDim.y + tid - TOPM];
            unsigned tmp = TOPM;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = arr1_val[res_id_vec[i] + halfsize];
                res_id_vec[i] += ((cand < val_vec[i]) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id_vec[i] += (arr1_val[res_id_vec[i]] < val_vec[i]);
            res_id_vec[i] += (i * blockDim.x * blockDim.y + tid - TOPM);
        }
        else{
            res_id_vec[i] = TOPM;
        }
    }
    __syncthreads();
    for(unsigned i = 0; i < (TOPM + K + 6 * 32 - 1)/ (6 * 32); i ++){
        if(res_id_vec[i] < TOPM){
            arr1[res_id_vec[i]] = id_reg_vec[i];
            arr1_val[res_id_vec[i]] = val_vec[i];
            arr_ent[res_id_vec[i]] = id_reg2[i];
            arr_hop[res_id_vec[i]] = id_reg3[i];
        }
    }
    __syncthreads();
}

// knowledge graph
__global__ void search_with_kg(unsigned* graph, const half* __restrict__ values, const half* __restrict__ query_data, unsigned* results_id, float* results_dis, unsigned* ent_pts, unsigned* sparse_off, unsigned *sparse_idx, float* sparse_val, unsigned *bm25_off, unsigned *bm25_idx, float *bm25_val, unsigned* sparse_off_query, unsigned *sparse_idx_query, float* sparse_val_query, unsigned *bm25_off_query, unsigned *bm25_idx_query, float *bm25_val_query, unsigned* e2doc_off, unsigned* e2doc_idx, unsigned* e2e_off, unsigned* e2e_idx, unsigned* query_ent_off, unsigned* query_ent_idx, float kg_weight, unsigned TOPM){
    __shared__ unsigned top_M_Cand[MAX_CAND_POOL + K];
    __shared__ float top_M_Cand_dis[MAX_CAND_POOL + K];
    __shared__ unsigned ent_shared[MAX_CAND_POOL + K];
    __shared__ unsigned kg_hop[MAX_CAND_POOL + K];

    __shared__ half4 tmp_val_sha[DIM / 4];

    __shared__ unsigned node_idx[MAX_SPARSE];
    __shared__ float node_val[MAX_SPARSE];
    __shared__ unsigned node_idx_bm25[MAX_BM25];
    __shared__ float node_val_bm25[MAX_BM25];

    __shared__ unsigned to_explore;

    __shared__ unsigned hash_table[HASHLEN];

    __shared__ unsigned kg_tmp;
    __shared__ unsigned kg_cur_id;

    unsigned long long start;
    double duration = 0.0;

    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;  
    if(tid == 0){
        kg_tmp = 0;
    }
    // initialize
    for(unsigned i = tid; i < HASHLEN; i += blockDim.x * blockDim.y){
        hash_table[i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = 0xFFFFFFFF;
        top_M_Cand_dis[i] = INF_DIS;
    }
    // fill top_M_Cand rapidly
    if(tid == 0){
        for(unsigned i = query_ent_off[bid]; i < query_ent_off[bid + 1]; i++){
            for(unsigned j = e2doc_off[query_ent_idx[i]]; j < e2doc_off[query_ent_idx[i] + 1]; j++){
                unsigned to_append = e2doc_idx[j];
                if(hash_insert(hash_table, to_append) != 0 && kg_tmp < TOPM+K){
                    top_M_Cand[kg_tmp] = to_append;
                    ent_shared[kg_tmp] = query_ent_idx[i];
                    kg_hop[kg_tmp] = 0;
                    kg_tmp++;
                }
            }
        }
    }
    __syncthreads();

    if(kg_tmp < TOPM){
        for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
            unsigned to_append = ent_pts[i];
            if(hash_insert(hash_table, to_append) != 0){
                unsigned tmp_id = atomicAdd(&kg_tmp, 1);
                if(tmp_id < TOPM + K){
                    top_M_Cand[tmp_id] = to_append;
                    ent_shared[tmp_id] = 0xFFFFFFFF;
                    kg_hop[tmp_id] = 0;
                }
            }
        }
    }
    __syncthreads();

    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&query_data[bid * DIM + 4 * i]);
	}
    for(unsigned i = sparse_off_query[bid] + tid; i < sparse_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx[i - sparse_off_query[bid]] = sparse_idx_query[i];
        node_val[i - sparse_off_query[bid]] = sparse_val_query[i];
    }
    for(unsigned i = bm25_off_query[bid] + tid; i < bm25_off_query[bid + 1]; i += blockDim.x * blockDim.y){
        node_idx_bm25[i - bm25_off_query[bid]] = bm25_idx_query[i];
        node_val_bm25[i - bm25_off_query[bid]] = bm25_val_query[i];
    }
    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < min(TOPM + K, kg_tmp); i += blockDim.y){
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            half4 val2 = Load(&values[top_M_Cand[i] * DIM + j * 4]);
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].x, val2.x));
            val_res = __hadd2(val_res, __hmul2(tmp_val_sha[j].y, val2.y));
        }
        float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[i]], sparse_val+sparse_off[top_M_Cand[i]], sparse_off[top_M_Cand[i]+1]-sparse_off[top_M_Cand[i]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid)
                                + cal_sparse_dist(bm25_idx+bm25_off[top_M_Cand[i]], bm25_val+bm25_off[top_M_Cand[i]], bm25_off[top_M_Cand[i]+1]-bm25_off[top_M_Cand[i]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid);
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
        }
        if(laneid == 0){
            // top_M_Cand_dis[TOPM + i] = __half2float(__hadd(val_res.x, val_res.y));
            top_M_Cand_dis[i] = 1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
            if(ent_shared[i] != 0xFFFFFFFF) top_M_Cand_dis[i] -= kg_weight;
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis_kg(top_M_Cand_dis, top_M_Cand, ent_shared, kg_hop, min(TOPM+K, kg_tmp));

    // begin search
    for(unsigned i = 0; i < MAX_ITER; i++){
        if(tid == 0){
            kg_tmp = 0;
        }
        if((i + 1) % HASH_RESET == 0){
            for(unsigned j = tid; j < HASHLEN; j += blockDim.x * blockDim.y){
                hash_table[j] = 0xFFFFFFFF;
            }
            __syncthreads();
            for(unsigned j = tid; j < TOPM + K; j += blockDim.x * blockDim.y){
                hash_insert(hash_table, (top_M_Cand[j] & 0x7FFFFFFF));
            }
        }
        __syncthreads();

        if(tid < 32){
            for(unsigned j = laneid; j < TOPM; j+=32){
                unsigned n_p = 0;
                if((top_M_Cand[j] & 0x80000000) == 0){
                    n_p = 1;
                }
                unsigned ballot_res = __ballot_sync(0xffffffff, n_p);
                if(ballot_res > 0){
                    if(laneid == (__ffs(ballot_res) - 1)){
                        to_explore = top_M_Cand[j];
                        top_M_Cand[j] |= 0x80000000;
                    }
                    break;
                }
                to_explore = 0xFFFFFFFF;
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            // if(bid == 0 && tid == 0) printf("BREAK: %d\n", i);
            break;
        }
        
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * (K+KEYEDGE) + j];
            if(hash_insert(hash_table, to_append) == 0){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
            ent_shared[TOPM + j] = 0xFFFFFFFF;
            kg_hop[TOPM + j] = kg_hop[kg_cur_id] + 1;
        }
        __syncthreads();
        if(kg_hop[kg_cur_id] < 3 && ent_shared[kg_cur_id] != 0xFFFFFFFF){
            // intersect computation
            for(unsigned j = e2e_off[ent_shared[kg_cur_id]] + tid; j < e2e_off[ent_shared[kg_cur_id] + 1]; j += blockDim.x * blockDim.y){
                for(unsigned k = e2doc_off[e2e_idx[j]]; k < e2doc_off[e2e_idx[j] + 1]; k++){
                    unsigned val = e2doc_idx[k];
                    unsigned tmp = K, res_id = 0;
                    while (tmp > 1) {
                        unsigned halfsize = tmp / 2;
                        unsigned cand = (top_M_Cand[TOPM + res_id + halfsize] & 0x3FFFFFFF);
                        res_id += ((cand < val) ? halfsize : 0);
                        tmp -= halfsize;
                    }
                    res_id += ((top_M_Cand[TOPM + res_id] & 0x3FFFFFFF) < val);
                    if(res_id <= (K - 1) && (top_M_Cand[TOPM + res_id] & 0x3FFFFFFF) == val && top_M_Cand_dis[TOPM + res_id] < 1.0){
                        top_M_Cand_dis[TOPM + res_id] = 2.0;
                        ent_shared[TOPM + res_id] = e2e_idx[j];
                    }
                }
            }
            __syncthreads();
            // kg edge augmentation
            for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
                if(top_M_Cand_dis[TOPM + j] > 3.0){
                    unsigned tmp_id = atomicAdd(&kg_tmp, 1);
                    if(tmp_id < e2e_off[ent_shared[kg_cur_id] + 1] - e2e_off[ent_shared[kg_cur_id]]){
                        for(unsigned k = e2doc_off[e2e_idx[e2e_off[ent_shared[kg_cur_id]] + tmp_id]]; k < e2doc_off[e2e_idx[e2e_off[ent_shared[kg_cur_id]] + tmp_id] + 1]; k++){
                            unsigned to_append = e2doc_idx[k];
                            if((to_explore & 0x3FFFFFFF) == to_append) continue;
                            if(hash_insert(hash_table, to_append) != 0){
                                top_M_Cand[TOPM + j] = to_append;
                                top_M_Cand_dis[TOPM + j] = 2.0;
                                ent_shared[TOPM + j] = e2e_idx[e2e_off[ent_shared[kg_cur_id]] + tmp_id];
                                kg_hop[TOPM + j] = kg_hop[kg_cur_id] + 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 3.0){
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].x, val2.x));
                    val_res = __hadd2(val_res, __hmul2(tmp_val_sha[k].y, val2.y));
                }
                float sparse_dis = cal_sparse_dist(sparse_idx+sparse_off[top_M_Cand[TOPM + j]], sparse_val+sparse_off[top_M_Cand[TOPM + j]], sparse_off[top_M_Cand[TOPM + j]+1]-sparse_off[top_M_Cand[TOPM + j]], node_idx, node_val, sparse_off_query[bid+1]-sparse_off_query[bid], laneid)
                                + cal_sparse_dist(bm25_idx+bm25_off[top_M_Cand[TOPM + j]], bm25_val+bm25_off[top_M_Cand[TOPM + j]], bm25_off[top_M_Cand[TOPM + j]+1]-bm25_off[top_M_Cand[TOPM + j]], node_idx_bm25, node_val_bm25, bm25_off_query[bid+1]-bm25_off_query[bid], laneid);
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                    sparse_dis += __shfl_down_sync(0xffffffff, sparse_dis, lane_mask);
                }

                if(laneid == 0){
                    if(top_M_Cand_dis[TOPM + j] > 1.0)
                        top_M_Cand_dis[TOPM + j] = (1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis) - (kg_weight / (float)kg_hop[TOPM + j]);
                    else
                        top_M_Cand_dis[TOPM + j] = 1.0 - __half2float(__hadd(val_res.x, val_res.y)) - sparse_dis;
                }
            }
        }
        __syncthreads();
        
        // merge into top_M
        bitonic_sort_id_by_dis_kg(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], &ent_shared[TOPM], &kg_hop[TOPM], K);
        merge_top_kg(top_M_Cand, &top_M_Cand[TOPM], top_M_Cand_dis, &top_M_Cand_dis[TOPM], ent_shared, &ent_shared[TOPM], kg_hop, &kg_hop[TOPM], tid, TOPM);
    }
    __syncthreads();
    
    for (unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y) {
        results_id[bid * TOPM + i] = (top_M_Cand[i] & 0x7FFFFFFF);
        results_dis[bid * TOPM + i] = top_M_Cand_dis[i];
    }

}

void search_index_impl(
    const std::string& dense_data_path,
    const std::string& dense_query_path,
    const std::string& sparse_data_path,
    const std::string& sparse_query_path,
    const std::string& bm25_data_path,
    const std::string& bm25_query_path,
    const std::string& keyword_id_path,
    const std::string& knowledge_path,
    const std::string& entity2doc_path,
    const std::string& query_entity_path,
    const std::string& graph_path,
    const std::string& ground_truth_path,
    int top_k,
    int cands,
    float sparse_weight,
    float bm25_weight,
    float dense_weight,
    float kg_weight
){
    float* data_load = NULL, *query_load = NULL;
    unsigned points_num, dim, query_num, query_dim, non_zero_num, bm25_non_zero, query_non_zero_num, query_bm25_non_zero;
    std::vector<unsigned> final_graph_;
    unsigned* sparse_off = NULL, *sparse_idx = NULL, *bm25_off = NULL, *bm25_idx = NULL, *sparse_off_query = NULL, *sparse_idx_query = NULL, *bm25_off_query = NULL, *bm25_idx_query = NULL;
    float* sparse_val = NULL, *bm25_val = NULL, *sparse_val_query = NULL, *bm25_val_query = NULL;

    unsigned* kg_graph_off = NULL, *kg_graph_idx = NULL, *kg2doc_off = NULL, *kg2doc_idx = NULL, *kg_query_off = NULL, *kg_query_idx = NULL;
    unsigned entity_num, query_entity_num;

    load_data((char*)dense_data_path.c_str(), data_load, points_num, dim);
    load_data((char*)dense_query_path.c_str(), query_load, query_num, query_dim);

    load_sparse_data((char*)sparse_data_path.c_str(), sparse_off, sparse_idx, sparse_val, points_num+1, non_zero_num);
    load_sparse_data((char*)bm25_data_path.c_str(), bm25_off, bm25_idx, bm25_val, points_num+1, bm25_non_zero);
    load_sparse_data((char*)sparse_query_path.c_str(), sparse_off_query, sparse_idx_query, sparse_val_query, query_num+1, query_non_zero_num);
    load_sparse_data((char*)bm25_query_path.c_str(), bm25_off_query, bm25_idx_query, bm25_val_query, query_num+1, query_bm25_non_zero);

    if(!knowledge_path.empty()){
        load_kg_graph((char*)knowledge_path.c_str(), kg_graph_off, kg_graph_idx, entity_num);
        load_kg_graph((char*)entity2doc_path.c_str(), kg2doc_off, kg2doc_idx, entity_num);
        load_kg_graph((char*)query_entity_path.c_str(), kg_query_off, kg_query_idx, query_entity_num);
        // cout << entity_num << endl;
    }

    query_num = 100; // specify the number of queries
    cout << "query num: " << query_num << endl;

    vector<vector<unsigned>> keyword_vec(query_num);
    if(!keyword_id_path.empty()){
        load_txt((char*)keyword_id_path.c_str(), keyword_vec, query_num);
        unsigned t_count = 0;
        for(unsigned i = 0; i < query_num; i++){
            for(unsigned j = bm25_off_query[i]; j < bm25_off_query[i + 1]; j++){
                for(unsigned k = 0; k < keyword_vec[i].size(); k++){
                    if(keyword_vec[i][k] == bm25_idx_query[j]){
                        bm25_idx_query[j] |= 0x80000000;
                        t_count += 1;
                    }
                }
            }
        }
        cout << "keywords: " << t_count << endl;
    }

    cout << "Dense weight: " << dense_weight << ", Sparse weight: " << sparse_weight << ", BM25 weight: " << bm25_weight << ", KG weight: " << kg_weight << endl;

    vector<float> res1, res2, res3;
    for(unsigned i = 0; i < min(points_num, 100); i++){
        for(unsigned j = 0; j < min(query_num, 100); j++){
            unsigned aa = j, bb = i;
            compute_hybrid_dis(data_load + aa * dim, query_load + bb * dim, dim, sparse_idx + sparse_off[aa], sparse_val + sparse_off[aa], sparse_off[aa+1] - sparse_off[aa], sparse_idx_query + sparse_off_query[bb], sparse_val_query + sparse_off_query[bb], sparse_off_query[bb+1] - sparse_off_query[bb], bm25_idx + bm25_off[aa], bm25_val + bm25_off[aa], bm25_off[aa+1] - bm25_off[aa], bm25_idx_query + bm25_off_query[bb], bm25_val_query + bm25_off_query[bb], bm25_off_query[bb+1] - bm25_off_query[bb], res1, res2, res3);
        }
    }
    sort(res1.begin(), res1.end());
    sort(res2.begin(), res2.end());
    sort(res3.begin(), res3.end());

    if((-res2[0]) > (-res3[0]) && (-res2[0]) > (-res1[0])){
        bm25_weight *= (res2[0]/res3[0]);
        dense_weight *= (res2[0]/res1[0]);
    }
    else if((-res3[0]) > (-res2[0]) && (-res3[0]) > (-res1[0])){
        sparse_weight *= (res3[0]/res2[0]);
        dense_weight *= (res3[0]/res1[0]);
    }
    else if((-res1[0]) > (-res2[0]) && (-res1[0]) > (-res3[0])){
        sparse_weight *= (res1[0]/res2[0]);
        bm25_weight *= (res1[0]/res3[0]);
    }

    for(unsigned i = 0; i < query_num; i++){
        for(unsigned j = 0; j < dim; j++){
            query_load[i * dim + j] *= dense_weight;
        }
    }
    for(unsigned i = 0; i < query_non_zero_num; i++){
        sparse_val_query[i] *= sparse_weight;
    }
    for(unsigned i = 0; i < query_bm25_non_zero; i++){
        bm25_val_query[i] *= bm25_weight;
    }
    vector<unsigned> ent_pts;

    load_graph((char*)graph_path.c_str(), final_graph_, ent_pts);
    cout << "Points: " << points_num << ", Dim: " << dim << endl;
    cout << "Query: " << query_num << ", Dim: " << query_dim << endl;

    unsigned max_sparse_dim = 0, max_bm25_dim = 0, max_query_sparse_dim = 0, max_query_bm25_dim = 0;
    for(unsigned i = 0; i < points_num; i++){
        if(max_sparse_dim < sparse_off[i + 1] - sparse_off[i]) max_sparse_dim = sparse_off[i + 1] - sparse_off[i];
        if(max_bm25_dim < bm25_off[i + 1] - bm25_off[i]) max_bm25_dim = bm25_off[i + 1] - bm25_off[i];
    }
    for(unsigned i = 0; i < query_num; i++){
        if(max_query_sparse_dim < sparse_off_query[i + 1] - sparse_off_query[i]) max_query_sparse_dim = sparse_off_query[i + 1] - sparse_off_query[i];
        if(max_query_bm25_dim < bm25_off_query[i + 1] - bm25_off_query[i]) max_query_bm25_dim = bm25_off_query[i + 1] - bm25_off_query[i];
    }
    
    cout << "Max sparse dim: " << max_sparse_dim << ", Max bm25 dim: " << max_bm25_dim << endl;
    cout << "Max query sparse dim: " << max_query_sparse_dim << ", Max query bm25 dim: " << max_query_bm25_dim << endl;
    
    unsigned* graph_dev;
    cudaMalloc((void**)&graph_dev, points_num * (K + KEYEDGE) * sizeof(unsigned));
    cudaMemcpy(graph_dev, final_graph_.data(), points_num * (K + KEYEDGE) * sizeof(unsigned), cudaMemcpyHostToDevice);

    unsigned* ent_pts_dev;
    cudaMalloc((void**)&ent_pts_dev, K * sizeof(unsigned));
    cudaMemcpy(ent_pts_dev, ent_pts.data(), K * sizeof(unsigned), cudaMemcpyHostToDevice);

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

    float* query_data_dev;
    cudaMalloc((void**)&query_data_dev, query_num * dim * sizeof(float));
    cudaMemcpy(query_data_dev, query_load, query_num * dim * sizeof(float), cudaMemcpyHostToDevice);

    half* query_data_half_dev;
    cudaMalloc((void**)&query_data_half_dev, query_num * dim * sizeof(half));

    unsigned* sparse_off_query_dev, * sparse_idx_query_dev, *bm25_off_query_dev, *bm25_idx_query_dev;
    float* sparse_val_query_dev, *bm25_val_query_dev;
    cudaMalloc((void**)&sparse_off_query_dev, (query_num+1) * sizeof(unsigned));
    cudaMemcpy(sparse_off_query_dev, sparse_off_query, (query_num+1) * sizeof(unsigned), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&sparse_idx_query_dev, query_non_zero_num * sizeof(unsigned));
    cudaMemcpy(sparse_idx_query_dev, sparse_idx_query, query_non_zero_num * sizeof(unsigned), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&sparse_val_query_dev, query_non_zero_num * sizeof(float));
    cudaMemcpy(sparse_val_query_dev, sparse_val_query, query_non_zero_num * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&bm25_off_query_dev, (query_num+1) * sizeof(unsigned));
    cudaMemcpy(bm25_off_query_dev, bm25_off_query, (query_num+1) * sizeof(unsigned), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&bm25_idx_query_dev, query_bm25_non_zero * sizeof(unsigned));
    cudaMemcpy(bm25_idx_query_dev, bm25_idx_query, query_bm25_non_zero * sizeof(unsigned), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&bm25_val_query_dev, query_bm25_non_zero * sizeof(float));
    cudaMemcpy(bm25_val_query_dev, bm25_val_query, query_bm25_non_zero * sizeof(float), cudaMemcpyHostToDevice);

    unsigned* kg2doc_off_dev, *kg2doc_idx_dev, *kg_off_dev, *kg_idx_dev, *kg_query_off_dev, *kg_query_idx_dev;
    if(!knowledge_path.empty()){
        cudaMalloc((void**)&kg2doc_off_dev, entity_num * sizeof(unsigned));
        cudaMemcpy(kg2doc_off_dev, kg2doc_off, entity_num * sizeof(unsigned), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&kg2doc_idx_dev, kg2doc_off[entity_num - 1] * sizeof(unsigned));
        cudaMemcpy(kg2doc_idx_dev, kg2doc_idx, kg2doc_off[entity_num - 1] * sizeof(unsigned), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&kg_off_dev, entity_num * sizeof(unsigned));
        cudaMemcpy(kg_off_dev, kg_graph_off, entity_num * sizeof(unsigned), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&kg_idx_dev, kg_graph_off[entity_num - 1] * sizeof(unsigned));
        cudaMemcpy(kg_idx_dev, kg_graph_idx, kg_graph_off[entity_num - 1] * sizeof(unsigned), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&kg_query_off_dev, query_entity_num * sizeof(unsigned));
        cudaMemcpy(kg_query_off_dev, kg_query_off, query_entity_num * sizeof(unsigned), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&kg_query_idx_dev, kg_query_off[query_entity_num - 1] * sizeof(unsigned));
        cudaMemcpy(kg_query_idx_dev, kg_query_idx, kg_query_off[query_entity_num - 1] * sizeof(unsigned), cudaMemcpyHostToDevice);
        cout << "KG Copy end" << endl;
    }

    std::vector<unsigned> query_results;
    std::vector<float> query_dis;

    query_results.resize(cands * query_num);
    query_dis.resize(cands * query_num);

    unsigned* query_results_dev;
    cudaMalloc((void**)&query_results_dev, query_num * cands * sizeof(unsigned));

    float* query_dis_dev;
    cudaMalloc((void**)&query_dis_dev, query_num * cands * sizeof(float));
    
    dim3 grid_s(query_num, 1, 1);
    dim3 block_s(32, 6, 1);

    cout << "Begin search" << endl;

    auto start = std::chrono::high_resolution_clock::now();

    f2h<<<points_num, 256>>>(data_dev, data_half_dev, points_num);
    f2h<<<query_num, 256>>>(query_data_dev, query_data_half_dev, query_num);
    
    if(kg_weight < 1e-5){
        if(!keyword_id_path.empty()){
            cout << "Search with dense + sparse + bm25 + keyword" << endl;
            search_with_keyword<<<grid_s, block_s>>>(graph_dev, data_half_dev, query_data_half_dev, query_results_dev, query_dis_dev, ent_pts_dev, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev, sparse_off_query_dev, sparse_idx_query_dev, sparse_val_query_dev, bm25_off_query_dev, bm25_idx_query_dev, bm25_val_query_dev, cands);
        }
        else if(bm25_weight < 1e-5){
            if(sparse_weight > 1e-5 && dense_weight > 1e-5){
                cout << "Search with dense + sparse" << endl;
                search_with_dense_and_sparse<<<grid_s, block_s>>>(graph_dev, data_half_dev, query_data_half_dev, query_results_dev, query_dis_dev, ent_pts_dev, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev, sparse_off_query_dev, sparse_idx_query_dev, sparse_val_query_dev, bm25_off_query_dev, bm25_idx_query_dev, bm25_val_query_dev, cands);
            }
            else if (sparse_weight < 1e-5 && dense_weight > 1e-5){
                cout << "Search with dense" << endl;
                search_with_dense<<<grid_s, block_s>>>(graph_dev, data_half_dev, query_data_half_dev, query_results_dev, query_dis_dev, ent_pts_dev, cands);
            }
            else if (sparse_weight > 1e-5 && dense_weight < 1e-5){
                cout << "Search with sparse" << endl;
                search_with_sparse<<<grid_s, block_s>>>(graph_dev, data_half_dev, query_data_half_dev, query_results_dev, query_dis_dev, ent_pts_dev, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev, sparse_off_query_dev, sparse_idx_query_dev, sparse_val_query_dev, bm25_off_query_dev, bm25_idx_query_dev, bm25_val_query_dev, cands);
            }
        }
        else{
            if(sparse_weight > 1e-5 && dense_weight > 1e-5){
                cout << "Search with dense + sparse + bm25" << endl;
                search_with_three_paths<<<grid_s, block_s>>>(graph_dev, data_half_dev, query_data_half_dev, query_results_dev, query_dis_dev, ent_pts_dev, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev, sparse_off_query_dev, sparse_idx_query_dev, sparse_val_query_dev, bm25_off_query_dev, bm25_idx_query_dev, bm25_val_query_dev, cands);
            }
            else if (sparse_weight < 1e-5 && dense_weight > 1e-5){
                cout << "Search with dense + bm25" << endl;
                search_with_bm25_and_dense<<<grid_s, block_s>>>(graph_dev, data_half_dev, query_data_half_dev, query_results_dev, query_dis_dev, ent_pts_dev, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev, sparse_off_query_dev, sparse_idx_query_dev, sparse_val_query_dev, bm25_off_query_dev, bm25_idx_query_dev, bm25_val_query_dev, cands);
            }
            else if (sparse_weight > 1e-5 && dense_weight < 1e-5){
                cout << "Search with sparse + bm25" << endl;
                search_with_sparse_and_bm25<<<grid_s, block_s>>>(graph_dev, data_half_dev, query_data_half_dev, query_results_dev, query_dis_dev, ent_pts_dev, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev, sparse_off_query_dev, sparse_idx_query_dev, sparse_val_query_dev, bm25_off_query_dev, bm25_idx_query_dev, bm25_val_query_dev, cands);
            }
            else{
                cout << "Search with bm25" << endl;    
                search_with_bm25<<<grid_s, block_s>>>(graph_dev, data_half_dev, query_data_half_dev, query_results_dev, query_dis_dev, ent_pts_dev, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev, sparse_off_query_dev, sparse_idx_query_dev, sparse_val_query_dev, bm25_off_query_dev, bm25_idx_query_dev, bm25_val_query_dev, cands);
            }
        }
    }
    else{
        cout << "Search with dense + sparse + bm25 + kg" << endl;
        if(!knowledge_path.empty()){
            search_with_kg<<<grid_s, block_s>>>(graph_dev, data_half_dev, query_data_half_dev, query_results_dev, query_dis_dev, ent_pts_dev, sparse_off_dev, sparse_idx_dev, sparse_val_dev, bm25_off_dev, bm25_idx_dev, bm25_val_dev, sparse_off_query_dev, sparse_idx_query_dev, sparse_val_query_dev, bm25_off_query_dev, bm25_idx_query_dev, bm25_val_query_dev, kg2doc_off_dev, kg2doc_idx_dev, kg_off_dev, kg_idx_dev, kg_query_off_dev, kg_query_idx_dev, kg_weight, cands);
        }
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Query time: " << duration.count() << "s" << std::endl;

    cudaMemcpy(query_results.data(), query_results_dev, query_num * cands * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(query_dis.data(), query_dis_dev, query_num * cands * sizeof(float), cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaGetLastError());

    cudaFree(graph_dev);
    cudaFree(ent_pts_dev);
    cudaFree(data_dev);
    cudaFree(data_half_dev);
    cudaFree(sparse_off_dev);
    cudaFree(sparse_idx_dev);
    cudaFree(bm25_off_dev);
    cudaFree(bm25_idx_dev);
    cudaFree(sparse_val_dev);
    cudaFree(bm25_val_dev);
    checkCudaErrors(cudaGetLastError());

    vector<vector<unsigned>> results2(query_num);
    for(unsigned i = 0; i < query_num; i++){
        for(unsigned j = 0; j < top_k; j++){
            results2[i].push_back(query_results[i * cands + j]);
        }
    }

    vector<vector<unsigned>> results3(query_num);

    load_txt((char*)ground_truth_path.c_str(), results3, query_num);
    cal_precision(results2, results3);
}