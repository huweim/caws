// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "gpu-cache.h"
#include "stat-tool.h"
#include <algorithm>
#include <assert.h>

#define MAX_DEFAULT_CACHE_SIZE_MULTIBLIER 4
#define USE_PREFETCH 1
// used to allocate memory that is large enough to adapt the changes in cache size across kernels

const char *cache_request_status_str(enum cache_request_status status)
{
    static const char *static_cache_request_status_str[] = {
        "HIT",
        "HIT_RESERVED",
        "MISS",
        "RESERVATION_FAIL"};

    assert(sizeof(static_cache_request_status_str) / sizeof(const char *) == NUM_CACHE_REQUEST_STATUS);
    assert(status < NUM_CACHE_REQUEST_STATUS);

    return static_cache_request_status_str[status];
}

unsigned l1d_cache_config::set_index(new_addr_type addr) const
{
    unsigned set_index = m_nset; // Default to linear set index function
    unsigned lower_xor = 0;
    unsigned upper_xor = 0;

    switch (m_set_index_function)
    {
    case FERMI_HASH_SET_FUNCTION:
        /*
        * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse Distance Theory"
        * Cedric Nugteren et al.
        * ISCA 2014
        */
        if (m_nset == 32 || m_nset == 64)
        {
            // Lower xor value is bits 7-11
            lower_xor = (addr >> m_line_sz_log2) & 0x1F;

            // Upper xor value is bits 13, 14, 15, 17, and 19
            upper_xor = (addr & 0xE000) >> 13;   // Bits 13, 14, 15
            upper_xor |= (addr & 0x20000) >> 14; // Bit 17
            upper_xor |= (addr & 0x80000) >> 15; // Bit 19

            set_index = (lower_xor ^ upper_xor);

            // 48KB cache prepends the set_index with bit 12
            if (m_nset == 64)
                set_index |= (addr & 0x1000) >> 7;
        }
        else
        { /* Else incorrect number of sets for the hashing function */
            assert("\nGPGPU-Sim cache configuration error: The number of sets should be "
                   "32 or 64 for the hashing set index function.\n" &&
                   0);
        }
        break;

    case CUSTOM_SET_FUNCTION:
        /* No custom set function implemented */
        break;

    case LINEAR_SET_FUNCTION:
        set_index = (addr >> m_line_sz_log2) & (m_nset - 1);
        break;
    }

    // Linear function selected or custom set index function not implemented
    assert((set_index < m_nset) && "\nError: Set index out of bounds. This is caused by "
                                   "an incorrect or unimplemented custom set index function.\n");

    return set_index;
}

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping)
{
    cache_config::init(m_config_string, FuncCachePreferNone);
    m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const
{
    if (!m_address_mapping)
    {
        return (addr >> m_line_sz_log2) & (m_nset - 1);
    }
    else
    {
        // Calculate set index without memory partition bits to reduce set camping
        new_addr_type part_addr = m_address_mapping->partition_address(addr);
        return (part_addr >> m_line_sz_log2) & (m_nset - 1);
    }
}

tag_array::~tag_array()
{
    delete[] m_lines;
    delete m_wle;
}

tag_array::tag_array(cache_config &config,
                     int core_id,
                     int type_id,
                     cache_block_t *new_lines)
    : m_config(config),
      m_lines(new_lines)
{
    init(core_id, type_id);
}

void tag_array::update_cache_parameters(cache_config &config)
{
    m_config = config;
}

tag_array::tag_array(cache_config &config,
                     int core_id,
                     int type_id)
    : m_config(config)
{
    //assert( m_config.m_write_policy == READ_ONLY ); Old assert
    m_lines = new cache_block_t[MAX_DEFAULT_CACHE_SIZE_MULTIBLIER * config.get_num_lines()];
    init(core_id, type_id);
}

void tag_array::init(int core_id, int type_id)
{
    m_access = 0;
    m_miss = 0;
    m_pending_hit = 0;
    m_res_fail = 0;
    // initialize snapshot counters for visualizer
    m_prev_snapshot_access = 0;
    m_prev_snapshot_miss = 0;
    m_prev_snapshot_pending_hit = 0;
    m_core_id = core_id;
    m_type_id = type_id;

    m_wle = new warp_locality_evaluation();
    m_cache_prefetch = new cache_prefetch();
    prefetch_hit_num = 0;
    prefetch_hit_res_num = 0;
    useless_prefetch = 0;
}
bool cache_prefetch::warp_find(int pc_index, unsigned cta_id, unsigned warp_id){
    std::vector<int>::iterator it;
    for(it = m_stride_buffer[pc_index].m_cta_entry[cta_id].warp_set.begin(); it != m_stride_buffer[pc_index].m_cta_entry[cta_id].warp_set.end();it++){
        if(*it == warp_id)
            return true;
    }
    return false;
}
void cache_prefetch::trace_warp_addr(int sid, unsigned cta_id, unsigned warp_id, new_addr_type warp_addr, int pc_index)
{
    m_stride_buffer[pc_index].m_warp_entry[warp_id].addr_record.push_back(warp_addr);
}
void cache_prefetch::find_warp_id(int &lead_warp_id, int &least_warp_id, int pref_cta_id){
    std::vector<int>::iterator it = m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.begin();
    for(it; it != m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.end();it++){
        if(m_stride_buffer[last_pc_index].m_warp_entry[*it].warp_ld_inst_num > 
        m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num){
            lead_warp_id = *it;
        }
    }
    it = m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.begin();
    for(it; it != m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.end();it++){
        if(m_stride_buffer[last_pc_index].m_warp_entry[*it].warp_ld_inst_num < 
        m_stride_buffer[last_pc_index].m_warp_entry[least_warp_id].warp_ld_inst_num){
            least_warp_id = *it;
        }
    }
    it = m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.begin();
    bool flag = false;
    for(it; it != m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.end();it++){
        if(lead_warp_id == *it)
            continue;
        if(m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num -
        m_stride_buffer[last_pc_index].m_warp_entry[*it].warp_ld_inst_num < 25){
            flag = true;
            break;
        }
    }
    if(!flag){
        int temp = lead_warp_id;
        it = m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.begin();
        lead_warp_id = *it;
        for(it; it != m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.end();it++){
            if(*it == temp)
                continue;
            if(m_stride_buffer[last_pc_index].m_warp_entry[*it].warp_ld_inst_num > 
            m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num){
                lead_warp_id = *it;
            }
        }
    }
}
void cache_prefetch::calculate_inter_pref_addr(int sid, int wid, int cta_id){
    m_prefetch_req.valid = false;
    m_prefetch_req.put_time++;
    //if(last_pc_index == -1 || last_cta_id == -1)
    if(last_pc_index == -1)
        return;
    // if(m_prefetch_req.put_time > 10){
    //     m_stride_buffer[last_pc_index].cta_offset++;
    // }
    // if(sid == 0)
    //     printf("inter_pref\n");
    m_stride_buffer[last_pc_index].cta_offset = (m_stride_buffer[last_pc_index].cta_offset + 1) % m_stride_buffer[last_pc_index].active_cta_num;
    if(m_stride_buffer[last_pc_index].cta_offset < 2){
        m_stride_buffer[last_pc_index].cta_offset = 2;
    }
    int pref_cta_id = (cta_id + 2) % m_stride_buffer[last_pc_index].active_cta_num;
    int lead_wid = -2;
    calcu_addr(sid, wid, pref_cta_id, lead_wid);
    if(!m_prefetch_req.valid){
        int pref_cta_id = cta_id;
        int lead_wid = -1;
        calcu_addr(sid, wid, pref_cta_id, lead_wid);
    }

    if(!m_prefetch_req.valid){
        int pref_cta_id = (cta_id + 1) % m_stride_buffer[last_pc_index].active_cta_num;
        int lead_wid = -2;
        calcu_addr(sid, wid, pref_cta_id, lead_wid);
    }
    // if(!m_prefetch_req.valid){
    //     int pref_cta_id = cta_id;
    //     calcu_addr(sid, wid, pref_cta_id);
    // }
    // if(!m_prefetch_req.valid){
    //     int pref_cta_id = (cta_id + 3) % m_stride_buffer[last_pc_index].active_cta_num;
    //     calcu_addr(sid, wid, pref_cta_id);
    // }
}
int ldst_num_diff = 0;
int stride_problem = 0;
int calcu_addr_times = 0;
void cache_prefetch::calcu_addr(int sid, int wid, int cta_id, int lead_wid){
    int pref_cta_id = cta_id;
    if(!m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].active){
        m_stride_buffer[last_pc_index].cta_offset = (m_stride_buffer[last_pc_index].cta_offset + 1) % m_stride_buffer[last_pc_index].active_cta_num;
        return;
    }
    //std::vector<int>::iterator it = m_stride_buffer[last_pc_index].m_cta_entry[last_cta_id].warp_set.begin();

    // if(least_warp_id == lead_warp_id)
    //     return;
    // if(sid == 0)
    //     printf("calcu_addr_times:%d\n", calcu_addr_times++);
    if(m_stride_buffer[last_pc_index].m_last_stride != 0 && m_stride_buffer[last_pc_index].valid_stride){
        // if(m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num - m_stride_buffer[last_pc_index].m_warp_entry[least_warp_id].warp_ld_inst_num < 5){
        //     assert(m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num - m_stride_buffer[last_pc_index].m_warp_entry[least_warp_id].warp_ld_inst_num > 0);
        //     new_addr_type warp_addr = m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].inter_warp_last_addr;
        //     m_prefetch_req.addr = warp_addr + m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].last_stride * (least_warp_id - lead_warp_id);
        //     m_prefetch_req.warp_id = least_warp_id;
        //     m_prefetch_req.valid = true;
        //     m_prefetch_req.put_time = 0;
        //     if(sid == 0){
        //         printf("warp_addr:%x warp_id:%d stride:%d ld_inst_num:%d ", 
        //         warp_addr, lead_warp_id, m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].last_stride/128, m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num);
        //         printf("prefetch_req_addr:%x prefetch_req_warp_id:%d ld_inst_num:%d \n", 
        //         m_prefetch_req.addr, m_prefetch_req.warp_id, m_stride_buffer[last_pc_index].m_warp_entry[least_warp_id].warp_ld_inst_num);
        //     }
        // }
        // else{
            // int lead_warp_id = wid;
            int lead_warp_id = -2;
            int least_warp_id = -2;
            std::vector<int>::iterator it = m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.begin();
            if(lead_wid == -1){
                lead_warp_id = wid;
            }
            else if(lead_wid == -2){
                lead_warp_id = *it;
                least_warp_id = *it;
                find_warp_id(lead_warp_id, least_warp_id, pref_cta_id);
            }
            m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_offset = (m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_offset + 1) % 
            m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.size();
            int warp_offset = m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_offset;
            it = m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.begin() + warp_offset;
            std::vector<int>::iterator it_end;
            if(warp_offset == 0)
                it_end = m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.end();
            else
                it_end = m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.begin() + warp_offset - 1;
            assert(it != it_end);
            //if(sid == 0 && last_pc_index==0 && pref_cta_id==1)
                //printf("warp_offset:%d warp_size:%d\n", m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_offset, m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.size());
            for(it; it != it_end;it++){
                //if(*it == lead_warp_id || *it == least_warp_id);
                if(it == m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.end()){
                    assert(it_end != m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.end());
                    it = m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.begin();
                }
                if(*it == lead_warp_id){
                    if(it_end == it){
                        assert(it_end == m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.begin());
                        break;
                    }
                    continue;
                }
                if(m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num - m_stride_buffer[last_pc_index].m_warp_entry[*it].warp_ld_inst_num < 25 &&
                m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num - m_stride_buffer[last_pc_index].m_warp_entry[*it].warp_ld_inst_num > 20){
                //if(m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num > m_stride_buffer[last_pc_index].m_warp_entry[*it_end].warp_ld_inst_num){
                    assert(*it != lead_warp_id);
                    new_addr_type warp_addr = m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].inter_warp_last_addr;
                    new_addr_type temp_addr = warp_addr + m_stride_buffer[last_pc_index].m_last_stride * (*it - lead_warp_id);
                    if(m_prefetch_req.addr != temp_addr){
                        m_prefetch_req.addr = temp_addr;
                        int pref_ld_num = m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num;
                        if(pref_ld_num < 500 && pref_ld_num > 0)
                            m_stride_buffer[last_pc_index].m_warp_entry[*it].pref_addr_record[pref_ld_num] = m_prefetch_req.addr;
                        m_prefetch_req.warp_id = *it;
                        m_prefetch_req.valid = true;
                        m_prefetch_req.put_time = 0;
                        // if(sid == 0){
                        //     printf("warp_addr:%x warp_id:%d stride:%d ld_inst_num:%d ", 
                        //     warp_addr, lead_warp_id, m_stride_buffer[last_pc_index].m_last_stride, m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num);
                        //     printf("prefetch_req_addr:%x prefetch_req_warp_id:%d ld_inst_num:%d \n", 
                        //     m_prefetch_req.addr, m_prefetch_req.warp_id, m_stride_buffer[last_pc_index].m_warp_entry[*it].warp_ld_inst_num);
                        // }
                        break;
                    }
                }
                // else{
                //     if(sid == 0)
                //         printf("ldst_num_diff:%d\n", ldst_num_diff++);
                // }
                if(it_end == it){
                    assert(it_end == m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.begin());
                    break;
                }
            }
            if(*it_end != lead_warp_id && it_end != m_stride_buffer[last_pc_index].m_cta_entry[pref_cta_id].warp_set.end() && !m_prefetch_req.valid){
                if(m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num - m_stride_buffer[last_pc_index].m_warp_entry[*it_end].warp_ld_inst_num < 25 &&
                m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num - m_stride_buffer[last_pc_index].m_warp_entry[*it_end].warp_ld_inst_num > 20){
                //if(m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num > m_stride_buffer[last_pc_index].m_warp_entry[*it_end].warp_ld_inst_num){
                    assert(*it_end != lead_warp_id);
                    new_addr_type warp_addr = m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].inter_warp_last_addr;
                    new_addr_type temp_addr = warp_addr + m_stride_buffer[last_pc_index].m_last_stride * (*it - lead_warp_id);
                    if(m_prefetch_req.addr != temp_addr){
                        m_prefetch_req.addr = temp_addr;
                        int pref_ld_num = m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num;
                        if(pref_ld_num < 500 && pref_ld_num > 0)
                            m_stride_buffer[last_pc_index].m_warp_entry[*it].pref_addr_record[pref_ld_num] = m_prefetch_req.addr;
                        m_prefetch_req.warp_id = *it;
                        m_prefetch_req.valid = true;
                        m_prefetch_req.put_time = 0;
                        // if(sid == 0){
                        //     printf("warp_addr:%x warp_id:%d stride:%d ld_inst_num:%d ", 
                        //     warp_addr, lead_warp_id, m_stride_buffer[last_pc_index].m_last_stride, m_stride_buffer[last_pc_index].m_warp_entry[lead_warp_id].warp_ld_inst_num);
                        //     printf("prefetch_req_addr:%x prefetch_req_warp_id:%d ld_inst_num:%d \n", 
                        //     m_prefetch_req.addr, m_prefetch_req.warp_id, m_stride_buffer[last_pc_index].m_warp_entry[*it].warp_ld_inst_num);
                        // }
                    }
                }
                // else{
                //     if(sid == 0)
                //         printf("ldst_num_diff:%d\n", ldst_num_diff++);
                // }
            }
            
   //     }
    }
    // else{
    //     if(sid == 0)
    //         printf("stride_problem:%d\n", stride_problem++);
    // }
}


void cache_prefetch::calculate_inter_warp_stride(int sid, unsigned cta_id, unsigned warp_id, new_addr_type warp_addr, int pc_index)
{
    if(!m_stride_buffer[pc_index].m_cta_entry[cta_id].active){
        m_stride_buffer[pc_index].m_cta_entry[cta_id].active = true;
        m_stride_buffer[pc_index].active_cta_num++;
    }
    m_stride_buffer[pc_index].m_warp_entry[warp_id].warp_ld_inst_num++;
    // if(sid == 0)
    //     printf("ld_inst_num:%d\n", m_stride_buffer[pc_index].m_warp_entry[warp_id].warp_ld_inst_num);
    int pref_ld_num = m_stride_buffer[pc_index].m_warp_entry[warp_id].warp_ld_inst_num;
    if(m_stride_buffer[pc_index].m_warp_entry[warp_id].pref_addr_record[pref_ld_num] != (unsigned)-1 && pref_ld_num < 500){
        if(m_stride_buffer[pc_index].m_warp_entry[warp_id].pref_addr_record[pref_ld_num] == warp_addr)
            m_stride_buffer[pc_index].prefetch_hit++;
        else    
            m_stride_buffer[pc_index].prefetch_miss++;
        //if(sid == 0 )
        //    printf("demand_addr:%x pref_addr:%x pc:%d stride:%d cta:%d warp:%d hit:%d miss:%d\n", warp_addr, m_stride_buffer[pc_index].m_warp_entry[warp_id].pref_addr_record[pref_ld_num],
        //    pc_index, m_stride_buffer[pc_index].m_last_stride, cta_id, warp_id, m_stride_buffer[pc_index].prefetch_hit, m_stride_buffer[pc_index].prefetch_miss);
    }
    if(!warp_find(pc_index, cta_id, warp_id)){
        m_stride_buffer[pc_index].m_cta_entry[cta_id].warp_set.push_back(warp_id);
        std::sort(m_stride_buffer[pc_index].m_cta_entry[cta_id].warp_set.begin(), m_stride_buffer[pc_index].m_cta_entry[cta_id].warp_set.end());
    }
    m_stride_buffer[pc_index].m_warp_entry[warp_id].inter_warp_last_addr = warp_addr;
    last_pc_index = pc_index;
    last_cta_id = cta_id;
    assert(cta_id>=0 && warp_id >=0);
    std::vector<int>::iterator it;
    for(it = m_stride_buffer[pc_index].m_cta_entry[cta_id].warp_set.begin(); it != m_stride_buffer[pc_index].m_cta_entry[cta_id].warp_set.end();it++){
        if(*it == warp_id)
            continue;
        if(m_stride_buffer[pc_index].m_warp_entry[*it].warp_ld_inst_num == m_stride_buffer[pc_index].m_warp_entry[warp_id].warp_ld_inst_num){
            assert(*it != warp_id);
            long long addr_stride = m_stride_buffer[pc_index].m_warp_entry[*it].inter_warp_last_addr - warp_addr;
            int warp_stride = *it - warp_id;
            long long stride = addr_stride/warp_stride;
            // if(stride == m_stride_buffer[pc_index].m_last_stride)
            //     m_stride_buffer[pc_index].m_cta_entry[cta_id].stride_hit++;
            // else    
            //     m_stride_buffer[pc_index].m_cta_entry[cta_id].stride_miss++;
            if(stride != 0){
                //m_stride_buffer[pc_index].m_cta_entry[cta_id].last_stride = stride;
                //m_stride_buffer[pc_index].m_cta_entry[cta_id].inter_warp_stride.push_back(stride);
                m_stride_buffer[pc_index].m_inter_warp_stride.push_back(stride);
            }
            break;
        }
    }
    //if (sid==0)
        //printf("pref_hit:%d pref_miss:%d\n",m_stride_buffer[pc_index].m_cta_entry[cta_id].stride_hit, m_stride_buffer[pc_index].m_cta_entry[cta_id].stride_miss);
    // if(m_stride_buffer[pc_index].m_cta_entry[cta_id].last_stride != 0){
    //     for(it = m_stride_buffer[pc_index].m_cta_entry[cta_id].warp_set.begin(); it != m_stride_buffer[pc_index].m_cta_entry[cta_id].warp_set.end();it++){
    //         if(*it == warp_id)
    //             continue;
    //         if(m_stride_buffer[pc_index].m_warp_entry[warp_id].warp_ld_inst_num - m_stride_buffer[pc_index].m_warp_entry[*it].warp_ld_inst_num < 15 &&
    //         m_stride_buffer[pc_index].m_warp_entry[warp_id].warp_ld_inst_num - m_stride_buffer[pc_index].m_warp_entry[*it].warp_ld_inst_num > 0){
    //             assert(*it != warp_id);
    //             valid_pref_addr = true;
    //             prefetch_req_addr = warp_addr + m_stride_buffer[pc_index].m_cta_entry[cta_id].last_stride * (int)(*it - warp_id);
    //             prefetch_req_warp_id = *it;
    //             if(sid == 0){
    //                 printf("warp_addr:%x warp_id:%d stride:%d ld_inst_num:%d ", 
    //                 warp_addr, warp_id, m_stride_buffer[pc_index].m_cta_entry[cta_id].last_stride/128, m_stride_buffer[pc_index].m_warp_entry[warp_id].warp_ld_inst_num);
    //                 printf("prefetch_req_addr:%x prefetch_req_warp_id:%d ld_inst_num:%d\n", 
    //                 prefetch_req_addr, prefetch_req_warp_id, m_stride_buffer[pc_index].m_warp_entry[*it].warp_ld_inst_num);
    //             }
    //             break;
    //         }
    //     }
    // }
    // if(sid == 0){
    //     printf("prefetch_hit:%x prefetch_miss:%d\n", m_stride_buffer[pc_index].m_cta_entry[cta_id].stride_hit,
    //     m_stride_buffer[pc_index].m_cta_entry[cta_id].stride_miss);
    // }
}
void cache_prefetch::calculate_intra_warp_stride(int sid, unsigned warp_id, new_addr_type warp_addr, int pc_index)
{
    //m_stride_buffer[pc_index].m_warp_entry[warp_id].warp_ld_inst_num++;
    if(!m_stride_buffer[pc_index].m_warp_entry[warp_id].active){
        m_stride_buffer[pc_index].m_warp_entry[warp_id].active = true;
        m_stride_buffer[pc_index].active_warp_num++;
    }
    last_pc_index = pc_index;
    if(m_stride_buffer[pc_index].m_warp_entry[warp_id].intra_warp_last_addr != (unsigned)-1){
        long long stride = warp_addr - m_stride_buffer[pc_index].m_warp_entry[warp_id].intra_warp_last_addr;
        if(stride != 0){
            m_stride_buffer[pc_index].m_warp_entry[warp_id].intra_warp_stride.push_back(stride);
            // if(sid == 0)
            //     printf("pc:%d warp_id:%d intra_stride:%d\n",pc_index, warp_id, stride);
        }

    }
    m_stride_buffer[pc_index].m_warp_entry[warp_id].intra_warp_last_addr = warp_addr;
}
void cache_prefetch::calculate_intra_pref_addr(int sid, int wid, int cta_id){
    m_prefetch_req.valid = false;
    m_prefetch_req.put_time++;
    //if(last_pc_index == -1 || last_cta_id == -1)
    if(last_pc_index == -1)
        return;
    // if(sid == 0)
    //     printf("intra_pref\n");
    int warp_id = (wid + 10) % m_stride_buffer[last_pc_index].active_warp_num;
    if(m_stride_buffer[last_pc_index].m_warp_entry[warp_id].intra_stride != 0 && m_stride_buffer[last_pc_index].m_warp_entry[warp_id].valid_intra_stride == true){
        int prefetch_offset = 10; // prefetch adjacent warp is useless, need an offset
        m_prefetch_req.valid = true;
        m_prefetch_req.addr = m_stride_buffer[last_pc_index].m_warp_entry[warp_id].intra_warp_last_addr + 
        1 * m_stride_buffer[last_pc_index].m_warp_entry[warp_id].intra_stride;
        m_prefetch_req.warp_id = warp_id;
        m_prefetch_req.put_time = 0;
    }

}
void cache_prefetch::fill_entry(address_type pc)
{
    assert(fill_counter < 32);
    m_stride_buffer[fill_counter].pc = pc;
    fill_counter++;
}
bool cache_prefetch::full_entry()
{
    return (fill_counter == 32) ? true : false;
}
int cache_prefetch::probe_entry(address_type pc)
{
    for (int i = 0; i < fill_counter; i++)
    {
        if (m_stride_buffer[i].pc == pc)
            return i;
    }
    return -1;
}
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx) const
{
    //assert( m_config.m_write_policy == READ_ONLY );
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;

    bool all_reserved = true;
    //printf("access_num:%d\n",access_num++);/*cory*/
    // check for hit or pending hit
    for (unsigned way = 0; way < m_config.m_assoc; way++)
    {
        //printf("m_config.m_assoc:%d\n",m_config.m_assoc); //m_config.m_assoc 4 8
        unsigned index = set_index * m_config.m_assoc + way;
        cache_block_t *line = &m_lines[index];
        if (line->m_tag == tag)
        {
            if (line->m_status == RESERVED)
            {
                idx = index;
                return HIT_RESERVED;
            }
            else if (line->m_status == VALID)
            {
                idx = index;
                return HIT;
            }
            else if (line->m_status == MODIFIED)
            {
                idx = index;
                return HIT;
            }
            else
            {
                assert(line->m_status == INVALID);
            }
        }
        if (line->m_status != RESERVED)
        {
            all_reserved = false;
            if (line->m_status == INVALID)
            {
                invalid_line = index;
            }
            else
            {
                // valid line : keep track of most appropriate replacement candidate
                if (m_config.m_replacement_policy == LRU)
                {
                    if (line->m_last_access_time < valid_timestamp)
                    {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                }
                else if (m_config.m_replacement_policy == FIFO)
                {
                    if (line->m_alloc_time < valid_timestamp)
                    {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            }
        }
    }
    if (all_reserved)
    {
        assert(m_config.m_alloc_policy == ON_MISS);
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }

    if (invalid_line != (unsigned)-1)
    {
        idx = invalid_line;
    }
    else if (valid_line != (unsigned)-1)
    {
        idx = valid_line;
    }
    else
        abort(); // if an unreserved block exists, it is either invalid or replaceable

    assert(m_lines[idx].m_status != RESERVED);
    return MISS;
}
//int probe_locality_num = 0;

/*add by cory*/
enum cache_request_status tag_array::probe_locality(new_addr_type addr, unsigned &idx, mem_fetch *mf, unsigned time)
{
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;

    bool all_reserved = true;
    unsigned warp_id = mf->get_wid();
    int tag_buffer_index = m_wle->probe_entry(tag, mf, set_index);
    int tag_buffer_num = m_wle->m_tag_entry[set_index].fill_counter;
    int is_odd = warp_id % 2;
    int is_hit;
    int stride_buffer_index = m_cache_prefetch->probe_entry(mf->get_pc());
    if (USE_PREFETCH && stride_buffer_index == -1 && !m_cache_prefetch->full_entry() && mf->get_thread0_active())
        m_cache_prefetch->fill_entry(mf->get_pc());
    else if (USE_PREFETCH && stride_buffer_index != -1 && mf->get_thread0_active())
    {
        m_cache_prefetch->calculate_inter_warp_stride(mf->get_sid(), mf->get_ctaid(), mf->get_wid(), tag, stride_buffer_index);
        //m_cache_prefetch->calculate_intra_warp_stride(mf->get_sid(), mf->get_wid(), tag, stride_buffer_index);
        m_cache_prefetch->trace_warp_addr(mf->get_sid(), mf->get_ctaid(), mf->get_wid(), tag, stride_buffer_index);
    }
    if(USE_PREFETCH && time % 200 == 10 && time != 10 && stride_buffer_index != -1 ){
        for(int i = 0; i < m_cache_prefetch->fill_counter; i++){
            m_cache_prefetch->select_inter_stride(mf->get_sid(), mf->get_ctaid(), mf->get_wid(), tag, i);
            //m_cache_prefetch->select_intra_stride(mf->get_sid(), mf->get_ctaid(), mf->get_wid(), tag, i);
        }
    }

    /*cory*/
    for (unsigned way = 0; way < m_config.m_assoc; way++)
    {
        unsigned index = set_index * m_config.m_assoc + way;
        cache_block_t *line = &m_lines[index];

        if (line->m_tag == tag)
        {
            /*cory*/
            is_hit = 1;
            if (tag_buffer_index >= 0 && tag_buffer_index < tag_buffer_num)
            {
                if (m_wle->window_valid(set_index, tag_buffer_index, is_odd))
                {
                    m_wle->evaluate_locality(warp_id, set_index, tag_buffer_index, is_odd, is_hit);
                }
                m_wle->update_window(warp_id, set_index, tag_buffer_index, is_odd);
                m_wle->m_tag_entry[set_index].warp_record[tag_buffer_index].push_back(warp_id);
            }
            else if (tag_buffer_index == -1 && !m_wle->full_entry(set_index))
            {
                m_wle->fill_entry(tag, warp_id, set_index);
                m_wle->update_window(warp_id, set_index, tag_buffer_num, is_odd);
                m_wle->m_tag_entry[set_index].warp_record[tag_buffer_num].push_back(warp_id);
            }
            // else
            //     assert(0);
            //     printf("tag_buffer_index:%d\n",tag_buffer_index);
            m_wle->cache_hit_num++;

            if (line->m_status == RESERVED)
            {
                idx = index;
                return HIT_RESERVED;
            }
            else if (line->m_status == VALID)
            {
                idx = index;
                return HIT;
            }
            else if (line->m_status == MODIFIED)
            {
                idx = index;
                return HIT;
            }
            else
            {
                assert(line->m_status == INVALID);
            }
        }
        if (line->m_status != RESERVED)
        {
            all_reserved = false;
            if (line->m_status == INVALID)
            {
                invalid_line = index;
            }
            else
            {
                // valid line : keep track of most appropriate replacement candidate
                if (m_config.m_replacement_policy == LRU)
                {
                    if (line->m_last_access_time < valid_timestamp)
                    {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                }
                else if (m_config.m_replacement_policy == FIFO)
                {
                    if (line->m_alloc_time < valid_timestamp)
                    {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            }
        }
    }
    if (all_reserved)
    {
        assert(m_config.m_alloc_policy == ON_MISS);
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }

    if (invalid_line != (unsigned)-1)
    {
        idx = invalid_line;
    }
    else if (valid_line != (unsigned)-1)
    {
        idx = valid_line;
    }
    else
        abort(); // if an unreserved block exists, it is either invalid or replaceable

    /*cory*/
    is_hit = 0;
    if (tag_buffer_index >= 0 && tag_buffer_index < tag_buffer_num)
    {
        if (m_wle->window_valid(set_index, tag_buffer_index, is_odd))
        {
            m_wle->evaluate_locality(warp_id, set_index, tag_buffer_index, is_odd, is_hit);
        }
        m_wle->update_window(warp_id, set_index, tag_buffer_index, is_odd);
        m_wle->m_tag_entry[set_index].warp_record[tag_buffer_index].push_back(warp_id);
    }
    else if (tag_buffer_index == -1 && !m_wle->full_entry(set_index))
    {
        m_wle->fill_entry(tag, warp_id, set_index);
        m_wle->m_tag_entry[set_index].warp_record[tag_buffer_num].push_back(warp_id);
    }
    m_wle->cache_miss_num++;
    /*cory*/
    return MISS;
}

enum cache_request_status tag_array::access(new_addr_type addr, unsigned time, unsigned &idx)
{
    bool wb = false;
    cache_block_t evicted;
    enum cache_request_status result = access(addr, time, idx, wb, evicted);
    assert(!wb);
    return result;
}
unsigned last_alloc_time[512] = {0};
float tot_interval = 0;
float interval_times = 0;
enum cache_request_status tag_array::access(new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted)
{
    m_access++;
    //shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
    enum cache_request_status status = probe(addr, idx);
    /*cory* call a LDU function*/
    //printf("idx:%d line_status:%d status:%d\n", idx, m_lines[idx].m_status, status);
    switch (status)
    {
    case HIT_RESERVED:
        m_pending_hit++;
        assert(m_lines[idx].m_status == RESERVED);
        if(m_lines[idx].m_prefetch_line && !m_lines[idx].m_used){
            prefetch_hit_res_num++;
            if(m_core_id == 0)
                printf("hit_reserved:%d addr:%x\n", prefetch_hit_res_num, addr);
        }
        m_lines[idx].m_used=true;
        break;
    case HIT:
        m_lines[idx].m_last_access_time = time;
        assert(m_lines[idx].m_status == VALID || m_lines[idx].m_status == MODIFIED);
        if(m_lines[idx].m_prefetch_line && !m_lines[idx].m_used){
            prefetch_hit_num++;
            int interval = time - m_lines[idx].m_fill_time;
            float avg_interval = 0;
            if( m_core_id == 0 ){
                tot_interval += interval;
                interval_times++;
                avg_interval = tot_interval/interval_times;
                printf("hit_pref:%d addr:%x\n", prefetch_hit_num, addr);
            }
            if( m_core_id == 0 && m_lines[idx].m_used==false)
                printf("first_use_time:%d fill_time:%d interval:%d avg:%f\n", time, m_lines[idx].m_fill_time, interval, avg_interval);
        }
        m_lines[idx].m_used=true;
        break;
    case MISS:
        m_miss++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        if(m_lines[idx].m_prefetch_line && !m_lines[idx].m_used){
            useless_prefetch++;
            int interval = time - m_lines[idx].m_fill_time;
            if(m_core_id == 0)
                printf("useless_pref:%d addr:%x\ntime:%d fill_time:%d interval:%d\n", useless_prefetch, m_lines[idx].m_tag, time, m_lines[idx].m_fill_time, interval);
        }
        if (m_config.m_alloc_policy == ON_MISS)
        {
            assert(m_lines[idx].m_status == VALID || m_lines[idx].m_status==INVALID 
            || m_lines[idx].m_status==MODIFIED);
            if (m_lines[idx].m_status == MODIFIED)
            {
                wb = true;
                evicted = m_lines[idx];
            }
            //if(idx==2)
            //printf("alloc line_index:%d time:%d last_alloc_time:%d alloc_status:%d\n",idx, time, last_alloc_time[idx], m_lines[idx].m_status );
            m_lines[idx].allocate(m_config.tag(addr), m_config.block_addr(addr), time);
            last_alloc_time[idx] = time;
        }

        break;
    case RESERVATION_FAIL:
        m_res_fail++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        break;
    default:
        fprintf(stderr, "tag_array::access - Error: Unknown"
                        "cache_request_status %d\n",
                status);
        abort();
    }
    return status;
}
enum cache_request_status tag_array::pref_access(new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted, bool read_only)
{
    //m_access++;
    //shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
    enum cache_request_status status = probe(addr, idx);
    /*cory* call a LDU function/
    //printf("cache status:%d\n",status);   /*cory*/
    //printf("idx:%d line_status:%d \n", idx, m_lines[idx].m_status);
    switch (status)
    {
    case HIT_RESERVED:
        //m_pending_hit++;
    case HIT:
        //m_lines[idx].m_last_access_time=time;
        break;
    case MISS:
        //m_miss++;
        //shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        assert(m_lines[idx].m_status == VALID || m_lines[idx].m_status==INVALID 
        || m_lines[idx].m_status==MODIFIED);
        if (m_config.m_alloc_policy == ON_MISS)
        {
            if (m_lines[idx].m_status == MODIFIED && read_only)
            {
                return RESERVATION_FAIL;
            }
            else if (m_lines[idx].m_status == MODIFIED && !read_only){
                wb = true;
                evicted = m_lines[idx];
            }
            //if(idx==2)
                //printf("pref_alloc line_index:%d time:%d last_alloc_time:%d alloc_status:%d\n",idx, time, last_alloc_time[idx], m_lines[idx].m_status );
            m_lines[idx].allocate(m_config.tag(addr), m_config.block_addr(addr), time);
            last_alloc_time[idx] = time;
        }
        break;
    case RESERVATION_FAIL:
        //m_res_fail++;
        //shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        break;
    default:
        fprintf(stderr, "tag_array::access - Error: Unknown"
                        "cache_request_status %d\n",
                status);
        abort();
    }
    //printf("idx:%d line_status:%d status:%d\n", idx, m_lines[idx].m_status, status);
    return status;
}

void tag_array::fill(new_addr_type addr, unsigned time)
{
    assert(m_config.m_alloc_policy == ON_FILL);
    unsigned idx;
    enum cache_request_status status = probe(addr, idx);
    assert(status == MISS); // MSHR should have prevented redundant memory request
    m_lines[idx].allocate(m_config.tag(addr), m_config.block_addr(addr), time);
    m_lines[idx].fill(time);
}
unsigned last_fill_time[512] = {0};
void tag_array::pref_fill(unsigned index, unsigned time, bool is_prefetch)
{
    assert(m_config.m_alloc_policy == ON_MISS);
    //if(index==2)
    //printf("line index:%d time:%d last_fill_time:%d fill_status:%d\n", index, time, last_fill_time[index], m_lines[index].m_status);
    last_fill_time[index] = time;
    m_lines[index].pref_fill(time, is_prefetch);
}
void tag_array::fill(unsigned index, unsigned time)
{
    assert(m_config.m_alloc_policy == ON_MISS);
    //if(index==2)
    //printf("line index:%d time:%d last_fill_time:%d fill_status:%d\n", index, time, last_fill_time[index], m_lines[index].m_status);
    last_fill_time[index] = time;
    m_lines[index].fill(time);
}

void tag_array::flush()
{
    for (unsigned i = 0; i < m_config.get_num_lines(); i++)
        m_lines[i].m_status = INVALID;
}

float tag_array::windowed_miss_rate() const
{
    unsigned n_access = m_access - m_prev_snapshot_access;
    unsigned n_miss = m_miss - m_prev_snapshot_miss;
    // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

    float missrate = 0.0f;
    if (n_access != 0)
        missrate = (float)n_miss / n_access;
    return missrate;
}

void tag_array::new_window()
{
    m_prev_snapshot_access = m_access;
    m_prev_snapshot_miss = m_miss;
    m_prev_snapshot_pending_hit = m_pending_hit;
}

void tag_array::print(FILE *stream, unsigned &total_access, unsigned &total_misses) const
{
    m_config.print(stream);
    fprintf(stream, "\t\tAccess = %d, Miss = %d (%.3g), PendingHit = %d (%.3g)\n",
            m_access, m_miss, (float)m_miss / m_access,
            m_pending_hit, (float)m_pending_hit / m_access);
    total_misses += m_miss;
    total_access += m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses, unsigned &total_hit_res, unsigned &total_res_fail) const
{
    // Update statistics from the tag array
    total_access = m_access;
    total_misses = m_miss;
    total_hit_res = m_pending_hit;
    total_res_fail = m_res_fail;
}

bool was_write_sent(const std::list<cache_event> &events)
{
    for (std::list<cache_event>::const_iterator e = events.begin(); e != events.end(); e++)
    {
        if (*e == WRITE_REQUEST_SENT)
            return true;
    }
    return false;
}

bool was_writeback_sent(const std::list<cache_event> &events)
{
    for (std::list<cache_event>::const_iterator e = events.begin(); e != events.end(); e++)
    {
        if (*e == WRITE_BACK_REQUEST_SENT)
            return true;
    }
    return false;
}

bool was_read_sent(const std::list<cache_event> &events)
{
    for (std::list<cache_event>::const_iterator e = events.begin(); e != events.end(); e++)
    {
        if (*e == READ_REQUEST_SENT)
            return true;
    }
    return false;
}
/****************************************************************** MSHR ******************************************************************/

/// Checks if there is a pending request to the lower memory level already
bool mshr_table::probe(new_addr_type block_addr) const
{
    table::const_iterator a = m_data.find(block_addr);
    return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access
bool mshr_table::full(new_addr_type block_addr) const
{
    table::const_iterator i = m_data.find(block_addr);
    if (i != m_data.end())
        return i->second.m_list.size() >= m_max_merged;
    else
        return m_data.size() >= m_num_entries;
}

/// Add or merge this access
void mshr_table::add(new_addr_type block_addr, mem_fetch *mf)
{
    m_data[block_addr].m_list.push_back(mf);
    assert(m_data.size() <= m_num_entries);
    assert(m_data[block_addr].m_list.size() <= m_max_merged);
    // indicate that this MSHR entry contains an atomic operation
    if (mf->isatomic())
    {
        m_data[block_addr].m_has_atomic = true;
    }
}

/// Accept a new cache fill response: mark entry ready for processing
void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic)
{
    assert(!busy());
    table::iterator a = m_data.find(block_addr);
    assert(a != m_data.end()); // don't remove same request twice
    m_current_response.push_back(block_addr);
    has_atomic = a->second.m_has_atomic;
    assert(m_current_response.size() <= m_data.size());
}

/// Returns next ready access
mem_fetch *mshr_table::next_access()
{
    assert(access_ready());
    new_addr_type block_addr = m_current_response.front();
    assert(!m_data[block_addr].m_list.empty());
    mem_fetch *result = m_data[block_addr].m_list.front();
    m_data[block_addr].m_list.pop_front();
    if (m_data[block_addr].m_list.empty())
    {
        // release entry
        m_data.erase(block_addr);
        m_current_response.pop_front();
    }
    return result;
}

void mshr_table::display(FILE *fp) const
{
    fprintf(fp, "MSHR contents\n");
    for (table::const_iterator e = m_data.begin(); e != m_data.end(); ++e)
    {
        unsigned block_addr = e->first;
        fprintf(fp, "MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr, e->second.m_has_atomic, e->second.m_list.size());
        if (!e->second.m_list.empty())
        {
            mem_fetch *mf = e->second.m_list.front();
            fprintf(fp, "%p :", mf);
            mf->print(fp);
        }
        else
        {
            fprintf(fp, " no memory requests???\n");
        }
    }
}
/***************************************************************** Caches *****************************************************************/
cache_stats::cache_stats()
{
    m_stats.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i)
    {
        m_stats[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_cache_port_available_cycles = 0;
    m_cache_data_port_busy_cycles = 0;
    m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear()
{
    ///
    /// Zero out all current cache statistics
    ///
    for (unsigned i = 0; i < NUM_MEM_ACCESS_TYPE; ++i)
    {
        std::fill(m_stats[i].begin(), m_stats[i].end(), 0);
    }
    m_cache_port_available_cycles = 0;
    m_cache_data_port_busy_cycles = 0;
    m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::inc_stats(int access_type, int access_outcome)
{
    ///
    /// Increment the stat corresponding to (access_type, access_outcome) by 1.
    ///
    if (!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    m_stats[access_type][access_outcome]++;
}

enum cache_request_status cache_stats::select_stats_status(enum cache_request_status probe, enum cache_request_status access) const
{
    ///
    /// This function selects how the cache access outcome should be counted. HIT_RESERVED is considered as a MISS
    /// in the cores, however, it should be counted as a HIT_RESERVED in the caches.
    ///
    if (probe == HIT_RESERVED && access != RESERVATION_FAIL)
        return probe;
    else
        return access;
}

unsigned &cache_stats::operator()(int access_type, int access_outcome)
{
    ///
    /// Simple method to read/modify the stat corresponding to (access_type, access_outcome)
    /// Used overloaded () to avoid the need for separate read/write member functions
    ///
    if (!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
}

unsigned cache_stats::operator()(int access_type, int access_outcome) const
{
    ///
    /// Const accessor into m_stats.
    ///
    if (!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
}

cache_stats cache_stats::operator+(const cache_stats &cs)
{
    ///
    /// Overloaded + operator to allow for simple stat accumulation
    ///
    cache_stats ret;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type)
    {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status)
        {
            ret(type, status) = m_stats[type][status] + cs(type, status);
        }
    }
    ret.m_cache_port_available_cycles = m_cache_port_available_cycles + cs.m_cache_port_available_cycles;
    ret.m_cache_data_port_busy_cycles = m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles;
    ret.m_cache_fill_port_busy_cycles = m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles;
    return ret;
}

cache_stats &cache_stats::operator+=(const cache_stats &cs)
{
    ///
    /// Overloaded += operator to allow for simple stat accumulation
    ///
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type)
    {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status)
        {
            m_stats[type][status] += cs(type, status);
        }
    }
    m_cache_port_available_cycles += cs.m_cache_port_available_cycles;
    m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles;
    m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles;
    return *this;
}

void cache_stats::print_stats(FILE *fout, const char *cache_name) const
{
    ///
    /// Print out each non-zero cache statistic for every memory access type and status
    /// "cache_name" defaults to "Cache_stats" when no argument is provided, otherwise
    /// the provided name is used.
    /// The printed format is "<cache_name>[<request_type>][<request_status>] = <stat_value>"
    ///
    std::string m_cache_name = cache_name;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type)
    {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status)
        {
            if (m_stats[type][status] > 0)
            {
                fprintf(fout, "\t%s[%s][%s] = %u\n",
                        m_cache_name.c_str(),
                        mem_access_type_str((enum mem_access_type)type),
                        cache_request_status_str((enum cache_request_status)status),
                        m_stats[type][status]);
            }
        }
    }
}

void cache_sub_stats::print_port_stats(FILE *fout, const char *cache_name) const
{
    float data_port_util = 0.0f;
    if (port_available_cycles > 0)
    {
        data_port_util = (float)data_port_busy_cycles / port_available_cycles;
    }
    fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util);
    float fill_port_util = 0.0f;
    if (port_available_cycles > 0)
    {
        fill_port_util = (float)fill_port_busy_cycles / port_available_cycles;
    }
    fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util);
}

unsigned cache_stats::get_stats(enum mem_access_type *access_type, unsigned num_access_type, enum cache_request_status *access_status, unsigned num_access_status) const
{
    ///
    /// Returns a sum of the stats corresponding to each "access_type" and "access_status" pair.
    /// "access_type" is an array of "num_access_type" mem_access_types.
    /// "access_status" is an array of "num_access_status" cache_request_statuses.
    ///
    unsigned total = 0;
    for (unsigned type = 0; type < num_access_type; ++type)
    {
        for (unsigned status = 0; status < num_access_status; ++status)
        {
            if (!check_valid((int)access_type[type], (int)access_status[status]))
                assert(0 && "Unknown cache access type or access outcome");
            total += m_stats[access_type[type]][access_status[status]];
        }
    }
    return total;
}
void cache_stats::get_sub_stats(struct cache_sub_stats &css) const
{
    ///
    /// Overwrites "css" with the appropriate statistics from this cache.
    ///
    struct cache_sub_stats t_css;
    t_css.clear();

    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type)
    {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status)
        {
            if (status == HIT || status == MISS || status == HIT_RESERVED)
                t_css.accesses += m_stats[type][status];

            if (status == MISS)
                t_css.misses += m_stats[type][status];

            if (status == HIT_RESERVED)
                t_css.pending_hits += m_stats[type][status];

            if (status == RESERVATION_FAIL)
                t_css.res_fails += m_stats[type][status];
        }
    }

    t_css.port_available_cycles = m_cache_port_available_cycles;
    t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles;
    t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles;

    css = t_css;
}

bool cache_stats::check_valid(int type, int status) const
{
    ///
    /// Verify a valid access_type/access_status
    ///
    if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) && (status < NUM_CACHE_REQUEST_STATUS))
        return true;
    else
        return false;
}

void cache_stats::sample_cache_port_utility(bool data_port_busy, bool fill_port_busy)
{
    m_cache_port_available_cycles += 1;
    if (data_port_busy)
    {
        m_cache_data_port_busy_cycles += 1;
    }
    if (fill_port_busy)
    {
        m_cache_fill_port_busy_cycles += 1;
    }
}

baseline_cache::bandwidth_management::bandwidth_management(cache_config &config)
    : m_config(config)
{
    m_data_port_occupied_cycles = 0;
    m_fill_port_occupied_cycles = 0;
}

/// use the data port based on the outcome and events generated by the mem_fetch request
void baseline_cache::bandwidth_management::use_data_port(mem_fetch *mf, enum cache_request_status outcome, const std::list<cache_event> &events)
{
    unsigned data_size = mf->get_data_size();
    unsigned port_width = m_config.m_data_port_width;
    switch (outcome)
    {
    case HIT:
    {
        unsigned data_cycles = data_size / port_width + ((data_size % port_width > 0) ? 1 : 0);
        m_data_port_occupied_cycles += data_cycles;
    }
    break;
    case HIT_RESERVED:
    case MISS:
    {
        // the data array is accessed to read out the entire line for write-back
        if (was_writeback_sent(events))
        {
            unsigned data_cycles = m_config.m_line_sz / port_width;
            m_data_port_occupied_cycles += data_cycles;
        }
    }
    break;
    case RESERVATION_FAIL:
        // Does not consume any port bandwidth
        break;
    default:
        assert(0);
        break;
    }
}

/// use the fill port
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf)
{
    // assume filling the entire line with the returned request
    unsigned fill_cycles = m_config.m_line_sz / m_config.m_data_port_width;
    m_fill_port_occupied_cycles += fill_cycles;
}

/// called every cache cycle to free up the ports
void baseline_cache::bandwidth_management::replenish_port_bandwidth()
{
    if (m_data_port_occupied_cycles > 0)
    {
        m_data_port_occupied_cycles -= 1;
    }
    assert(m_data_port_occupied_cycles >= 0);

    if (m_fill_port_occupied_cycles > 0)
    {
        m_fill_port_occupied_cycles -= 1;
    }
    assert(m_fill_port_occupied_cycles >= 0);
}

/// query for data port availability
bool baseline_cache::bandwidth_management::data_port_free() const
{
    return (m_data_port_occupied_cycles == 0);
}

/// query for fill port availability
bool baseline_cache::bandwidth_management::fill_port_free() const
{
    return (m_fill_port_occupied_cycles == 0);
}

/// Sends next request to lower level of memory
void baseline_cache::cycle()
{
    if (!m_pref_miss_queue.empty())
    {
        mem_fetch *mf = m_pref_miss_queue.front();
        if (!m_memport->full(mf->size(), mf->get_is_write()))
        {
            //if(mf->get_sid()==0)
                //printf("actual_send_addr:%x warp_id:%d alloc_time:%d\n", mf->get_addr(), mf->get_wid(), mf->get_timestamp());
            m_pref_miss_queue.pop_front();
            m_memport->push(mf);
        }
    }
    else if (!m_miss_queue.empty())
    {
        mem_fetch *mf = m_miss_queue.front();
        // mem_fetch *prefetch_mf = mf;
        // new_addr_type prefetch_addr = mf->get_addr()+128;
        // prefetch_mf->set_addr(prefetch_addr);
        // if(mf->get_sid()==0 && mf->check_pair==true){
        //     printf("process miss queue, addr:%x wid:%d req_id:%d time:%d\n", mf->get_addr(),
        //     mf->get_wid(), mf->get_request_uid(), mf->get_timestamp());
        // }
        if (!m_memport->full(mf->size(), mf->get_is_write()))
        {
            m_miss_queue.pop_front();
            m_memport->push(mf);
        }
        // if ( !m_memport->full(mf->size(),mf->get_is_write()) ) {
        //     m_memport->push(prefetch_mf);
        // }
    }

    bool data_port_busy = !m_bandwidth_management.data_port_free();
    bool fill_port_busy = !m_bandwidth_management.fill_port_free();
    m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
    m_bandwidth_management.replenish_port_bandwidth();
}

/// Interface for response from lower memory level (model bandwidth restictions in caller)
void baseline_cache::fill(mem_fetch *mf, unsigned time)
{
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    assert(e != m_extra_mf_fields.end());
    assert(e->second.m_valid);
    //if(mf->get_is_prefetch() && mf->get_sid() == 0)
    //if( mf->get_sid() == 0)
        //printf("actual_fill_addr:%x warp_id:%d alloc_time:%d\n", mf->get_addr(), mf->get_wid(), mf->get_timestamp());
    mf->set_data_size(e->second.m_data_size);
    if (m_config.m_alloc_policy == ON_MISS){ //m_config.m_alloc_policy is ON_MISS, like non-blocking?
        // if(e->first->get_is_prefetch())
        // printf("addr:%x set_index:%d is_prefetch:%d sid:%d ", 
        //  e->first->get_addr(), m_config.set_index(e->first->get_addr()), e->first->get_is_prefetch(), mf->get_sid());
        //m_tag_array->fill(e->second.m_cache_index, time);
        m_tag_array->pref_fill(e->second.m_cache_index, time, mf->get_is_prefetch());
    }
    else if (m_config.m_alloc_policy == ON_FILL)
        m_tag_array->fill(e->second.m_block_addr, time);
    else
        abort();
    bool has_atomic = false;
    m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
    if (has_atomic)
    {
        assert(m_config.m_alloc_policy == ON_MISS);
        cache_block_t &block = m_tag_array->get_block(e->second.m_cache_index);
        block.m_status = MODIFIED; // mark line as dirty for atomic operation
    }
    m_extra_mf_fields.erase(mf);
    m_bandwidth_management.use_fill_port(mf);
}

/// Checks if mf is waiting to be filled by lower memory level
bool baseline_cache::waiting_for_fill(mem_fetch *mf)
{
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses, unsigned &misses) const
{
    fprintf(fp, "Cache %s:\t", m_name.c_str());
    m_tag_array->print(fp, accesses, misses);
}

void baseline_cache::display_state(FILE *fp) const
{
    fprintf(fp, "Cache %s:\n", m_name.c_str());
    m_mshrs.display(fp);
    fprintf(fp, "\n");
}

/// Read miss handler without writeback
void baseline_cache::send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss, std::list<cache_event> &events, bool read_only, bool wa)
{

    bool wb = false;
    cache_block_t e;
    send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e, events, read_only, wa);
}

/// Read miss handler. Check MSHR hit or MSHR available
//int mshr_avail_fail_time = 0;
void baseline_cache::send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf, 
                                       unsigned time, bool &do_miss, bool &wb, cache_block_t &evicted, std::list<cache_event> &events, bool read_only, bool wa)
{
    bool mshr_hit = m_mshrs.probe(block_addr);
    bool mshr_avail = !m_mshrs.full(block_addr);
    // if(!mshr_avail && mf->get_sid()==0){
    //     printf("mshr_avail_fail_time:%d\n", mshr_avail_fail_time++);
    // }
    if (mshr_hit && mshr_avail)
    {
        if (read_only)
            m_tag_array->access(block_addr, time, cache_index);
        else
            m_tag_array->access(block_addr, time, cache_index, wb, evicted);

        m_mshrs.add(block_addr, mf);
        do_miss = true;
    }
    else if (!mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size))
    {
        if (read_only)
            m_tag_array->access(block_addr, time, cache_index);
        else
            m_tag_array->access(block_addr, time, cache_index, wb, evicted);
        //if(mf->get_sid() == 0)
            //printf("demand_push_addr:%x time:%d\n", block_addr, time);
        m_mshrs.add(block_addr, mf);
        m_extra_mf_fields[mf] = extra_mf_fields(block_addr, cache_index, mf->get_data_size());
        mf->set_data_size(m_config.get_line_sz());
        m_miss_queue.push_back(mf);
        mf->set_status(m_miss_queue_status, time);
        if (!wa)
            events.push_back(READ_REQUEST_SENT);
        do_miss = true;
    }
}
int delete_pref = 0;
int actual_push_time = 0;
int mshr_full = 0;
int mshr_hit = 0;
int cache_hit = 0;
void baseline_cache::send_read_request_pref(new_addr_type pref_addr, new_addr_type pref_block_addr, unsigned pref_cache_index, mem_fetch *pref_mf,
                                           unsigned time, bool &pref_do_miss, bool &pref_wb, cache_block_t &pref_evicted, std::list<cache_event> &events, bool read_only, bool wa)
{
    //printf("pref req\n");
    bool pref_mshr_hit = m_mshrs.probe(pref_block_addr);
    bool pref_mshr_avail = !m_mshrs.full(pref_block_addr);
    if ( pref_mshr_hit) {
        //printf("delete pref mf:%d\n",delete_pref++);
        // if(pref_mf->get_sid() == 0)
        //     printf("mshr_hit:%d\n", mshr_hit++);
        delete pref_mf;
        pref_mf = NULL;
    }
    else if (!pref_mshr_hit && pref_mshr_avail && (m_pref_miss_queue.size() < m_config.m_miss_queue_size) && (m_miss_queue.size() < 3))
    //else if (!pref_mshr_hit && pref_mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size))
    {
        cache_request_status pref_status = m_tag_array->pref_access(pref_block_addr, time, pref_cache_index, pref_wb, pref_evicted, read_only);
        if (pref_status == MISS)
        {
            //assert(m_pref_miss_queue.size() <= 1);
            m_mshrs.add(pref_block_addr,pref_mf);
            // if(pref_block_addr == 0x80059200)
            //     printf("pref_block_addr:%x cache_index:%d\n", pref_block_addr, pref_cache_index);
            if(pref_mf->get_sid() == 0)
              printf("actual_push_addr:%x times:%d time:%d\n", pref_block_addr, actual_push_time++, time);
            m_extra_mf_fields[pref_mf] = extra_mf_fields(pref_block_addr, pref_cache_index, pref_mf->get_data_size());
            pref_mf->set_data_size(m_config.get_line_sz());
            //printf("push into pref_miss_queue\n");
            // if(!m_pref_miss_queue.empty()){
            //     mem_fetch *mf = m_pref_miss_queue.front();
            //     if(time - mf->get_timestamp() > 30){
            //         printf("pref_req_stall_too_long:%d\n", time - mf->get_timestamp());
            //     }
            // }
            m_pref_miss_queue.push_back(pref_mf);
            //m_miss_queue.push_back(pref_mf);
            pref_mf->set_status(m_miss_queue_status, time);
            //   if(!wa)
            //   	events.push_back(READ_REQUEST_SENT);
            pref_do_miss = true;
            //printf("virtual delete pref mf:%d\n",delete_pref++);
        }
        else{
            // if(pref_mf->get_sid() == 0)
            //     printf("cache_hit/res_fail:%d\n", cache_hit++);
            //printf("delete pref mf:%d\n",delete_pref++);
            delete pref_mf;
            pref_mf = NULL;
        }
    }
    else{
        // if(pref_mf->get_sid() == 0)
        //     printf("mshr_full:%d\n", mshr_full++);
        //printf("delete pref mf:%d\n",delete_pref++);
        delete pref_mf;
        pref_mf = NULL;
    }
}
/// Read miss handler. Check MSHR hit or MSHR available
void baseline_cache::send_read_request_l1d(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, unsigned pref_cache_index, mem_fetch *mf, mem_fetch *pref_mf,
                                           unsigned time, bool &do_miss, bool &pref_do_miss, bool &wb, bool &pref_wb, cache_block_t &evicted, cache_block_t &pref_evicted, std::list<cache_event> &events, bool read_only, bool wa)
{

    bool mshr_hit = m_mshrs.probe(block_addr);
    bool mshr_avail = !m_mshrs.full(block_addr);
    new_addr_type pref_addr = pref_mf->get_addr();
    new_addr_type pref_block_addr = m_config.block_addr(pref_addr);
    /*cory*/
    //mf->set_prefetch();
    /*cory*/

    if (mshr_hit && mshr_avail)
    {
        if (read_only)
            m_tag_array->access(block_addr, time, cache_index);
        else
            m_tag_array->access(block_addr, time, cache_index, wb, evicted);

        m_mshrs.add(block_addr, mf);
        do_miss = true;
    }
    else if (!mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size))
    { //m_config.m_miss_queue_size = 8
        if (read_only)
            m_tag_array->access(block_addr, time, cache_index);
        else
            m_tag_array->access(block_addr, time, cache_index, wb, evicted);

        m_mshrs.add(block_addr, mf);
        m_extra_mf_fields[mf] = extra_mf_fields(block_addr, cache_index, mf->get_data_size());
        mf->set_data_size(m_config.get_line_sz());
        m_miss_queue.push_back(mf);
        mf->set_status(m_miss_queue_status, time);
        if (!wa)
            events.push_back(READ_REQUEST_SENT);
        do_miss = true;
    }
    bool pref_mshr_hit = m_mshrs.probe(pref_block_addr);
    bool pref_mshr_avail = !m_mshrs.full(pref_block_addr);
    if ( pref_mshr_hit && pref_mshr_avail ) {
        delete pref_mf;
    }
    else if (!pref_mshr_hit && pref_mshr_avail && (m_pref_miss_queue.size() < m_config.m_miss_queue_size))
    {
        cache_request_status pref_status = m_tag_array->pref_access(pref_block_addr, time, pref_cache_index, pref_wb, pref_evicted, read_only);
        if (pref_status == MISS)
        {
            m_mshrs.add(pref_block_addr,pref_mf);
            m_extra_mf_fields[pref_mf] = extra_mf_fields(pref_block_addr, pref_cache_index, pref_mf->get_data_size());
            pref_mf->set_data_size(m_config.get_line_sz());
            //printf("push into pref_miss_queue\n");
            m_pref_miss_queue.push_back(pref_mf);
            pref_mf->set_status(m_miss_queue_status, time);
            //   if(!wa)
            //   	events.push_back(READ_REQUEST_SENT);
            pref_do_miss = true;
        }
        // else{
        //     delete pref_mf;
        // }
    }
}

/// Sends write request to lower level memory (write or writeback)
void data_cache::send_write_request(mem_fetch *mf, cache_event request, unsigned time, std::list<cache_event> &events)
{
    events.push_back(request);
    m_miss_queue.push_back(mf);
    mf->set_status(m_miss_queue_status, time);
}

void data_cache::send_write_request_pref(mem_fetch *pref_mf, cache_event request, unsigned time, std::list<cache_event> &events)
{
    //events.push_back(request);
    m_pref_miss_queue.push_back(pref_mf);
    pref_mf->set_status(m_miss_queue_status, time);
}

/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
cache_request_status data_cache::wr_hit_wb(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status)
{
    new_addr_type block_addr = m_config.block_addr(addr);
    m_tag_array->access(block_addr, time, cache_index); // update LRU state
    cache_block_t &block = m_tag_array->get_block(cache_index);
    block.m_status = MODIFIED;

    return HIT;
}

/// Write-through hit: Directly send request to lower level memory
cache_request_status data_cache::wr_hit_wt(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status)
{
    if (miss_queue_full(0))
        return RESERVATION_FAIL; // cannot handle request this cycle

    new_addr_type block_addr = m_config.block_addr(addr);
    m_tag_array->access(block_addr, time, cache_index); // update LRU state
    cache_block_t &block = m_tag_array->get_block(cache_index);
    block.m_status = MODIFIED;

    // generate a write-through
    send_write_request(mf, WRITE_REQUEST_SENT, time, events);

    return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate corresponding block
cache_request_status data_cache::wr_hit_we(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status)
{
    if (miss_queue_full(0))
        return RESERVATION_FAIL; // cannot handle request this cycle

    // generate a write-through/evict
    cache_block_t &block = m_tag_array->get_block(cache_index);
    send_write_request(mf, WRITE_REQUEST_SENT, time, events);

    // Invalidate block
    block.m_status = INVALID;

    return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status data_cache::wr_hit_global_we_local_wb(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status)
{
    bool evict = (mf->get_access_type() == GLOBAL_ACC_W); // evict a line that hits on global memory write
    if (evict)
        return wr_hit_we(addr, cache_index, mf, time, events, status); // Write-evict
    else
        return wr_hit_wb(addr, cache_index, mf, time, events, status); // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
enum cache_request_status
data_cache::wr_miss_wa(new_addr_type addr,
                       unsigned cache_index, mem_fetch *mf,
                       unsigned time, std::list<cache_event> &events,
                       enum cache_request_status status)
{
    new_addr_type block_addr = m_config.block_addr(addr);

    // Write allocate, maximum 3 requests (write miss, read request, write back request)
    // Conservatively ensure the worst-case request can be handled this cycle
    bool mshr_hit = m_mshrs.probe(block_addr);
    bool mshr_avail = !m_mshrs.full(block_addr);
    if (miss_queue_full(2) || (!(mshr_hit && mshr_avail) && !(!mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size))))
        return RESERVATION_FAIL;

    send_write_request(mf, WRITE_REQUEST_SENT, time, events);
    // Tries to send write allocate request, returns true on success and false on failure
    //if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
    //    return RESERVATION_FAIL;

    const mem_access_t *ma = new mem_access_t(m_wr_alloc_type,
                                              mf->get_addr(),
                                              mf->get_data_size(),
                                              false, // Now performing a read
                                              mf->get_access_warp_mask(),
                                              mf->get_access_byte_mask());

    mem_fetch *n_mf = new mem_fetch(*ma,
                                    NULL,
                                    mf->get_ctrl_size(),
                                    mf->get_wid(),
                                    mf->get_sid(),
                                    mf->get_tpc(),
                                    mf->get_mem_config());

    bool do_miss = false;
    bool wb = false;
    cache_block_t evicted;

    // Send read request resulting from write miss
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb, //cory, second mf is no use
                      evicted, events, false, true);

    if (do_miss)
    {
        // If evicted block is modified and not a write-through
        // (already modified lower level)
        if (wb && (m_config.m_write_policy != WRITE_THROUGH))
        {
            mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr,
                                                      m_wrbk_type, m_config.get_line_sz(), true);
            m_miss_queue.push_back(wb);
            wb->set_status(m_miss_queue_status, time);
        }
        return MISS;
    }

    return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status
data_cache::wr_miss_no_wa(new_addr_type addr,
                          unsigned cache_index,
                          mem_fetch *mf,
                          unsigned time,
                          std::list<cache_event> &events,
                          enum cache_request_status status)
{
    if (miss_queue_full(0))
        return RESERVATION_FAIL; // cannot handle request this cycle

    // on miss, generate write through (no write buffering -- too many threads for that)
    send_write_request(mf, WRITE_REQUEST_SENT, time, events);

    return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
enum cache_request_status
data_cache::rd_hit_base(new_addr_type addr,
                        unsigned cache_index,
                        mem_fetch *mf,
                        unsigned time,
                        std::list<cache_event> &events,
                        enum cache_request_status status)
{
    new_addr_type block_addr = m_config.block_addr(addr);
    m_tag_array->access(block_addr, time, cache_index);
    // Atomics treated as global read/write requests - Perform read, mark line as
    // MODIFIED
    if (mf->isatomic())
    {
        assert(mf->get_access_type() == GLOBAL_ACC_R);
        cache_block_t &block = m_tag_array->get_block(cache_index);
        block.m_status = MODIFIED; // mark line as dirty
    }
    return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
enum cache_request_status
data_cache::rd_miss_base(new_addr_type addr,
                         unsigned cache_index,
                         mem_fetch *mf,
                         unsigned time,
                         std::list<cache_event> &events,
                         enum cache_request_status status,
                         bool is_l1_cache)
{
    if (miss_queue_full(1))
        // cannot handle request this cycle
        // (might need to generate two requests)
        return RESERVATION_FAIL;

    new_addr_type block_addr = m_config.block_addr(addr);
    //printf("cory, addr:%x, block_addr:%x\n",addr,block_addr);
    bool do_miss = false;
    bool wb = false;
    cache_block_t evicted;
    // bool pref_do_miss = false;
    // bool pref_wb = false;
    // cache_block_t pref_evicted;
    // if (is_l1_cache && USE_PREFETCH)
    //     send_read_request_l1d(addr,
    //                           block_addr,
    //                           cache_index, pref_cache_index,
    //                           mf, pref_mf, time, do_miss, pref_do_miss, wb, pref_wb, evicted, pref_evicted, events, false, false);
    // else{
        send_read_request(addr,
                    block_addr,
                    cache_index,
                    mf, time, do_miss, wb, evicted, events, false, false);
    //}
    // if (pref_do_miss)
    // {
    //     if (pref_wb && (m_config.m_write_policy != WRITE_THROUGH))
    //     {
    //         mem_fetch *pref_wb = m_memfetch_creator->alloc(pref_evicted.m_block_addr,
    //                                                        m_wrbk_type, m_config.get_line_sz(), true);
    //         send_write_request_pref(pref_wb, WRITE_BACK_REQUEST_SENT, time, events); //wb == 1 means cache line is modified, it needs to write back to memory
    //     }
    // }
    if (do_miss)
    {
        // If evicted block is modified and not a write-through
        // (already modified lower level)
        if (wb && (m_config.m_write_policy != WRITE_THROUGH))
        {
            mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr,
                                                      m_wrbk_type, m_config.get_line_sz(), true);
            send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events); //wb == 1 means cache line is modified, it needs to write back to memory
        }
        return MISS;
    }
    return RESERVATION_FAIL;
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
//int read_only_access = 0;
enum cache_request_status
read_only_cache::access(new_addr_type addr,
                        mem_fetch *mf,
                        mem_fetch *pref_mf,
                        unsigned time,
                        std::list<cache_event> &events)
{
    //printf("read_only_access:%d\n",read_only_access++); /*cory*/
    // if(pref_mf != NULL){
    //     printf("delete pref mf:%d\n",delete_pref++);
    //     delete pref_mf;
    //     pref_mf = NULL;
    // }
    assert(mf->get_data_size() <= m_config.get_line_sz());
    assert(m_config.m_write_policy == READ_ONLY);
    assert(!mf->get_is_write());
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status status = m_tag_array->probe(block_addr, cache_index);
    enum cache_request_status cache_status = RESERVATION_FAIL;

    if (status == HIT)
    {
        cache_status = m_tag_array->access(block_addr, time, cache_index); // update LRU state
    }
    else if (status != RESERVATION_FAIL)
    {
        if (!miss_queue_full(0))
        {
            bool do_miss = false;
            /*cory, second mf is no use*/
            send_read_request(addr, block_addr, cache_index, mf, time, do_miss, events, true, false);
            if (do_miss)
                cache_status = MISS;
            else
                cache_status = RESERVATION_FAIL;
        }
        else
        {
            cache_status = RESERVATION_FAIL;
        }
    }
    unsigned pref_cache_index = (unsigned)-1;
    // if(USE_PREFETCH && pref_mf != NULL ){
    //     //m_tag_array->m_cache_prefetch->calculate_pref_addr(mf->get_sid(), mf->get_wid(), mf->get_ctaid());
    //     assert(pref_mf != NULL);
    //     bool pref_do_miss = false;
    //     bool pref_wb = false;
    //     cache_block_t pref_evicted;
    //     new_addr_type pref_addr = m_tag_array->get_prefetch_addr();
    //     //printf("pref_addr:%x time:%d\n", pref_addr, time);
    //     //new_addr_type pref_addr = mf->get_addr() + 256;
    //     if(pref_addr > 0x80000000 && pref_addr <= 0x803fffff && (m_tag_array->valid_addr() || (!m_tag_array->valid_addr()) &&
    //     m_tag_array->get_put_time() < 10)){ //jugde prefetch addr is valid
    //         new_addr_type pref_block_addr = m_config.block_addr(pref_addr);
    //         //if(mf->get_sid() == 0)
    //             //printf("actual_pref_addr:%x time:%d\n", pref_block_addr, time);
    //         pref_mf->set_warp_id(m_tag_array->get_prefetch_warp_id());
    //         pref_mf->set_addr(pref_block_addr);
    //         send_read_request_pref(pref_addr, pref_block_addr,
    //                     pref_cache_index, pref_mf, time, 
    //                     pref_do_miss, pref_wb, pref_evicted, events, true, false);
    //     }
    //     else{
    //         //printf("delete pref mf:%d\n",delete_pref++);
    //         delete pref_mf;
    //         pref_mf = NULL;
    //     }
    // }
    m_stats.inc_stats(mf->get_access_type(), m_stats.select_stats_status(status, cache_status));
    return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
enum cache_request_status
data_cache::process_tag_probe(bool wr,
                              enum cache_request_status probe_status,
                              new_addr_type addr,
                              unsigned cache_index,
                              mem_fetch *mf,
                              unsigned time,
                              std::list<cache_event> &events,
                              bool is_l1_cache)
{
    // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
    // data_cache constructor to reflect the corresponding cache configuration
    // options. Function pointers were used to avoid many long conditional
    // branches resulting from many cache configuration options.
    cache_request_status access_status = probe_status;
    if (wr)
    { // Write
        if (probe_status == HIT)
        {
            access_status = (this->*m_wr_hit)(addr,
                                              cache_index,
                                              mf, time, events, probe_status);
        }
        else if (probe_status != RESERVATION_FAIL)
        { //Why != Reservation Fail do nothing? MSHR is full, can't process any access, thus just wait, just stall
            access_status = (this->*m_wr_miss)(addr,
                                               cache_index,
                                               mf, time, events, probe_status);
        }
    }
    else
    { // Read
        if (probe_status == HIT)
        {
            access_status = (this->*m_rd_hit)(addr,
                                              cache_index,
                                              mf, time, events, probe_status);
        }
        else if (probe_status != RESERVATION_FAIL)
        {
            access_status = (this->*m_rd_miss)(addr,
                                               cache_index, 
                                               mf,  time, events, probe_status, is_l1_cache);
        }
    }

    m_bandwidth_management.use_data_port(mf, access_status, events);
    return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.
int actual_pref_times = 0;
enum cache_request_status
data_cache::access(new_addr_type addr,
                   mem_fetch *mf,
                   mem_fetch *pref_mf,
                   unsigned time,
                   std::list<cache_event> &events,
                   bool is_l1_cache)
{
    assert(mf->get_data_size() <= m_config.get_line_sz());
    bool wr = mf->get_is_write();
    new_addr_type block_addr = m_config.block_addr(addr);
    //new_addr_type pref_block_addr = m_config.block_addr(pref_mf->get_addr());
    unsigned pref_cache_index = (unsigned)-1;
    //enum cache_request_status pref_probe_status;
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status probe_status;
    if ((mf->get_access_type() == GLOBAL_ACC_R || mf->get_access_type() == LOCAL_ACC_R) && is_l1_cache)
    {
        probe_status = m_tag_array->probe_locality(block_addr, cache_index, mf, time);
        //  if(mf->get_sid()==4);
        //      printf("demand addr:%x demand warp_id:%d\n ", mf->get_addr(), mf->get_wid());
        //pref_probe_status = m_tag_array->probe(pref_block_addr, pref_cache_index);
    }
    else
    {
        probe_status = m_tag_array->probe(block_addr, cache_index);
    }
    enum cache_request_status access_status = process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events, is_l1_cache);

    //if(is_l1_cache && (mf->get_access_type() == GLOBAL_ACC_R || mf->get_access_type() == LOCAL_ACC_R)){
    if(USE_PREFETCH && is_l1_cache && (mf->get_access_type() == GLOBAL_ACC_R || mf->get_access_type() == LOCAL_ACC_R ) ){
        if(scheduler_policy_gto == 0)
            m_tag_array->m_cache_prefetch->calculate_inter_pref_addr(mf->get_sid(), mf->get_wid(), mf->get_ctaid());
        else
            m_tag_array->m_cache_prefetch->m_prefetch_req.valid = false;
            // m_tag_array->m_cache_prefetch->calculate_intra_pref_addr(mf->get_sid(), mf->get_wid(), mf->get_ctaid());
        // if(mf->get_sid() == 0){
        //     printf("wid:%d\n", mf->get_wid());
        // }
    }
    //if(USE_PREFETCH && pref_mf != NULL && is_l1_cache){
    if(USE_PREFETCH && pref_mf != NULL && is_l1_cache){
        assert(pref_mf != NULL);
        bool pref_do_miss = false;
        bool pref_wb = false;
        cache_block_t pref_evicted;
        new_addr_type pref_addr = m_tag_array->get_prefetch_addr();
        //pref_addr = 0x80000000 + time * 128; //useless_pref_addr test

        //printf("pref_addr:%x time:%d\n", pref_addr, time);
        //pref_addr = mf->get_addr() + 256;
        if(pref_addr > 0x00000000 && pref_addr <= 0x8fffffff && (m_tag_array->valid_addr() 
        || (!m_tag_array->valid_addr()
        && m_tag_array->get_put_time() < 10
        ))){ //jugde prefetch addr is valid
        //if(pref_addr > 0x00000000 && pref_addr <= 0x8fffffff){
            new_addr_type pref_block_addr = m_config.block_addr(pref_addr);
            pref_mf->set_warp_id(m_tag_array->get_prefetch_warp_id());
            pref_mf->set_addr(pref_block_addr);
            // if(mf->get_sid() == 0)
            //     printf("actual_pref_addr:%x time:%d pref_wid:%d times:%d\n", pref_block_addr, time, pref_mf->get_wid(), actual_pref_times++);
            send_read_request_pref(pref_addr, pref_block_addr,
                        pref_cache_index, pref_mf, time, 
                        pref_do_miss, pref_wb, pref_evicted, events, false, false);
            if(pref_do_miss)
            {
                if (pref_wb && (m_config.m_write_policy != WRITE_THROUGH))
                {
                    mem_fetch *pref_wb = m_memfetch_creator->alloc(pref_evicted.m_block_addr,
                                                                m_wrbk_type, m_config.get_line_sz(), true);
                    send_write_request_pref(pref_wb, WRITE_BACK_REQUEST_SENT, time, events); //wb == 1 means cache line is modified, it needs to write back to memory
                }
            }
        }
        else{
            //printf("delete pref mf:%d\n",delete_pref++);
            delete pref_mf;
            pref_mf = NULL;
        }
    }
    // else if( USE_PREFETCH && pref_mf != NULL && is_l1_cache ){
    //     //printf("delete pref mf:%d\n",delete_pref++);
    //     delete pref_mf;
    //     pref_mf = NULL;
    // }
    m_stats.inc_stats(mf->get_access_type(),
                      m_stats.select_stats_status(probe_status, access_status));
    return access_status;
}

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the
/// granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
//int access_num = 0;
enum cache_request_status
l1_cache::access(new_addr_type addr,
                 mem_fetch *mf,
                 mem_fetch *pref_mf,
                 unsigned time,
                 std::list<cache_event> &events)
{
    //printf("access_num:%d addr:%X time:%d \n",access_num++,addr,time);/*cory*/
    return data_cache::access(addr, mf, pref_mf, time, events, 1);
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status
l2_cache::access(new_addr_type addr,
                 mem_fetch *mf,
                 mem_fetch *pref_mf,
                 unsigned time,
                 std::list<cache_event> &events)
{
    return data_cache::access(addr, mf, pref_mf, time, events, 0);
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access(new_addr_type addr, mem_fetch *mf,
                                            mem_fetch *pref_mf, unsigned time, std::list<cache_event> &events)
{
    if (m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full())
        return RESERVATION_FAIL;

    assert(mf->get_data_size() <= m_config.get_line_sz());

    // at this point, we will accept the request : access tags and immediately allocate line
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status status = m_tags.access(block_addr, time, cache_index);
    enum cache_request_status cache_status = RESERVATION_FAIL;
    assert(status != RESERVATION_FAIL);
    assert(status != HIT_RESERVED); // as far as tags are concerned: HIT or MISS
    m_fragment_fifo.push(fragment_entry(mf, cache_index, status == MISS, mf->get_data_size()));
    if (status == MISS)
    {
        // we need to send a memory request...
        unsigned rob_index = m_rob.push(rob_entry(cache_index, mf, block_addr));
        m_extra_mf_fields[mf] = extra_mf_fields(rob_index);
        mf->set_data_size(m_config.get_line_sz());
        m_tags.fill(cache_index, time); // mark block as valid
        m_request_fifo.push(mf);
        mf->set_status(m_request_queue_status, time);
        events.push_back(READ_REQUEST_SENT);
        cache_status = MISS;
    }
    else
    {
        // the value *will* *be* in the cache already
        cache_status = HIT_RESERVED;
    }
    m_stats.inc_stats(mf->get_access_type(), m_stats.select_stats_status(status, cache_status));
    return cache_status;
}

//int flag = 0;
void tex_cache::cycle()
{
    // send next request to lower level of memory
    // if(flag==0){
    //     printf("enter_tex_cache!\n");
    //     flag = 1;
    // }
    if (!m_request_fifo.empty())
    {
        mem_fetch *mf = m_request_fifo.peek();
        if (!m_memport->full(mf->get_ctrl_size(), false))
        {
            m_request_fifo.pop();
            m_memport->push(mf);
        }
    }
    // read ready lines from cache
    if (!m_fragment_fifo.empty() && !m_result_fifo.full())
    {
        const fragment_entry &e = m_fragment_fifo.peek();
        if (e.m_miss)
        {
            // check head of reorder buffer to see if data is back from memory
            unsigned rob_index = m_rob.next_pop_index();
            const rob_entry &r = m_rob.peek(rob_index);
            assert(r.m_request == e.m_request);
            assert(r.m_block_addr == m_config.block_addr(e.m_request->get_addr()));
            if (r.m_ready)
            {
                assert(r.m_index == e.m_cache_index);
                m_cache[r.m_index].m_valid = true;
                m_cache[r.m_index].m_block_addr = r.m_block_addr;
                m_result_fifo.push(e.m_request);
                m_rob.pop();
                m_fragment_fifo.pop();
            }
        }
        else
        {
            // hit:
            assert(m_cache[e.m_cache_index].m_valid);
            assert(m_cache[e.m_cache_index].m_block_addr == m_config.block_addr(e.m_request->get_addr()));
            m_result_fifo.push(e.m_request);
            m_fragment_fifo.pop();
        }
    }
}

/// Place returning cache block into reorder buffer
void tex_cache::fill(mem_fetch *mf, unsigned time)
{
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    assert(e != m_extra_mf_fields.end());
    assert(e->second.m_valid);
    assert(!m_rob.empty());
    mf->set_status(m_rob_status, time);

    unsigned rob_index = e->second.m_rob_index;
    rob_entry &r = m_rob.peek(rob_index);
    assert(!r.m_ready);
    r.m_ready = true;
    r.m_time = time;
    assert(r.m_block_addr == m_config.block_addr(mf->get_addr()));
}

void tex_cache::display_state(FILE *fp) const
{
    fprintf(fp, "%s (texture cache) state:\n", m_name.c_str());
    fprintf(fp, "fragment fifo entries  = %u / %u\n",
            m_fragment_fifo.size(), m_fragment_fifo.capacity());
    fprintf(fp, "reorder buffer entries = %u / %u\n",
            m_rob.size(), m_rob.capacity());
    fprintf(fp, "request fifo entries   = %u / %u\n",
            m_request_fifo.size(), m_request_fifo.capacity());
    if (!m_rob.empty())
        fprintf(fp, "reorder buffer contents:\n");
    for (int n = m_rob.size() - 1; n >= 0; n--)
    {
        unsigned index = (m_rob.next_pop_index() + n) % m_rob.capacity();
        const rob_entry &r = m_rob.peek(index);
        fprintf(fp, "tex rob[%3d] : %s ",
                index, (r.m_ready ? "ready  " : "pending"));
        if (r.m_ready)
            fprintf(fp, "@%6u", r.m_time);
        else
            fprintf(fp, "       ");
        fprintf(fp, "[idx=%4u]", r.m_index);
        r.m_request->print(fp, false);
    }
    if (!m_fragment_fifo.empty())
    {
        fprintf(fp, "fragment fifo (oldest) :");
        fragment_entry &f = m_fragment_fifo.peek();
        fprintf(fp, "%s:          ", f.m_miss ? "miss" : "hit ");
        f.m_request->print(fp, false);
    }            
}
/******************************************************************************************************************************************/
