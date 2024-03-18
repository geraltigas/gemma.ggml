//
// Created by geraltigas on 3/17/24.
//

#include "profiling.h"
// unique_ptr
#include <memory>
#include <string>
#include <chrono>
#include <map>
#include <iostream>
#include <vector>
#include <algorithm>

struct profiler_context {
    // map: string : time(in chrono, ms)
public:
    std::map<std::string, std::chrono::microseconds> records;
    std::map<std::string, std::vector<std::string>> overlapping;
    std::map<std::string, uint32_t> _call_count;
    std::map<std::string, uint32_t> call_count;

    void start_recording(const std::string &name) {
        if (start_time.find(name) != start_time.end()) {
            std::cerr << "profiling: " << name << " already exists" << std::endl;
            exit(1);
        }
        start_time[name] = std::chrono::high_resolution_clock::now();
        _call_count[name] = _call_count[name] + 1;
        if (start_time.size() > 1) {
            for (auto &record: start_time) {
                if (record.first != name) {
                    if (std::find(overlapping[name].begin(), overlapping[name].end(), record.first) ==
                        overlapping[name].end()) {
                        overlapping[name].push_back(record.first);
                    }
                }
            }
        }
    }

    void stop_recording(const std::string &name) {
        if (start_time.find(name) == start_time.end()) {
            std::cerr << "profiling: " << name << " not found" << std::endl;
            exit(1);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        // number of milliseconds
        auto _start_time = start_time[name];
        records[name] = records[name] + std::chrono::duration_cast<std::chrono::microseconds>(end_time - _start_time);
        start_time.erase(name);
    }

    void add_count(const std::string &name) {
        if (call_count.find(name) == call_count.end()) {
            call_count[name] = 0;
        }
        call_count[name] = call_count[name] + 1;
    }

private:
    std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> start_time;
};

std::unique_ptr<profiler_context> profiler;

void check_profiler() {
    if (profiler == nullptr) {
        std::cerr << "profiler not initialized" << std::endl;
        exit(1);
    }
}

int init_profiling() {
    if (profiler != nullptr) {
        return 0;
    }
    profiler = std::make_unique<profiler_context>();
    return 0;
}

void add_count(const char *name) {
    check_profiler();
    profiler->add_count(name);
}

void start_recording(const char *name) {
    check_profiler();
    profiler->start_recording(name);
}

void stop_recording(const char *name) {
    check_profiler();
    profiler->stop_recording(name);
}

void print_prefix(const char *name) {
    std::cout << "----------------- " << name << " -----------------" << std::endl;
    uint64_t total = 0;
    for (auto &record: profiler->records) {
        if (record.first.find(name) == 0) {
            total += record.second.count();
        }
    }

    for (auto &record: profiler->records) {
        if (record.first.find(name) == 0) {
            std::cout << record.first << ": " << record.second.count() << " μs";
            std::cout << " (" << (double) record.second.count() / total * 100 << "%), called "
                      << profiler->_call_count[record.first] << " times, overlapping with: ";
            if (profiler->overlapping[record.first].empty()) {
                std::cout << "none";
            } else {
                for (auto &overlap: profiler->overlapping[record.first]) {
                    std::cout << overlap << " ";
                }
            }
            std::cout << std::endl;
        }
    }

}

void print_profiling_result() {
    check_profiler();
    std::cout << std::endl;
    print_prefix("op");
    print_prefix("mul_mat");
}