//
// Created by geraltigas on 3/20/24.
//

#include "thread_pool.h"

thread_pool::thread_pool(int n_threads) {
    stop = false;
    for (size_t i = 0; i < n_threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                    if (this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
    }
}

thread_pool::~thread_pool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker: workers) worker.join();
}

std::unique_ptr<thread_pool> g_thread_pool;

int init_thread_pool(int n_threads) {
    g_thread_pool = std::make_unique<thread_pool>(n_threads);
    return 0;
}
