//
// Created by geraltigas on 3/20/24.
//

#ifndef GEMMA_GGML_THREAD_POOL_H
#define GEMMA_GGML_THREAD_POOL_H


#include <future>
#include <vector>
#include <queue>
#include <functional>


struct thread_pool {
    thread_pool(int n_threads);
    ~thread_pool();
    template<class F, class... Args>
    auto enqueue(F &&f, Args &&... args) -> std::future<typename std::invoke_result<F, Args...>::type>;
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

extern std::unique_ptr<thread_pool> g_thread_pool;
int init_thread_pool(int n_threads);

template<class F, class... Args>
auto thread_pool::enqueue(F &&f, Args &&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // Don't allow enqueueing after stopping the pool
        if (stop) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

#endif //GEMMA_GGML_THREAD_POOL_H
