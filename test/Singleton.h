#ifndef WMMA_TEST_SINGLETON_H
#define WMMA_TEST_SINGLETON_H

#include <memory>

template <typename T>
class LazySingleton
{
public:
    static std::unique_ptr<T> const& instance()
    {
        static std::unique_ptr<T> sInstance = std::make_unique<T>();
        return sInstance;
    }
};

#endif // WMMA_TEST_SINGLETON_H
