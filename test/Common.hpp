#include <type_traits>

#include <Types.h>

template <uint32_t N>
using I = std::integral_constant<uint32_t, N>;

template <class... Ts, class F>
void for_each(std::tuple<Ts...>, F f) {
    std::initializer_list<int> _ = { (f(Ts{}), 0)... }; // poor man's fold expression for C++11/14
    // (f(Ts{}), ...); // fold expression is for C++17 only
}
