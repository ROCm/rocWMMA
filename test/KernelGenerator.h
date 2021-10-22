/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef WMMA_KERNEL_GENERATOR_H
#define WMMA_KERNEL_GENERATOR_H

#include <tuple>
#include <vector>

/// Definitions
// TestParams: outer set of nested kernel parameters.
// tuple< tuple<KernelParams...>, ...>
//
// KernelParams: tuple of Params to build a SINGLE kernel.
// tuple<KernelParams...>
//

/// Notes:
// These utilities to follow are used to build
// TestParams, which are generated at compile time
// and used to build kernels of interest for testing.
//
// First step is to build TestParams with the
// appropriate format in the definition from basic
// types.
//
// Second step is to generate kernels from TestParams
// and provide them as a vector so they may be used
// under test conditions.
//

// There are several classes that provide functionality
// to build combinations of basic types.
//
// Concat: Concatenation of types together.
// E.g.
// Concat( A, B ) = tuple<A, B>
// Concat( tuple<A>, B ) = tuple<A, B>
// Concat( A, tuple<B> ) = tuple<A, B>
// Concat( tuple<A>, tuple<B> ) = tuple<A, B>
template <typename Lhs, typename Rhs>
struct Concat
{
    using Result = decltype(std::make_tuple(Lhs(), Rhs()));
};

template <typename Lhs, typename... Rhs>
struct Concat<Lhs, std::tuple<Rhs...>>
{
    using Result = decltype(std::tuple_cat(std::make_tuple(Lhs()), std::tuple<Rhs...>()));
};

template <typename... Lhs, typename... Rhs>
struct Concat<std::tuple<Lhs...>, std::tuple<Rhs...>>
{
    using Result = decltype(std::tuple_cat(std::tuple<Lhs...>(), std::tuple<Rhs...>()));
};

template <typename... Lhs, typename Rhs>
struct Concat<std::tuple<Lhs...>, Rhs>
{
    using Result = decltype(std::tuple_cat(std::tuple<Lhs...>(), std::make_tuple(Rhs())));
};

// CombineOne: Creates combinatorial pairs of LHS
// with EACH type of RHS if RHS is a tuple.
// NOTE: First level of tuples are collapsed into
// nested tuples, as required by the generator.
//
// E.g.
// CombineOne( A, B ) = tuple< tuple<A, B> >
// CombineOne( A, tuple<B> ) = tuple< tuple<A, B> >
// CombineOne( A, tuple<B, C, D> ) = tuple< tuple<A, B>,
//                                          tuple<A, C>,
//                                          tuple<A, D>>
// CombineOne( tuple<A, B>, tuple<C, D> ) = tuple< tuple<A, B, C>,
//                                          tuple<A, B, D>>
//

template <typename Lhs, typename Rhs>
struct CombineOne
{
    using Result = decltype(std::make_tuple(typename Concat<Lhs, Rhs>::Result()));
};

template <typename Lhs, typename Rhs>
struct CombineOne<Lhs, std::tuple<Rhs>>
{
    using Result = decltype(std::make_tuple(typename Concat<Lhs, Rhs>::Result()));
};

template <typename Lhs, typename Rhs0, typename... Rhs>
struct CombineOne<Lhs, std::tuple<Rhs0, Rhs...>>
{
    using Mine   = typename Concat<Lhs, Rhs0>::Result;
    using Next   = CombineOne<Lhs, std::tuple<Rhs...>>;
    using Result = decltype(std::tuple_cat(std::make_tuple(Mine()), typename Next::Result()));
};

// CombineMany: Creates combinatorial pairs two lists:
// EACH type of LHS with EACH type of RHS.
// NOTE: First level of tuples are collapsed into
// nested tuples, as required by the generator.
//
// E.g.
// CombineMany( A, B ) = tuple< tuple<A, B> >
// CombineMany( A, tuple<B> ) = tuple< tuple<A, B> >
// CombineMany( A, tuple<B, C, D> ) = tuple< tuple<A, B>,
//                                           tuple<A, C>,
//                                           tuple<A, D>>
// CombineMany( tuple<A, B>, tuple<C, D> ) = tuple< tuple<A, C>,
//                                                 tuple<A, D>,
//                                                 tuple<B, C>,
//                                                 tuple<B, D>>
template <typename Lhs, typename Rhs>
struct CombineMany
{
    using Result = typename CombineOne<Lhs, Rhs>::Result;
};

template <typename Lhs, typename Rhs>
struct CombineMany<std::tuple<Lhs>, Rhs>
{
    using Result = typename CombineOne<Lhs, Rhs>::Result;
};

template <typename Lhs0, typename... Lhs, typename Rhs>
struct CombineMany<std::tuple<Lhs0, Lhs...>, Rhs>
{
    using Mine   = typename CombineOne<Lhs0, Rhs>::Result;
    using Next   = CombineMany<std::tuple<Lhs...>, Rhs>;
    using Result = decltype(std::tuple_cat(Mine(), typename Next::Result()));
};

// CombineLists: Creates combinatorial sets from multiple lists.
//
// E.g:
// CombineLists( tuple< tuple<A, B> >, tuple< tuple<C, D> > ) =
//     tuple< tuple<A, B, C, D> >
// CombineLists( tuple< tuple<A, B>, tuple<C, D> >, tuple< tuple<E, F>, tuple<G, H> > ) =
//     tuple< tuple<A, B, E, F>, tuple<A, B, G, H>, tuple<C, D, E, F>, tuple<C, D, G, H> >

template <typename List0, typename... Lists>
struct CombineLists
{
    using Result = typename CombineMany<List0, typename CombineLists<Lists...>::Result>::Result;
};

template <typename List>
struct CombineLists<List>
{
    using Result = List;
};

/// Kernel Generator
// Requires two inputs:
// TestParams: nested tuple of KernelParams
// GeneratorImpl: a generator class that
// maps KernelParams to instantiation of the
// actual kernel.
//
// NOTE: The GeneratorImpl class decides the final
// generated kernel instantiated type. This class
// simply returns a vector of generated kernels of
// this type.
template <typename TestParams, class GeneratorImpl>
struct KernelGenerator
{
    template <typename... Ts>
    static void generate(Ts...);
};

template <typename KernelParams, typename... Next, class GeneratorImpl>
struct KernelGenerator<std::tuple<KernelParams, Next...>, GeneratorImpl>
{
    using ResultT = std::vector<typename GeneratorImpl::ResultT>;
    static ResultT generate()
    {
        auto result = ResultT();
        generate(result);
        return result;
    }

    static void generate(ResultT& kernels)
    {
        // Generate kernels
        kernels.push_back(GeneratorImpl::generate(KernelParams()));
        KernelGenerator<std::tuple<Next...>, GeneratorImpl>::generate(kernels);
    }
};

#endif // WMMA_KERNEL_GENERATOR_H
