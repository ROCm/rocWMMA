/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_LOGGING_HPP
#define ROCWMMA_LOGGING_HPP

#include "singleton.hpp"
#include "rocwmma_ostream.hpp"
#include <stdlib.h>

namespace rocwmma
{
    struct RocwmmaLogging : public LazySingleton<RocwmmaLogging>
    {
        // For static initialization
        friend std::unique_ptr<RocwmmaLogging> std::make_unique<RocwmmaLogging>();

        private: // No public instantiation except make_unique.
                 // No copy
            RocwmmaLogging(RocwmmaLogging const&) = delete;
            RocwmmaLogging& operator=(RocwmmaLogging const&) = delete;

        public:
            RocwmmaLogging(RocwmmaLogging&&);
            ~RocwmmaLogging() = default;

            RocwmmaLogging()
                : mOstream()
                , mOmitSkipped(false)
                , mOmitFailed(false)
                , mOmitPassed(false)
                , mOmitCout(false)
            {
            }

            void setOmits(int mask)
            {
                if (mask & 1)
                    mOmitSkipped = true;
                if (mask & 2)
                    mOmitFailed = true;
                if (mask & 4)
                    mOmitPassed = true;
                if (mask & 8)
                    mOmitCout = true;
            }

            void parseOptions(int argc, char** argv)
            {
                const std::vector<std::string> args(argv + 1, argv + argc);
                std::string fileName;

                for (auto i = 0; i < argc - 1; i++)
                {
                    if (args[i] == "-os" || args[i] == "--output_stream")
                    {
                        if (i + 2 >= argc)
                        {
                            std::cerr << "Missing output stream\n";
                            std::cerr << "Usage: -os || --output_stream *file.csv*\n";
                            exit(EXIT_FAILURE);
                        }
                        fileName = args[i + 1];
                        i++;
                    }
                    if (args[i] == "--omit")
                    {
                        if (i + 2 >= argc)
                        {
                            std::cerr << "Missing omit integer mask\n";
                            std::cerr << "Usage: --omit *integer_mask*\n";
                            exit(EXIT_FAILURE);
                        }
                        setOmits(std::stoi(args[i + 1]));
                    }
                }

                mOstream.initializeStream(fileName);
            }
            

            rocwmmaOStream& ostream()
            {
                return mOstream;
            }

            bool omitSkipped()
            {
                return mOmitSkipped;
            }

            bool omitFailed()
            {
                return mOmitFailed;
            }

            bool omitPassed()
            {
                return mOmitPassed;
            }

            bool omitCout()
            {
                return mOmitCout;
            }

        protected:
            rocwmmaOStream mOstream;

            bool mOmitSkipped, mOmitFailed, mOmitPassed, mOmitCout;
    };
}


#endif // ROCWMMA_LOGGING_HPP