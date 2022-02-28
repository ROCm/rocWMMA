###############################################################################
 #
 # MIT License
 #
 # Copyright 2021 Advanced Micro Devices, Inc.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 ###############################################################################


 # Documentation requires doxygen installation
find_package( Doxygen REQUIRED )

set(ROCWMMA_DOCS_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/../rocWMMA-Guide.tar.gz")
set(ROCWMMA_DOXYGEN_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/../library" "${CMAKE_CURRENT_SOURCE_DIR}/../test")
set(ROCWMMA_DOXYGEN_OUTPUT "${CMAKE_BINARY_DIR}/docs")

add_custom_target(
  build_docs ALL
  DEPENDS ${ROCWMMA_DOCS_ARCHIVE}
)

# Convert input list to space separated string
string (REPLACE ";" " " ROCWMMA_DOXYGEN_INPUT "${ROCWMMA_DOXYGEN_INPUT}")

# Adjust the doxygen inputs and output directories dynamically
add_custom_command(
  OUTPUT ${ROCWMMA_DOCS_ARCHIVE}
  COMMAND bash -c "(cat ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile; echo \"INPUT=${ROCWMMA_DOXYGEN_INPUT}\"; echo \"OUTPUT_DIRECTORY=${ROCWMMA_DOXYGEN_OUTPUT}\") | doxygen -"
  COMMAND tar -czvf ${ROCWMMA_DOCS_ARCHIVE} ${ROCWMMA_DOXYGEN_OUTPUT}
  VERBATIM)
