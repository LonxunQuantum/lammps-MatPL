#pragma once
// #include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#define CHECK_CUDA_NEP(call)                                                                                \
  do {                                                                                             \
    const cudaError_t error_code = call;                                                           \
    if (error_code != cudaSuccess) {                                                               \
      fprintf(stderr, "CUDA Error:\n");                                                            \
      fprintf(stderr, "    File:       %s\n", __FILE__);                                           \
      fprintf(stderr, "    Line:       %d\n", __LINE__);                                           \
      fprintf(stderr, "    Error code: %d\n", error_code);                                         \
      fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(error_code));                     \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define PRINT_SCANF_ERROR(count, n, text)                                                          \
  do {                                                                                             \
    if (count != n) {                                                                              \
      fprintf(stderr, "Input Error:\n");                                                           \
      fprintf(stderr, "    File:       %s\n", __FILE__);                                           \
      fprintf(stderr, "    Line:       %d\n", __LINE__);                                           \
      fprintf(stderr, "    Error text: %s\n", text);                                               \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define PRINT_INPUT_ERROR(text)                                                                    \
  do {                                                                                             \
    fprintf(stderr, "Input Error:\n");                                                             \
    fprintf(stderr, "    File:       %s\n", __FILE__);                                             \
    fprintf(stderr, "    Line:       %d\n", __LINE__);                                             \
    fprintf(stderr, "    Error text: %s\n", text);                                                 \
    exit(1);                                                                                       \
  } while (0)

#define PRINT_KEYWORD_ERROR(keyword)                                                               \
  do {                                                                                             \
    fprintf(stderr, "Input Error:\n");                                                             \
    fprintf(stderr, "    File:       %s\n", __FILE__);                                             \
    fprintf(stderr, "    Line:       %d\n", __LINE__);                                             \
    fprintf(stderr, "    Error text: '%s' is an invalid keyword.\n", keyword);                     \
    exit(1);                                                                                       \
  } while (0)

#ifdef STRONG_DEBUG
#define CUDA_CHECK_KERNEL                                                                          \
  {                                                                                                \
    CHECK_CUDA_NEP(cudaGetLastError());                                                                     \
    CHECK_CUDA_NEP(cudaDeviceSynchronize());                                                                \
  }
#else
#define CUDA_CHECK_KERNEL                                                                          \
  {                                                                                                \
    CHECK_CUDA_NEP(cudaGetLastError());                                                                     \
  }
#endif
FILE* my_fopen(const char* filename, const char* mode);
std::vector<std::string> get_tokens(const std::string& line);
std::vector<std::string> get_tokens(std::ifstream& input);
std::vector<std::string> get_tokens_without_unwanted_spaces(std::ifstream& input);
int get_int_from_token(const std::string& token, const char* filename, const int line);
float get_float_from_token(const std::string& token, const char* filename, const int line);
double get_double_from_token(const std::string& token, const char* filename, const int line);
