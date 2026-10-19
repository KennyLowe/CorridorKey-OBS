#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

TEST_CASE("a CUDA device is present", "[smoke]")
{
	int n = 0;
	REQUIRE(cudaGetDeviceCount(&n) == cudaSuccess);
	REQUIRE(n > 0);
}
