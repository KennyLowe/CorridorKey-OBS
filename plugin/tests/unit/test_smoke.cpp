#include <catch2/catch_test_macros.hpp>

TEST_CASE("catch2 runs", "[smoke]")
{
	REQUIRE(1 + 1 == 2);
}
