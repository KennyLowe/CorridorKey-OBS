// Parser for the helper's JSON-lines protocol, with lines taken from contracts/prepare-cli.md.
#include <catch2/catch_test_macros.hpp>

#include "core/helper_proto.hpp"

using namespace ck;
using K = HelperEvent::Kind;

TEST_CASE("prepare progress and results", "[helper]")
{
	auto p = parse_helper_line(R"({"event":"progress","stage":"engine","pct":45,"msg":"building TensorRT engine"})");
	REQUIRE(p);
	CHECK(p->kind == K::Progress);
	CHECK(p->stage == "engine");
	CHECK(p->pct == 45);
	CHECK(p->msg == "building TensorRT engine");

	auto ok = parse_helper_line(
		R"({"event":"result","ok":true,"key":"abc","engine_path":"C:/c/e.engine","trt_version":"10.16.1.11","build_s":312.4})");
	REQUIRE(ok);
	CHECK(ok->kind == K::Result);
	CHECK(ok->ok);
	CHECK(ok->engine_path == "C:/c/e.engine");
	CHECK(ok->trt_version == "10.16.1.11");
	CHECK(ok->build_s == 312.4);

	auto bad = parse_helper_line(R"({"event":"result","ok":false,"error":"weights_invalid","msg":"not a checkpoint"})");
	REQUIRE(bad);
	CHECK_FALSE(bad->ok);
	CHECK(bad->error == "weights_invalid");
	CHECK(bad->msg == "not a checkpoint");
}

TEST_CASE("fetch-weights progress and results", "[helper]")
{
	auto p = parse_helper_line(R"({"event":"progress","stage":"download","pct":42,"bytes":168000000,"total":398849256})");
	REQUIRE(p);
	CHECK(p->bytes == 168000000);
	CHECK(p->total == 398849256);
	auto r = parse_helper_line(
		R"({"event":"result","ok":true,"weights_path":"C:/w/CorridorKey_v1.0.safetensors","sha256":"74d6","model_id":"CorridorKey_v1.0"})");
	REQUIRE(r);
	CHECK(r->ok);
	CHECK(r->weights_path == "C:/w/CorridorKey_v1.0.safetensors");
	CHECK(r->model_id == "CorridorKey_v1.0");
	auto f = parse_helper_line(R"({"event":"result","ok":false,"error":"hash_mismatch","msg":"sha256 x != y"})");
	REQUIRE(f);
	CHECK(f->error == "hash_mismatch");
}

TEST_CASE("check-models result", "[helper]")
{
	auto r = parse_helper_line(
		R"({"event":"result","ok":true,"current":"CorridorKey_v1.0","newer":[{"model_id":"CorridorKey_v1.1","hf_repo":"n/CorridorKey_v1.1","last_modified":"2026-10-01"}]})");
	REQUIRE(r);
	CHECK(r->current == "CorridorKey_v1.0");
	REQUIRE(r->newer.size() == 1);
	CHECK(r->newer[0] == "CorridorKey_v1.1");
	auto none = parse_helper_line(R"({"event":"result","ok":true,"current":"CorridorKey_v1.0","newer":[]})");
	REQUIRE(none);
	CHECK(none->newer.empty());
}

TEST_CASE("junk lines are ignored", "[helper]")
{
	CHECK_FALSE(parse_helper_line(""));
	CHECK_FALSE(parse_helper_line("   \r"));
	CHECK_FALSE(parse_helper_line("Resolved 65 packages in 1ms"));
	CHECK_FALSE(parse_helper_line(R"({"event":"other"})"));
	CHECK_FALSE(parse_helper_line(R"([1,2,3])"));
	CHECK_FALSE(parse_helper_line(R"({"event":"progress","pct":"x"})"));
	auto crlf = parse_helper_line("{\"event\":\"progress\",\"stage\":\"verify\",\"pct\":0,\"msg\":\"\"}\r");
	REQUIRE(crlf);
	CHECK(crlf->stage == "verify");
}
