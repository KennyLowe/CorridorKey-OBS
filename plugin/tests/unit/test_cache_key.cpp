#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "core/cache_key.hpp"

using namespace ck;
namespace fs = std::filesystem;

static KeyComponents fixture_comp()
{
	KeyComponents c;
	c.gpu_uuid = "GPU-0763dee5-2e3a-a1b2-955d-4600c81fd6c0";
	c.driver_version = "576.80";
	c.trt_version = "10.16.1.11";
	c.weights_sha256 = std::string(32 * 2, 'a');
	for (size_t i = 1; i < c.weights_sha256.size(); i += 2)
		c.weights_sha256[i] = 'b';
	c.internal_w = 1024;
	c.internal_h = 576;
	c.refiner = false;
	return c;
}

TEST_CASE("sha256 known vectors", "[cache_key]")
{
	CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
	CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
	CHECK(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
	      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
	std::string million(1000000, 'a');
	CHECK(sha256_hex(million) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("sha256 of a file streams the content", "[cache_key]")
{
	auto p = fs::temp_directory_path() / "ck_sha_test.bin";
	{
		std::ofstream f(p, std::ios::binary);
		std::string s(5 << 20, 'x');
		f << s;
	}
	auto h = sha256_file(p.string());
	REQUIRE(h);
	CHECK(*h == sha256_hex(std::string(5 << 20, 'x')));
	fs::remove(p);
	CHECK_FALSE(sha256_file((fs::temp_directory_path() / "does_not_exist.bin").string()));
}

TEST_CASE("canonical JSON and key match the Python helper", "[cache_key]")
{
	auto c = fixture_comp();
	CHECK(canonical_json(c) ==
	      R"({"driver_version":"576.80","gpu_uuid":"GPU-0763dee5-2e3a-a1b2-955d-4600c81fd6c0","internal_h":576,)"
	      R"("internal_w":1024,"refiner":false,"trt_version":"10.16.1.11","weights_sha256":")" +
		      c.weights_sha256 + "\"}");
	CHECK(cache_key(c) == "12a5b6076e3cf1ebd65acf2baf4679580dec8f63482848cbe874c31c61272a98");
}

TEST_CASE("reads the shared manifest fixture written by Python", "[cache_key]")
{
	auto m = read_manifest(std::string(CK_TEST_DATA_DIR) + "/manifest_fixture.json");
	REQUIRE(m);
	REQUIRE(m->entries.size() == 1);
	const auto &e = m->entries[0];
	CHECK(e.comp == fixture_comp());
	CHECK(e.key == cache_key(fixture_comp()));
	CHECK(e.gpu_name == "NVIDIA GeForce RTX 3060");
	CHECK(e.build_s == 312.4);
	CHECK(e.engine_path.find("ck_1024x576_ref0.engine") != std::string::npos);
}

TEST_CASE("missing manifest is empty, corrupt manifest is nullopt", "[cache_key]")
{
	auto dir = fs::temp_directory_path() / "ck_manifest_test";
	fs::create_directories(dir);
	auto p = (dir / "manifest.json").string();
	fs::remove(p);
	auto m = read_manifest(p);
	REQUIRE(m);
	CHECK(m->entries.empty());
	{
		std::ofstream f(p);
		f << "{\"schema_version\": \"1.0\", \"entries\": [ {";
	}
	CHECK_FALSE(read_manifest(p));
	fs::remove_all(dir);
}

TEST_CASE("atomic write round-trips and leaves no temp file", "[cache_key]")
{
	auto dir = fs::temp_directory_path() / "ck_manifest_rt";
	fs::remove_all(dir);
	fs::create_directories(dir);
	auto p = (dir / "manifest.json").string();
	Manifest m;
	CacheEntry e;
	e.comp = fixture_comp();
	e.key = cache_key(e.comp);
	e.engine_path = (dir / "x.engine").string();
	e.gpu_name = "NVIDIA RTX A2000";
	e.build_s = 10.5;
	m.entries.push_back(e);
	REQUIRE(write_manifest(p, m));
	CHECK_FALSE(fs::exists(p + ".tmp"));
	// overwrite an existing file too
	REQUIRE(write_manifest(p, m));
	auto r = read_manifest(p);
	REQUIRE(r);
	REQUIRE(r->entries.size() == 1);
	CHECK(r->entries[0].comp == e.comp);
	CHECK(r->entries[0].gpu_name == "NVIDIA RTX A2000");
	fs::remove_all(dir);
}

TEST_CASE("lookup: hit, miss and stale", "[cache_key]")
{
	Manifest m;
	CacheEntry e;
	e.comp = fixture_comp();
	e.key = cache_key(e.comp);
	e.engine_path = "unused";
	m.entries.push_back(e);

	auto want = fixture_comp();
	auto r = lookup(m, want, false);
	CHECK(r.kind == Lookup::Hit);
	REQUIRE(r.entry);
	CHECK(r.entry->key == e.key);

	auto other_q = want;
	other_q.internal_w = 512;
	other_q.internal_h = 512;
	CHECK(lookup(m, other_q, false).kind == Lookup::Miss);

	auto new_driver = want;
	new_driver.driver_version = "581.15";
	r = lookup(m, new_driver, false);
	CHECK(r.kind == Lookup::Stale);
	CHECK(r.why.find("driver") != std::string::npos);

	auto new_gpu = want;
	new_gpu.gpu_uuid = "GPU-ffffffff-0000-0000-0000-000000000000";
	CHECK(lookup(m, new_gpu, false).kind == Lookup::Stale);

	auto new_trt = want;
	new_trt.trt_version = "10.16.1.12";
	CHECK(lookup(m, new_trt, false).kind == Lookup::Stale);

	auto new_weights = want;
	new_weights.weights_sha256 = std::string(64, 'c');
	CHECK(lookup(m, new_weights, false).kind == Lookup::Stale);

	// a hit whose engine file is gone is a miss
	CHECK(lookup(m, want, true).kind == Lookup::Miss);

	// an entry whose stored key doesn't match its components is never trusted
	Manifest bad = m;
	bad.entries[0].key = std::string(64, '0');
	CHECK(lookup(bad, want, false).kind != Lookup::Hit);
}
