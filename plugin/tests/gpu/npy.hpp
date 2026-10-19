// Minimal .npy reader for test fixtures written by bench/ckbench/parity.py (little-endian
// float32 '<f4' or uint8 '|u1', C order).
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

struct Array {
	std::string descr;
	std::vector<size_t> shape;
	std::vector<uint8_t> bytes;
	size_t count() const
	{
		size_t n = 1;
		for (auto s : shape)
			n *= s;
		return n;
	}
	const float *f32() const
	{
		if (descr != "<f4")
			throw std::runtime_error("not float32: " + descr);
		return (const float *)bytes.data();
	}
	const uint8_t *u8() const
	{
		if (descr != "|u1")
			throw std::runtime_error("not uint8: " + descr);
		return bytes.data();
	}
};

inline Array load(const std::string &path)
{
	std::ifstream f(path, std::ios::binary);
	if (!f)
		throw std::runtime_error("cannot open " + path +
					 " (run: uv run --project bench python -m ckbench.parity all)");
	char magic[6];
	f.read(magic, 6);
	if (memcmp(magic, "\x93NUMPY", 6) != 0)
		throw std::runtime_error("not an npy file: " + path);
	uint8_t ver[2];
	f.read((char *)ver, 2);
	uint32_t hlen = 0;
	if (ver[0] == 1) {
		uint16_t h16;
		f.read((char *)&h16, 2);
		hlen = h16;
	} else {
		f.read((char *)&hlen, 4);
	}
	std::string header(hlen, '\0');
	f.read(header.data(), hlen);
	Array a;
	auto d = header.find("'descr': '");
	a.descr = header.substr(d + 10, header.find('\'', d + 10) - (d + 10));
	if (header.find("'fortran_order': False") == std::string::npos)
		throw std::runtime_error("fortran order not supported");
	auto s = header.find("'shape': (");
	auto e = header.find(')', s);
	std::string dims = header.substr(s + 10, e - (s + 10));
	size_t pos = 0;
	while (pos < dims.size()) {
		auto c = dims.find(',', pos);
		std::string tok = dims.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
		if (tok.find_first_not_of(' ') != std::string::npos)
			a.shape.push_back(std::stoul(tok));
		if (c == std::string::npos)
			break;
		pos = c + 1;
	}
	size_t esz = a.descr == "<f4" ? 4 : 1;
	a.bytes.resize(a.count() * esz);
	f.read((char *)a.bytes.data(), (std::streamsize)a.bytes.size());
	if (!f)
		throw std::runtime_error("short read: " + path);
	return a;
}

} // namespace npy
