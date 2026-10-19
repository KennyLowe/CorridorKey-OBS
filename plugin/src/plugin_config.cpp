#include "plugin_config.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#include <nlohmann/json.hpp>

namespace ck::config {
namespace {
std::mutex mu;
std::string path_;
nlohmann::json doc = nlohmann::json::object();

void save()
{
	if (path_.empty())
		return;
	auto p = std::filesystem::path((const char8_t *)path_.c_str());
	std::error_code ec;
	std::filesystem::create_directories(p.parent_path(), ec);
	auto tmp = p;
	tmp += ".tmp";
	{
		std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
		f << doc.dump(2);
	}
	std::filesystem::rename(tmp, p, ec);
}
} // namespace

void init(const std::string &path)
{
	std::lock_guard lk(mu);
	path_ = path;
	std::ifstream f(std::filesystem::path((const char8_t *)path.c_str()), std::ios::binary);
	if (f) {
		try {
			doc = nlohmann::json::parse(f);
		} catch (...) {
			doc = nlohmann::json::object();
		}
	}
	if (!doc.is_object())
		doc = nlohmann::json::object();
}

std::string licence_acknowledged(const std::string &model_id)
{
	std::lock_guard lk(mu);
	auto &a = doc["licence_acknowledged"];
	return a.is_object() && a.contains(model_id) && a[model_id].is_string() ? a[model_id].get<std::string>() : "";
}

void acknowledge_licence(const std::string &model_id)
{
	std::lock_guard lk(mu);
	auto &a = doc["licence_acknowledged"];
	if (!a.is_object())
		a = nlohmann::json::object();
	if (a.contains(model_id))
		return;
	std::time_t t = std::time(nullptr);
	char buf[32];
	std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&t));
	a[model_id] = buf;
	save();
}

long long last_model_check()
{
	std::lock_guard lk(mu);
	return doc.value("last_model_check", 0LL);
}

void set_model_check(long long when, const std::vector<std::string> &newer)
{
	std::lock_guard lk(mu);
	doc["last_model_check"] = when;
	doc["newer_models"] = newer;
	save();
}

std::vector<std::string> newer_models()
{
	std::lock_guard lk(mu);
	auto it = doc.find("newer_models");
	return it != doc.end() && it->is_array() ? it->get<std::vector<std::string>>() : std::vector<std::string>{};
}

} // namespace ck::config
