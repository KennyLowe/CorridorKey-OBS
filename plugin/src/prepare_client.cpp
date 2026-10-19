#include "prepare_client.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>

#include "ck-log.hpp"

namespace ck {

static std::wstring widen(const std::string &s)
{
	if (s.empty())
		return {};
	int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
	std::wstring w(n, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
	return w;
}

std::string quote_arg(const std::string &a)
{
	if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos)
		return a;
	std::string q = "\"";
	size_t bs = 0;
	for (char c : a) {
		if (c == '\\') {
			bs++;
			continue;
		}
		if (c == '"') {
			q.append(bs * 2 + 1, '\\');
			q += '"';
		} else {
			q.append(bs, '\\');
			q += c;
		}
		bs = 0;
	}
	q.append(bs * 2, '\\');
	return q + "\"";
}

std::unique_ptr<HelperProcess> HelperProcess::start(const std::string &command, const std::vector<std::string> &args,
						    const std::string &log_name, std::string &err)
{
	std::unique_ptr<HelperProcess> h(new HelperProcess());
	h->cmdline_ = command;
	for (const auto &a : args)
		h->cmdline_ += " " + quote_arg(a);

	SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
	HANDLE out_rd = nullptr, out_wr = nullptr;
	if (!CreatePipe(&out_rd, &out_wr, &sa, 0)) {
		err = "CreatePipe failed";
		return nullptr;
	}
	SetHandleInformation(out_rd, HANDLE_FLAG_INHERIT, 0);
	// stderr -> log file
	const char *la = std::getenv("LOCALAPPDATA");
	std::filesystem::path logdir = std::filesystem::path(widen(la ? la : ".")) / L"corridorkey-obs" / L"logs";
	std::error_code ec;
	std::filesystem::create_directories(logdir, ec);
	auto logp = logdir / widen(log_name + ".log");
	h->log_path_ = logp.string();
	HANDLE err_h = CreateFileW(logp.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS,
				   FILE_ATTRIBUTE_NORMAL, nullptr);
	HANDLE nul_in = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0,
				    nullptr);

	STARTUPINFOW si{};
	si.cb = sizeof(si);
	si.dwFlags = STARTF_USESTDHANDLES;
	si.hStdInput = nul_in;
	si.hStdOutput = out_wr;
	si.hStdError = err_h != INVALID_HANDLE_VALUE ? err_h : out_wr;
	PROCESS_INFORMATION pi{};
	std::wstring cl = widen(h->cmdline_);
	HANDLE job = CreateJobObjectW(nullptr, nullptr);
	JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
	li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
	if (job)
		SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
	BOOL ok = CreateProcessW(nullptr, cl.data(), nullptr, nullptr, TRUE,
				 CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT, nullptr, nullptr, &si, &pi);
	DWORD e = GetLastError();
	CloseHandle(out_wr);
	if (err_h != INVALID_HANDLE_VALUE)
		CloseHandle(err_h);
	if (nul_in != INVALID_HANDLE_VALUE)
		CloseHandle(nul_in);
	if (!ok) {
		CloseHandle(out_rd);
		if (job)
			CloseHandle(job);
		err = "cannot start the helper (" + h->cmdline_ + "): Windows error " + std::to_string(e) +
		      ". Check the helper command in Advanced settings (uv must be on PATH).";
		return nullptr;
	}
	if (job)
		AssignProcessToJobObject(job, pi.hProcess);
	ResumeThread(pi.hThread);
	CloseHandle(pi.hThread);
	h->proc_ = pi.hProcess;
	h->job_ = job;
	h->out_rd_ = out_rd;
	h->reader_ = std::thread([p = h.get()] { p->read_loop(); });
	return h;
}

void HelperProcess::read_loop()
{
	std::string buf;
	char chunk[4096];
	DWORD n = 0;
	auto flush_line = [&](std::string line) {
		if (auto ev = parse_helper_line(line)) {
			std::lock_guard lk(mu_);
			if (ev->kind == HelperEvent::Kind::Result)
				result_ = *ev;
			q_.push_back(std::move(*ev));
		} else if (line.find_first_not_of(" \r\n\t") != std::string::npos) {
			log(LOG_DEBUG, "helper: %s", line.c_str());
		}
	};
	while (ReadFile((HANDLE)out_rd_, chunk, sizeof(chunk), &n, nullptr) && n > 0) {
		buf.append(chunk, n);
		size_t pos;
		while ((pos = buf.find('\n')) != std::string::npos) {
			flush_line(buf.substr(0, pos));
			buf.erase(0, pos + 1);
		}
	}
	if (!buf.empty())
		flush_line(buf);
	WaitForSingleObject((HANDLE)proc_, 10000);
	DWORD code = 0;
	GetExitCodeProcess((HANDLE)proc_, &code);
	exit_code_ = (int)code;
	finished_ = true;
}

std::vector<HelperEvent> HelperProcess::poll()
{
	std::lock_guard lk(mu_);
	std::vector<HelperEvent> v(q_.begin(), q_.end());
	q_.clear();
	return v;
}

std::optional<HelperEvent> HelperProcess::result() const
{
	std::lock_guard lk(mu_);
	return result_;
}

void HelperProcess::kill()
{
	if (job_)
		TerminateJobObject((HANDLE)job_, 1);
	else if (proc_)
		TerminateProcess((HANDLE)proc_, 1);
}

HelperProcess::~HelperProcess()
{
	if (!finished_)
		kill();
	if (reader_.joinable())
		reader_.join();
	if (out_rd_)
		CloseHandle((HANDLE)out_rd_);
	if (proc_)
		CloseHandle((HANDLE)proc_);
	if (job_)
		CloseHandle((HANDLE)job_); // KILL_ON_JOB_CLOSE: nothing outlives us
}

} // namespace ck
