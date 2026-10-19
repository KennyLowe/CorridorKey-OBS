// Runs one ckbench helper command (prepare / fetch-weights / check-models) as a child process with
// no console window, reads its JSON-lines stdout on a reader thread, and hands parsed events to
// the caller (the worker thread) through poll(). The process tree is in a Job object that kills
// it when the client is destroyed (filter removed, OBS closing) or kill() is called. stderr goes
// to a log file under %LOCALAPPDATA%\corridorkey-obs\logs\ for diagnosis.
#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "core/helper_proto.hpp"

namespace ck {

class HelperProcess {
public:
	~HelperProcess(); // kills the process tree
	// `command`: the helper_cmd prefix (e.g. `uv run --project "C:\repo\bench" ckbench`);
	// `args`: the subcommand and its arguments (quoted here). nullptr + err on failure to start.
	static std::unique_ptr<HelperProcess> start(const std::string &command, const std::vector<std::string> &args,
						     const std::string &log_name, std::string &err);
	std::vector<HelperEvent> poll();
	bool finished() const { return finished_; } // stdout closed and process exited
	int exit_code() const { return exit_code_; }
	// The final result line, once seen.
	std::optional<HelperEvent> result() const;
	void kill();
	std::string command_line() const { return cmdline_; }
	std::string log_path() const { return log_path_; }

private:
	HelperProcess() = default;
	void read_loop();
	void *proc_ = nullptr, *job_ = nullptr, *out_rd_ = nullptr;
	std::thread reader_;
	mutable std::mutex mu_;
	std::deque<HelperEvent> q_;
	std::optional<HelperEvent> result_;
	std::atomic<bool> finished_{false};
	std::atomic<int> exit_code_{-1};
	std::string cmdline_, log_path_;
};

std::string quote_arg(const std::string &a);

} // namespace ck
