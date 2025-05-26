#include "compile_commands.hpp"

#include <fstream>
#include <stdexcept>

void tt::tt_metal::CompileCommandsDatabase::add(
    const std::string& directory, const std::string& command, const std::string& file) {
    database.push_back({directory, command, file});
}

void tt::tt_metal::CompileCommandsDatabase::dump(const std::string& directory) const {
    std::string path = directory + "/compile_commands.json";

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open compile_commands.json for writing");
    }

    ofs << "[\n";
    for (size_t i = 0; i < database.size(); ++i) {
        const auto& entry = database[i];
        ofs << "  {\n"
            << R"(    "directory": ")" << entry.directory << "\",\n"
            << R"(    "command": ")" << entry.command << "\",\n"
            << R"(    "file": ")" << entry.file << "\"\n"
            << "  }" << (i + 1 < database.size() ? "," : "") << "\n";
    }
    ofs << "]\n";
}
