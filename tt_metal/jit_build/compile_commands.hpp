#include <string>
#include <vector>

using namespace tt;
namespace tt::tt_metal {
    class CompileCommandsDatabase {
    private:
        struct Entry  {
            std::string directory;
            std::string command;
            std::string file;
        };

        std::vector<Entry> database;
    public:

        void add(const std::string& directory, const std::string& command, const std::string& file) {
            database.push_back({directory, command, file});
        }

        void dump(const std::string& directory) const {
            // sstanisicTT fixup
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
        

        // Generates compile commands for the given device id and num_hw_cqs.
        void generate_compile_commands(chip_id_t device_id, uint8_t num_hw_cqs);

        // Returns the generated compile commands.
        const std::vector<std::string>& get_compile_commands() const;
    }

} // namespace tt::tt_metal