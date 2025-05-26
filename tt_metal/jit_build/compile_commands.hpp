#include <string>
#include <vector>

namespace tt::tt_metal {
class CompileCommandsDatabase {
private:
    struct Entry {
        std::string directory;
        std::string command;
        std::string file;
    };

    std::vector<Entry> database;

public:
    void add(const std::string& directory, const std::string& command, const std::string& file);
    void dump(const std::string& directory) const;
};

}  // namespace tt::tt_metal
