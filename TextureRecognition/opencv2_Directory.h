#pragma once
#include <vector>

namespace cv
{
	/**
	* Class used for obtaining file and folder names.
	* Source code from OpenCV 2 library, not implemented in OpenCV 3.
	*/
	class Directory {
	public:
		static std::vector<std::string> GetListFiles(const std::string& path, const std::string & exten = "*", bool addPath = true);
		static std::vector<std::string> GetListFilesR(const std::string& path, const std::string & exten = "*", bool addPath = true);
		static std::vector<std::string> GetListFolders(const std::string& path, const std::string & exten = "*", bool addPath = true);
	};
}