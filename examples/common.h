#include <string>
#include <vector>

/**
 * @brief Reads a WAV file from disk and stores the audio data in a vector of floats.
 *
 * @param in_path Path to the input WAV file.
 * @param audio_arr Vector to store the audio data.
 * @return true If the file was successfully read.
 * @return false If the file could not be read.
 */
bool read_wav_from_disk(std::string in_path, std::vector<float> & audio_arr);
