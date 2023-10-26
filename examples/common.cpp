#include <string> 
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#define SAMPLE_RATE 24000

void write_wav_on_disk(std::vector<float> & audio_arr, std::string dest_path) {
    drwav_data_format format;
    format.bitsPerSample = 32;
    format.sampleRate = SAMPLE_RATE;
    format.container = drwav_container_riff;
    format.channels = 1;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;

    drwav wav;
    drwav_init_file_write(&wav, dest_path.c_str(), &format, NULL);
    drwav_uint64 frames = drwav_write_pcm_frames(&wav, audio_arr.size(), audio_arr.data());
    drwav_uninit(&wav);

    fprintf(stderr, "%s: Number of frames written = %lld.\n", __func__, frames);
}
