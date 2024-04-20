#include <iostream>
#include <string>
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "common.h"
#include "dr_wav.h"

#define SAMPLE_RATE 24000

void write_wav_on_disk(std::vector<float>& audio_arr, std::string dest_path) {
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

void bark_print_usage(char** argv, const bark_params& params) {
    std::cout << "usage: " << argv[0] << " [options]\n"
              << "\n"
              << "options:\n"
              << "  -h, --help            show this help message and exit\n"
              << "  -t N, --threads N     number of threads to use during computation (default: " << params.n_threads << ")\n"
              << "  -s N, --seed N        seed for random number generator (default: " << params.seed << ")\n"
              << "  -p PROMPT, --prompt PROMPT\n"
              << "                        prompt to start generation with (default: random)\n"
              << "  -m FNAME, --model FNAME\n"
              << "                        model path (default: " << params.model_path << ")\n"
              << "  -o FNAME, --outwav FNAME\n"
              << "                        output generated wav (default: " << params.dest_wav_path << ")\n"
              << "\n";
}

int bark_params_parse(int argc, char** argv, bark_params& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            params.prompt = argv[++i];
        } else if (arg == "-m" || arg == "--model_path") {
            params.model_path = argv[++i];
        } else if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-o" || arg == "--outwav") {
            params.dest_wav_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            bark_print_usage(argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bark_print_usage(argv, params);
            exit(0);
        }
    }

    return 0;
}
