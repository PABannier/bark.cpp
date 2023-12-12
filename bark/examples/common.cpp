#include <string> 
#include <vector>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "common.h"

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

void bark_print_usage(char ** argv, const bark_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -s N, --seed N        seed for random number generator (default: %d)\n", params.seed);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model_path.c_str());
    fprintf(stderr, "  -o FNAME, --outwav FNAME\n");
    fprintf(stderr, "                        output generated wav (default: %s)\n", params.dest_wav_path.c_str());
    fprintf(stderr, "\n");
}

int bark_params_parse(int argc, char ** argv, bark_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            params.prompt = argv[++i];
        } else if (arg == "-m" || arg == "--model") {
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
