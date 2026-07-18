// Phase 0 gate binary: load the TRT engine, run batch-1 inference over a raw fp32 input file,
// write the logits out. Parity vs the Python TRT path is checked by bench/parity_check.py.
//
//   reefscan_infer <engine.plan> <input.bin> <output.bin> <N>
//     input.bin  : N * 3*224*224 fp32, row-major [N,3,224,224]
//     output.bin : N * 2         fp32, row-major [N,2]
//
// Batch-1 loop on purpose — Phase 0 has no batching; that's Phase 1 (batch_queue).
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "trt_engine.h"

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: " << argv[0] << " <engine.plan> <input.bin> <output.bin> <N>\n";
    return 2;
  }
  const std::string engine_path = argv[1];
  const std::string in_path = argv[2];
  const std::string out_path = argv[3];
  const int N = std::stoi(argv[4]);
  const size_t per_image = static_cast<size_t>(3) * 224 * 224;

  std::ifstream fin(in_path, std::ios::binary);
  if (!fin) { std::cerr << "cannot open input: " << in_path << "\n"; return 1; }
  std::vector<float> input(static_cast<size_t>(N) * per_image);
  if (!fin.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(input.size() * sizeof(float)))) {
    std::cerr << "input.bin too small for N=" << N << "\n";
    return 1;
  }

  try {
    reefscan::TrtEngine engine(engine_path);
    std::ofstream fout(out_path, std::ios::binary);
    if (!fout) { std::cerr << "cannot open output: " << out_path << "\n"; return 1; }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
      auto logits = engine.infer(input.data() + static_cast<size_t>(i) * per_image, 1);
      fout.write(reinterpret_cast<const char*>(logits[0].data()), 2 * sizeof(float));
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cerr << "[phase0] " << N << " images, batch-1, " << (ms / N) << " ms/img (incl. H2D/D2H+sync); "
              << "wrote " << out_path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[phase0] FATAL: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
