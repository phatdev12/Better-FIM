import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from betterFIM import betterFIM

if __name__ == "__main__":
    links_file = "dataset/synth3.links"
    attr_file = "dataset/synth3.attr"
    results = []
    for i in range(20):
      result = betterFIM(links_file, attr_file)
      results.append(result)
    for i, (fit, (mf, dcv), seed_set) in enumerate(results):
        print(f"Run {i+1}: F={fit:.4f}, MF={mf:.4f}, DCV={dcv:.4f}")
    avg_F = sum([r[0] for r in results]) / len(results)
    print(f"Average F over 20 runs: {avg_F:.4f}")