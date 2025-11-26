import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from betterFIM import betterFIM

if __name__ == "__main__":
    links_file = "dataset/rice_subset.pickle"
    attr_file = None
    
    results = []
    mf_list = []
    dcv_list = []
    
    print("Running Better-FIM algorithm...")
    for i in range(10):
        result = betterFIM(links_file, attr_file)
        if result:
            fit, (mf, dcv), seed_set = result
            results.append((fit, mf, dcv, seed_set))
            mf_list.append(mf)
            dcv_list.append(dcv)
            print(f"F = mean(MF) - mean(DCV) = {mf - dcv:.2f} (MF: {mf:.4f}, DCV: {dcv:.4f})")
    
    if results:
        # Tính F giống CEA-FIM: F = mean(MF) - mean(DCV)
        import numpy as np
        avg_mf = np.mean(mf_list)
        avg_dcv = np.mean(dcv_list)
        F_score = avg_mf - avg_dcv
        
        print("\n" + "="*60)
        print(f"Results over {len(results)} runs:")
        print(f"Average MF (Min Fraction): {avg_mf:.4f}")
        print(f"Average DCV (Deviation Coverage Violation): {avg_dcv:.4f}")
        print(f"F = mean(MF) - mean(DCV) = {F_score:.4f}")
        print("="*60)
