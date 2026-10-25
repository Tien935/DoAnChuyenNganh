import sys
from pathlib import Path

# Add current dir to path
sys.path.append(str(Path(__file__).parent))

from test_similar_classes import run_similar_classes_test
from test_prompt_sensitivity import run_prompt_sensitivity_test
from test_ood_unknown import run_ood_unknown_test

def main():
    print("******************************************************************")
    print("   CHUONG TRINH KIEM THU TOAN DIEN VLM - TUAN 7 (LE DAI CUONG)   ")
    print("******************************************************************\n")

    # 1. Kiem thu cac lop de nham lan
    df_sim, cm_sim, rep_sim = run_similar_classes_test()
    print("\n" + "-"*60 + "\n")

    # 2. Kiem thu do nhay cam prompt
    df_prompt_sum, df_prompt_det = run_prompt_sensitivity_test(sample_per_class=30)
    print("\n" + "-"*60 + "\n")

    # 3. Kiem thu anh OOD va quet nguong unknown
    df_sweep, df_ood = run_ood_unknown_test()
    print("\n" + "-"*60 + "\n")

    print("******************************************************************")
    print("   HOAN TAT TAT CA CAC PHAN KIEM THU TUAN 7!                      ")
    print("******************************************************************")

if __name__ == "__main__":
    main()
