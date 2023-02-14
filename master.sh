rm -rf outputs
rm bfloat16_*.log
source hpu_test_001.sh > bfloat16_A21_1.log 2>&1
source hpu_test_002.sh > bfloat16_A21_2.log 2>&1
source hpu_test_004.sh > bfloat16_A21_4.log 2>&1
source hpu_test_008.sh > bfloat16_A21_8.log 2>&1
