rm -rf outputs
rm bfloat16_*.log
source ipu_test_popdist_1.sh > bfloat16_2x10_1_nvme.log 2>&1
source ipu_test_popdist_2.sh > bfloat16_2x10_2_nvme.log 2>&1
source ipu_test_popdist_4.sh > bfloat16_2x10_4_nvme.log 2>&1
source ipu_test_popdist_8.sh > bfloat16_2x10_8_nvme.log 2>&1
source ipu_test_popdist_16.sh > bfloat16_2x10_16_nvme.log 2>&1
source ipu_test_popdist_64.sh > bfloat16_2x10_64_nvme.log 2>&1
