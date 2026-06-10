import subprocess
import sys
import os

os.chdir('d:/SOLO-2/AI_solo_coder_task_B_038')

test_files = [
    'tests/test_endpoint_detection.py',
    'tests/test_defrost_optimization.py',
    'tests/test_fleet_control.py',
    'tests/test_defect_detection.py'
]

all_output = []

for tf in test_files:
    header = f"\n{'='*60}\nRunning: {tf}\n{'='*60}\n"
    all_output.append(header)
    
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', tf, '--tb=short', '-v', '--no-header'],
        capture_output=True,
        text=True
    )
    
    all_output.append(result.stdout)
    if result.stderr:
        all_output.append("STDERR: " + result.stderr)

with open('d:/SOLO-2/AI_solo_coder_task_B_038/captured_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_output))

print("Captured", len(all_output), "sections")
print("Results saved to captured_results.txt")
