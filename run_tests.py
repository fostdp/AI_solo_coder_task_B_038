import subprocess
import sys

def run_tests(test_file):
    print(f"\n{'='*60}")
    print(f"Running tests in: {test_file}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', test_file, '--tb=short', '-v'],
        capture_output=True,
        text=True,
        cwd='d:/SOLO-2/AI_solo_coder_task_B_038'
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode

if __name__ == '__main__':
    test_files = [
        'tests/test_endpoint_detection.py',
        'tests/test_defrost_optimization.py', 
        'tests/test_fleet_control.py',
        'tests/test_defect_detection.py'
    ]
    
    total_failures = 0
    for tf in test_files:
        rc = run_tests(tf)
        if rc != 0:
            total_failures += 1
    
    print(f"\n{'='*60}")
    print(f"Total failing test files: {total_failures}/{len(test_files)}")
    print('='*60)
