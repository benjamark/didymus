import os
import subprocess
import sys
sys.path.append('../src')

def run_test(test_name):
    print(f"Running {test_name}...")
    subprocess.run(["python", f"./{test_name}.py"])

def run_all_tests():
    for test in os.listdir("."):
        if test.startswith("test_") and test.endswith(".py"):
            run_test(test[:-3])

if __name__ == "__main__":
    run_all_tests()
    # to run a specific test, call run_test('test_name')
    # eg: run_test('test_ray_triangle_intersection')
