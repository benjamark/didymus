import os

def run_all_tests():
    # list all test scripts in the unittests/ directory
    test_scripts = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]

    for script in test_scripts:
        print(f"Running {script}...")
        os.system(f"python {script}")

run_all_tests()
