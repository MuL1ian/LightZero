#!/usr/bin/env python3
"""
Runner script to execute all validation tests for the MassGym implementation.

This script runs:
1. Ground-truth trajectory recovery and action masking validation
2. Value recovery and discounted returns computation validation
"""

import sys
import os
import subprocess
import time

def run_test_script(script_name, description):
    """Run a test script and report results"""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING: {description}")
    print(f"   Script: {script_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            status = "✅ PASSED"
            print(result.stdout)
        else:
            status = "❌ FAILED"
            print(result.stdout)
            if result.stderr:
                print("\n🔴 STDERR:")
                print(result.stderr)
        
        print(f"\n⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"🎯 Final status: {status}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"❌ Test script timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test script: {e}")
        return False


def main():
    """Run all validation tests"""
    print("🌟 MassGym Implementation Validation Test Suite")
    print("=" * 80)
    print("This suite validates the complete implementation including:")
    print("  1. Ground-truth trajectory recovery")
    print("  2. Action masking in environment") 
    print("  3. Dynamic episode length prediction")
    print("  4. Value recovery and discounted returns computation")
    print("  5. Mathematical properties and edge cases")
    
    # Test configurations
    tests = [
        {
            'script': 'test_ground_truth_trajectory_validation.py',
            'description': 'Ground-Truth Trajectory Recovery & Action Masking Validation'
        },
        {
            'script': 'test_value_recovery_validation.py', 
            'description': 'Value Recovery & Discounted Returns Computation Validation'
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    # Run each test
    for test in tests:
        if os.path.exists(test['script']):
            success = run_test_script(test['script'], test['description'])
            results.append({
                'name': test['description'],
                'script': test['script'],
                'passed': success
            })
        else:
            print(f"\n❌ Test script not found: {test['script']}")
            results.append({
                'name': test['description'],
                'script': test['script'],
                'passed': False
            })
    
    # Overall summary
    total_execution_time = time.time() - total_start_time
    passed_tests = sum(1 for r in results if r['passed'])
    total_tests = len(results)
    
    print(f"\n{'='*80}")
    print("🏆 FINAL VALIDATION RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📊 Test Summary:")
    for result in results:
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"  {status} - {result['name']}")
    
    print(f"\n🎯 Overall Statistics:")
    print(f"   Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
    print(f"   Total Execution Time: {total_execution_time:.2f} seconds")
    
    # Determine overall result
    overall_success = passed_tests == total_tests
    
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("   ✅ Ground-truth trajectory recovery is working correctly")
        print("   ✅ Action masking and dynamic episode length are functioning properly")
        print("   ✅ Value recovery and discounted returns computation are mathematically correct")
        print("   ✅ Edge cases and numerical stability are handled appropriately")
        print("\n🚀 The MassGym implementation is ready for training!")
    else:
        print(f"\n⚠️  SOME TESTS FAILED!")
        print("   Please review the detailed output above to identify and fix issues.")
        print("   Consider running individual test scripts for more detailed debugging.")
        
        failed_tests = [r for r in results if not r['passed']]
        print(f"\n🔧 Failed Tests Requiring Attention:")
        for test in failed_tests:
            print(f"   ❌ {test['name']} ({test['script']})")
    
    print(f"\n{'='*80}")
    return overall_success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error during test execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 