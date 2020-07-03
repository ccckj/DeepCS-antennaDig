import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--measurement', type=int)
args = parser.parse_args()
print("The result is:",args.measurement)
