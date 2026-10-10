# Drop the scikit-build editable finder so graph2mat imports come from the worktree on PYTHONPATH.
import sys
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]
