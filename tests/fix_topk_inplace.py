import sys

path = "csrc/ops/pointwise_ops.cpp"
with open(path) as f:
    content = f.read()

old = '''                scalar_t prev_val = scalar_t(0);
                int64_t  prev_idx = -1;
                bool     has_prev = false;

                for (int64_t s = 0; s < k; ++s) {
                    scalar_t best_val = scalar_t(0);
                    int64_t  best_idx = -1;
                    bool     has_best = false;

                    for (int64_t j = 0; j < red_n; ++j) {
                        const scalar_t v = pin[base_in + rin.index(j)];'''

new = '''                double  prev_val = 0.0;
                int64_t prev_idx = -1;
                bool    has_prev = false;

                for (int64_t s = 0; s < k; ++s) {
                    double  best_val = 0.0;
                    int64_t best_idx = -1;
                    bool    has_best = false;

                    for (int64_t j = 0; j < red_n; ++j) {
                        const double v = static_cast<double>(
                            pin[base_in + rin.index(j)]);'''

if old not in content:
    print("ERROR: expected block not found -- file may already be fixed, or differs from what I expect.")
    print("Nothing was changed. Paste me the topk section of your file if this happens.")
    sys.exit(1)

content = content.replace(old, new, 1)

old2 = "                    pval[base_val + rval.index(s)] = best_val;"
new2 = "                    pval[base_val + rval.index(s)] = static_cast<scalar_t>(best_val);"

if old2 not in content:
    print("ERROR: second expected line not found. Partial fix may have been applied -- check the file manually.")
    sys.exit(1)

content = content.replace(old2, new2, 1)

with open(path, "w") as f:
    f.write(content)

print("Fix applied successfully to", path)
