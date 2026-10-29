// Float64 kernels. Keep IEEE arithmetic: no fastmath or parallel component updates.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;
using FArray = py::array_t<double, py::array::f_style>;
using CArray = py::array_t<double, py::array::c_style>;
// SciPy exports C-callable BLAS wrappers with LP64 int dimensions, even when the
// linked BLAS itself uses a different ABI. Validate the full capsule signature.
using Gemv = void (*)(char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
using Gemm = void (*)(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
static Gemv dgemv = nullptr;
static Gemm dgemm = nullptr;

static int dim(py::ssize_t n) {
    if (n < 0 || n > std::numeric_limits<int>::max())
        throw py::value_error("array dimension exceeds the SciPy BLAS integer range");
    return static_cast<int>(n);
}
static double floor_at(double x, double bound) {
    return std::isnan(x) ? x : std::max(x, bound);
}
static void shape(const py::array& a, std::initializer_list<py::ssize_t> sizes, const char* name) {
    if (a.ndim() != static_cast<py::ssize_t>(sizes.size()))
        throw py::value_error(std::string(name) + " has incorrect dimensions");
    int j = 0;
    for (auto size : sizes) {
        if (a.shape(j++) != size) throw py::value_error(std::string(name) + " has incorrect shape");
    }
}
static void writable(const py::array& a, const char* name) {
    if (!a.writeable()) throw py::value_error(std::string(name) + " must be writable");
}
static void gemv(char trans, int n, int p, const double* G, const double* x, double* y) {
    int count = trans == 'N' ? n : p;
    if (n == 0 || p == 0) {
        std::fill(y, y + count, 0.0);
        return;
    }
    double one = 1.0, zero = 0.0;
    int inc = 1, ld = std::max(n, 1);
    dgemv(&trans, &n, &p, &one, const_cast<double*>(G), &ld,
          const_cast<double*>(x), &inc, &zero, y, &inc);
}

struct Projection {
    int n, p, s;
    const double *G, *u, *T, *A;
    std::vector<double> B;
    Projection(const FArray& genotype, const CArray& weights, const FArray& cov, const FArray& inverse)
        : n(0), p(0), s(0) {
        if (genotype.ndim() != 2 || cov.ndim() != 2)
            throw py::value_error("G and T must be matrices");
        n = dim(genotype.shape(0)); p = dim(genotype.shape(1)); s = dim(cov.shape(1));
        shape(weights, {n}, "u"); shape(cov, {n, s}, "T"); shape(inverse, {s, s}, "Ainv");
        G = genotype.data(); u = weights.data(); T = cov.data(); A = inverse.data();
        B.resize(static_cast<size_t>(s) * p);
    }
    void prepare() {
        if (!s || !p || !n) return;
        char trans = 'T', normal = 'N';
        double one = 1., zero = 0.;
        int ld = std::max(n, 1), outld = std::max(s, 1);
        // B = T.T @ G, stored column-major with shape (s, p).
        dgemm(&trans, &normal, &s, &p, &n, &one, const_cast<double*>(T), &ld,
              const_cast<double*>(G), &ld, &zero, B.data(), &outld);
    }
    double diagonal(int j) const {
        double out = 0., lane0 = 0., lane1 = 0., lane2 = 0., lane3 = 0.;
        const double* col = G + static_cast<size_t>(j) * n;
        // Independent accumulators allow SIMD without fastmath. The summation
        // tree differs from Numba's scalar reduction, within float64 tolerance.
        int i = 0;
        for (; i + 3 < n; i += 4) {
            lane0 += (col[i] * col[i]) * u[i];
            lane1 += (col[i+1] * col[i+1]) * u[i+1];
            lane2 += (col[i+2] * col[i+2]) * u[i+2];
            lane3 += (col[i+3] * col[i+3]) * u[i+3];
        }
        out = ((lane0 + lane1) + lane2) + lane3;
        for (; i < n; ++i) out += (col[i] * col[i]) * u[i];
        if (s) {
            const double* bj = B.data() + static_cast<size_t>(j) * s;
            for (int i = 0; i < s; ++i) {
                double ab = 0.;
                for (int k = 0; k < s; ++k) ab += A[i + static_cast<size_t>(k) * s] * bj[k];
                out -= bj[i] * ab;
            }
        }
        return floor_at(out, 1e-12);
    }
    void apply(const double* v, double* out, std::vector<double>& small) const {
        for (int k = 0; k < s; ++k) {
            double acc = 0.;
            for (int i = 0; i < n; ++i) acc += T[i + static_cast<size_t>(k) * n] * v[i];
            small[k] = acc;
        }
        for (int k = 0; k < s; ++k) {
            double acc = 0.;
            for (int j = 0; j < s; ++j) acc += A[k + static_cast<size_t>(j) * s] * small[j];
            small[s + k] = acc;
        }
        for (int i = 0; i < n; ++i) {
            double correction = 0.;
            for (int k = 0; k < s; ++k) correction += T[i + static_cast<size_t>(k) * n] * small[s + k];
            out[i] = u[i] * v[i] - correction;
        }
    }
};

static CArray diag_gpg(FArray G, CArray u, FArray T, FArray Ainv) {
    Projection projection(G, u, T, Ainv);
    CArray output(projection.p);
    double* out = output.mutable_data();
    {
        py::gil_scoped_release release;
        projection.prepare();
        for (int j = 0; j < projection.p; ++j) out[j] = projection.diagonal(j);
    }
    return output;
}

static double trace_gpg(FArray G, CArray u, FArray T, FArray Ainv, CArray d, FArray U) {
    Projection projection(G, u, T, Ainv);
    shape(d, {projection.p}, "d");
    if (U.ndim() != 2 || U.shape(0) != projection.n)
        throw py::value_error("U must have shape (n, L)");
    int L = dim(U.shape(1));
    std::vector<double> hu(projection.n), small(2 * projection.s);
    double diagonal = 0., rank1 = 0.;
    {
        py::gil_scoped_release release;
        projection.prepare();
        for (int j = 0; j < projection.p; ++j) diagonal += d.data()[j] * projection.diagonal(j);
        for (int l = 0; l < L; ++l) {
            const double* v = U.data() + static_cast<size_t>(l) * projection.n;
            projection.apply(v, hu.data(), small);
            for (int i = 0; i < projection.n; ++i) rank1 += v[i] * hu[i];
        }
    }
    return floor_at(diagonal - rank1, 0.);
}

static void ibss_sweep(FArray G, CArray u, FArray T, FArray Ainv, CArray b, CArray sP,
                       CArray alpha, CArray mu, CArray var, CArray components,
                       CArray prior, CArray total, bool update_prior) {
    Projection projection(G, u, T, Ainv);
    if (components.ndim() != 2) throw py::value_error("components must be a matrix");
    int n = projection.n, p = projection.p, L = dim(components.shape(0));
    shape(b, {n}, "b"); shape(sP, {p}, "sP"); shape(total, {p}, "total");
    shape(prior, {L}, "prior");
    for (auto* array : {&alpha, &mu, &var, &components}) {
        shape(*array, {L, p}, "posterior"); writable(*array, "posterior");
    }
    writable(total, "total"); writable(prior, "prior");
    // Reject overlapping outputs rather than silently corrupting a posterior.
    const py::array* outputs[] = {&alpha, &mu, &var, &components, &prior, &total};
    for (int i = 0; i < 6; ++i) for (int j = i + 1; j < 6; ++j) {
        auto a = reinterpret_cast<uintptr_t>(outputs[i]->data());
        auto z = reinterpret_cast<uintptr_t>(outputs[j]->data());
        if (outputs[i]->nbytes() && outputs[j]->nbytes() &&
            a < z + outputs[j]->nbytes() && z < a + outputs[i]->nbytes())
            throw py::value_error("posterior outputs must not overlap");
    }
    if (!p || !L) return;
    std::vector<double> minus(p), h(n), hh(n), residual(n), a(p), logits(p), small(2 * projection.s);
    auto* ap = alpha.mutable_data(); auto* mp = mu.mutable_data(); auto* vp = var.mutable_data();
    auto* cp = components.mutable_data(); auto* tp = total.mutable_data(); auto* pp = prior.mutable_data();
    {
        py::gil_scoped_release release;
        const double log_prior = -std::log(static_cast<double>(p));
        for (int l = 0; l < L; ++l) {
            size_t offset = static_cast<size_t>(l) * p;
            for (int j = 0; j < p; ++j) minus[j] = tp[j] - cp[offset + j];
            gemv('N', n, p, G.data(), minus.data(), h.data());
            projection.apply(h.data(), hh.data(), small);
            for (int i = 0; i < n; ++i) residual[i] = b.data()[i] - hh[i];
            gemv('T', n, p, G.data(), residual.data(), a.data());
            double sigma2 = floor_at(pp[l], 1e-12);
            double max_logit = -std::numeric_limits<double>::infinity();
            bool nan_logit = false;
            for (int j = 0; j < p; ++j) {
                double variance = floor_at(1. / floor_at(sP.data()[j] + 1. / sigma2, 1e-20), 1e-18);
                double mean = variance * a[j];
                vp[offset + j] = variance; mp[offset + j] = mean;
                logits[j] = log_prior + .5 * (std::log(variance) - std::log(sigma2)) + .5 * (mean * mean / floor_at(variance, 1e-18));
                nan_logit = nan_logit || std::isnan(logits[j]);
                max_logit = std::max(max_logit, logits[j]);
            }
            if (nan_logit) max_logit = std::numeric_limits<double>::quiet_NaN();
            double norm = 0.;
            for (int j = 0; j < p; ++j) { logits[j] = std::exp(logits[j] - max_logit); norm += logits[j]; }
            double pip_sum = 0.;
            bool valid = std::isfinite(norm) && norm > 0.;
            for (int j = 0; j < p; ++j) {
                double pip = valid ? logits[j] / norm : 1. / p;
                ap[offset + j] = std::clamp(pip, 0., 1.); pip_sum += ap[offset + j];
            }
            double new_prior = 0.;
            for (int j = 0; j < p; ++j) {
                ap[offset + j] /= floor_at(pip_sum, 1e-20);
                double mean = mp[offset + j];
                cp[offset + j] = ap[offset + j] * mean;
                tp[j] = minus[j] + cp[offset + j];
                new_prior += ap[offset + j] * (mean * mean + vp[offset + j]);
            }
            if (update_prior) pp[l] = floor_at(new_prior, 1e-12);
        }
    }
}

class PriorLikelihood {
    std::vector<double> log_prior, variance, squared_beta, null_log_density;
    std::vector<bool> valid;
public:
    PriorLikelihood(CArray beta, CArray shat2, CArray prior) {
        if (beta.ndim() != 1) throw py::value_error("betahat must be a vector");
        auto p = beta.shape(0);
        if (!p) throw py::value_error("betahat must not be empty");
        shape(shat2, {p}, "shat2"); shape(prior, {p}, "prior");
        log_prior.resize(p); variance.resize(p); squared_beta.resize(p);
        null_log_density.resize(p); valid.resize(p);
        const double epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
        const double two_pi = 2. * std::acos(-1.);
        py::gil_scoped_release release;
        for (py::ssize_t j = 0; j < p; ++j) {
            log_prior[j] = std::log(prior.data()[j] + epsilon);
            variance[j] = shat2.data()[j];
            squared_beta[j] = beta.data()[j] * beta.data()[j];
            valid[j] = std::isfinite(variance[j]) && variance[j] > 0.;
            if (valid[j]) null_log_density[j] = -.5 * std::log(two_pi * variance[j]) - .5 * squared_beta[j] / variance[j];
        }
    }
    double value(double V) const {
        // Immutable prepared statistics; each evaluation has a local workspace,
        // allowing independent Python workers to release the GIL safely.
        std::vector<double> lpo(log_prior.size());
        double maximum = -std::numeric_limits<double>::infinity(), norm = 0.;
        const double two_pi = 2. * std::acos(-1.);
        py::gil_scoped_release release;
        for (size_t j = 0; j < lpo.size(); ++j) {
            lpo[j] = log_prior[j];
            if (valid[j]) {
                double denom = floor_at(V + variance[j], std::numeric_limits<double>::min());
                lpo[j] += (-.5 * std::log(two_pi * denom) - .5 * squared_beta[j] / denom) - null_log_density[j];
            }
            if (std::isnan(lpo[j])) maximum = std::numeric_limits<double>::quiet_NaN();
            else if (!std::isnan(maximum)) maximum = std::max(maximum, lpo[j]);
        }
        for (double value : lpo) norm += std::exp(value - maximum);
        return maximum + std::log(norm);
    }
};

PYBIND11_MODULE(_cpp, module) {
    py::dict capsules = py::module_::import("scipy.linalg.cython_blas").attr("__pyx_capi__");
    const char* scalar = "__pyx_t_5scipy_6linalg_11cython_blas_d";
    std::string signature_v = "void (char *, int *, int *, " + std::string(scalar) + " *, " + scalar + " *, int *, " + scalar + " *, int *, " + scalar + " *, " + scalar + " *, int *)";
    std::string signature_m = "void (char *, char *, int *, int *, int *, " + std::string(scalar) + " *, " + scalar + " *, int *, " + scalar + " *, int *, " + scalar + " *, " + scalar + " *, int *)";
    py::object cv = capsules["dgemv"], cm = capsules["dgemm"];
    if (!PyCapsule_IsValid(cv.ptr(), signature_v.c_str()) || !PyCapsule_IsValid(cm.ptr(), signature_m.c_str()))
        throw py::import_error("Unsupported SciPy BLAS capsule ABI; use backend='numba' or rebuild for this SciPy version");
    dgemv = reinterpret_cast<Gemv>(PyCapsule_GetPointer(cv.ptr(), signature_v.c_str()));
    dgemm = reinterpret_cast<Gemm>(PyCapsule_GetPointer(cm.ptr(), signature_m.c_str()));
    module.doc() = "Float64 scELF kernels using SciPy BLAS; GIL released during calculation.";
    module.def("diag_gpg", &diag_gpg, py::arg("G").noconvert(), py::arg("u").noconvert(), py::arg("T").noconvert(), py::arg("Ainv").noconvert());
    module.def("trace_gpg", &trace_gpg, py::arg("G").noconvert(), py::arg("u").noconvert(), py::arg("T").noconvert(), py::arg("Ainv").noconvert(), py::arg("d").noconvert(), py::arg("U").noconvert());
    module.def("ibss_sweep", &ibss_sweep,
        py::arg("G").noconvert(), py::arg("u").noconvert(), py::arg("T").noconvert(), py::arg("Ainv").noconvert(),
        py::arg("b").noconvert(), py::arg("sP").noconvert(), py::arg("alpha").noconvert(), py::arg("mu").noconvert(),
        py::arg("var").noconvert(), py::arg("components").noconvert(), py::arg("prior").noconvert(), py::arg("total").noconvert(), py::arg("update_prior") = true);
    module.attr("blas_provider") = "scipy.linalg.cython_blas";
    module.attr("fastmath") = false;
    py::class_<PriorLikelihood>(module, "PriorLikelihood")
        .def(py::init<CArray, CArray, CArray>(), py::arg("betahat").noconvert(), py::arg("shat2").noconvert(), py::arg("prior").noconvert())
        .def("value", &PriorLikelihood::value);
}
