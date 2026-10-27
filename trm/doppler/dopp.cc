// The numerical core of Doppler tomography, free of Python. op projects
// images into data, tr is its transpose and defimage computes the default
// images for MEM. op and tr include openmp-parallelised sections so take
// care with thread-safety if changing either of them.

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>
#include <fftw3.h>
#include <omp.h>
#include "dopp.h"

using cplx = std::complex<double>;

// FFT buffers come from fftw_malloc for the alignment FFTW's SIMD code
// needs, which std::vector doesn't guarantee. op also relies on it: its
// threads execute the plans on their own buffers, which must be aligned
// like the ones the plans were made with.
template <class T> using Buf = std::unique_ptr<T[], decltype(&fftw_free)>;

template <class T> static Buf<T> falloc(size_t n) {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc): see Buf
    auto* p = static_cast<T*>(fftw_malloc(sizeof(T) * n));
    if(!p) throw std::bad_alloc();
    return Buf<T>(p, fftw_free);
}

template <class T> static std::span<T> view(const Buf<T>& b, size_t n) {
    return {b.get(), n};
}

static fftw_complex* fc(cplx* p) {
    return reinterpret_cast<fftw_complex*>(p);
}

// FFTW's planner isn't thread-safe, and op and tr can run in several Python
// threads at once as the GIL is released around them.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::mutex planner;

struct Unplan {
    void operator()(fftw_plan p) const {
        const std::lock_guard lock(planner);
        fftw_destroy_plan(p);
    }
};
using Plan = std::unique_ptr<fftw_plan_s, Unplan>;

static Plan r2c(size_t n, double* in, cplx* out) {
    const std::lock_guard lock(planner);
    Plan p(
        fftw_plan_dft_r2c_1d(static_cast<int>(n), in, fc(out), FFTW_ESTIMATE));
    if(!p) throw std::runtime_error("doppler: FFTW planning failed");
    return p;
}

static Plan c2r(size_t n, cplx* in, double* out) {
    const std::lock_guard lock(planner);
    Plan p(
        fftw_plan_dft_c2r_1d(static_cast<int>(n), fc(in), out, FFTW_ESTIMATE));
    if(!p) throw std::runtime_error("doppler: FFTW planning failed");
    return p;
}

// Gaussian of the given sigma (pixels) running nh-1 pixels each side of
// b[0], wrapped round to the end of b. It is normalised to unit sum times
// b.size() so that an FFT / inverse FFT pair needs no further scaling.
static void kernel(std::span<double> b, size_t nh, double sigma) {
    const size_t n = b.size();
    std::ranges::fill(b, 0.);
    double norm = b[0] = 1.;
    for(size_t m = 1; m < nh; m++) {
        const double prf =
            std::exp(-std::pow(static_cast<double>(m) / sigma, 2) / 2.);
        b[m] = b[n - m] = prf;
        norm += 2. * prf;
    }
    norm *= static_cast<double>(n);
    for(double& x : b) x /= norm;
}

// Blurs lines of n pixels spaced by step, starting at each of starts, by a
// gaussian of FWHM fwhm pixels. The FFTs are zero-padded so the ends of the
// lines don't wrap round into each other.
static void blur(std::span<float> img, size_t n, size_t step,
                 const std::vector<size_t>& starts, double fwhm) {
    if(n < 2 || fwhm <= 0.) return;

    const size_t nadd = 2 * static_cast<size_t>(3. * fwhm + 1.);
    const size_t NTOT = std::bit_ceil(n + nadd);
    const size_t NFFT = NTOT / 2 + 1;
    const auto a = falloc<double>(NTOT);
    const auto b = falloc<double>(NTOT);
    const auto afft = falloc<cplx>(NFFT);
    const auto bfft = falloc<cplx>(NFFT);
    const Plan pb = r2c(NTOT, b.get(), bfft.get());
    const Plan pf = r2c(NTOT, a.get(), afft.get());
    const Plan pi = c2r(NTOT, afft.get(), a.get());

    kernel(view(b, NTOT), nadd / 2, fwhm / EFAC);
    fftw_execute(pb.get());

    const auto as = view(a, NTOT);
    const auto af = view(afft, NFFT);
    const auto bf = view(bfft, NFFT);
    for(size_t start : starts) {
        for(size_t i = 0; i < n; i++)
            as[i] = static_cast<double>(img[start + i * step]);
        std::ranges::fill(as.subspan(n), 0.);
        fftw_execute(pf.get());
        for(size_t k = 0; k < NFFT; k++) af[k] *= bf[k];
        fftw_execute(pi.get());
        for(size_t i = 0; i < n; i++)
            img[start + i * step] = static_cast<float>(as[i]);
    }
}

// Blurs an image in each direction in turn. FWHMs in pixels, <= 0 to skip.
static void gaussblur(std::span<float> img, const Image& im, double fx,
                      double fy, double fz) {
    const size_t nx = im.nx;
    const size_t ny = im.ny;
    const size_t nz = im.nz;
    const size_t nxy = nx * ny;
    std::vector<size_t> sx;
    std::vector<size_t> sy;
    std::vector<size_t> sz;
    for(size_t iz = 0; iz < nz; iz++) {
        for(size_t iy = 0; iy < ny; iy++) sx.push_back(iz * nxy + iy * nx);
        for(size_t ix = 0; ix < nx; ix++) sy.push_back(iz * nxy + ix);
    }
    for(size_t ixy = 0; ixy < nxy; ixy++) sz.push_back(ixy);

    blur(img, nx, 1, sx, fx);
    blur(img, ny, nx, sy, fy);
    blur(img, nz, nxy, sz, fz);
}

// The squeezed part of a GAUSS3D default: at each vx-vy, a gaussian in vz
// of FWHM sqfwhm centred on vz = 0, with the same total along vz as the
// image, times the squeeze factor. Mixing this in pulls the default
// towards vz = 0.
static std::vector<float> squeezed(std::span<const float> in, const Image& im) {
    const size_t nxy = im.nx * im.ny;
    const auto sigma = static_cast<float>(im.sqfwhm / EFAC);
    std::vector<float> sq(im.npix());
    for(size_t ixy = 0; ixy < nxy; ixy++) {
        double znorm = 0.;
        double znorm2 = 0.;
        auto vslice =
            static_cast<float>(-im.vz * static_cast<double>(im.nz - 1) / 2);
        for(size_t iz = 0, ip = ixy; iz < im.nz; iz++, ip += nxy) {
            znorm += in[ip];
            sq[ip] =
                static_cast<float>(std::exp(-std::pow(vslice / sigma, 2) / 2.));
            znorm2 += sq[ip];
            vslice = static_cast<float>(vslice + im.vz);
        }

        const double s = znorm / znorm2 * im.squeeze;
        for(size_t iz = 0, ip = ixy; iz < im.nz; iz++, ip += nxy)
            sq[ip] = static_cast<float>(sq[ip] * s);
    }
    return sq;
}

void defimage(const Map& map, std::span<const float> image,
              std::span<float> def) {
    size_t off = 0;
    for(const Image& im : map.images) {
        const size_t npix = im.npix();
        const auto in = image.subspan(off, npix);
        const auto out = def.subspan(off, npix);
        off += npix;

        if(im.def == DefOpt::UNIFORM) {
            double ave = 0.;
            for(float f : in) ave += f;
            ave /= static_cast<double>(npix);
            ave *= im.bias;
            std::ranges::fill(out, static_cast<float>(ave));
            continue;
        }

        const double fxy = im.fwhmxy / im.vxy;
        const double fz = im.nz > 1 ? im.fwhmz / im.vz : 0.;
        std::ranges::copy(in, out.begin());
        gaussblur(out, im, fxy, fxy, fz);

        if(im.def == DefOpt::GAUSS3D && im.squeeze > 0.) {
            std::vector<float> sq = squeezed(in, im);
            gaussblur(sq, im, fxy, fxy, 0.);
            for(size_t ip = 0; ip < npix; ip++)
                out[ip] =
                    static_cast<float>((1 - im.squeeze) * out[ip] + sq[ip]);
        }

        if(im.bias != 1.0)
            for(float& f : out) f = static_cast<float>(f * im.bias);
    }
}

// Sizes of the fine array that op and tr project images onto before
// blurring them to the resolution of the data
struct Fine {
    size_t nblurr; // the blurring kernel runs nblurr-1 pixels each side
    int nfine;     // pixels spanning all projected velocities, plus blurring
    double vmax;   // half the velocity range of the fine array
    size_t NFINE;  // power of 2 >= nfine+nblurr so the blurring can't wrap
    size_t NFFT;
};

static Fine sizes(const Map& map, const std::vector<Spectra>& data) {
    Fine f{};
    for(const Spectra& sp : data) {
        const double psigma = sp.fwhm / map.vfine / EFAC;
        f.nblurr = std::max(f.nblurr,
                            static_cast<size_t>(std::round(6. * psigma) + 1));
    }

    // twice the largest projected velocity of any pixel
    double vmax = 0.;
    for(const Image& im : map.images) {
        const auto nx = static_cast<double>(im.nx);
        const auto ny = static_cast<double>(im.ny);
        double v2 = std::pow(im.vxy, 2) * (std::pow(nx, 2) + std::pow(ny, 2));
        if(im.nz > 1) v2 += std::pow(im.vz * static_cast<double>(im.nz), 2);
        vmax = std::max(std::sqrt(v2), vmax);
    }

    f.nfine =
        std::max(5, static_cast<int>(std::ceil(vmax / map.vfine)
                                     + 2. * static_cast<double>(f.nblurr)));
    f.vmax = f.nfine * map.vfine / 2.;
    f.NFINE = std::bit_ceil(static_cast<size_t>(f.nfine) + f.nblurr);
    f.NFFT = f.NFINE / 2 + 1;
    return f;
}

// Per-thread work space for op
struct Work {
    Buf<double> fine, tfine;
    Buf<cplx> fft;
};

// What op and tr share: sizes, the FFT of the current dataset's blurring
// kernel, FFT plans and work space
struct Proj {
    Proj(const Map& m, const std::vector<Spectra>& data, int nwork);

    const Map& map;
    Fine f;
    Buf<double> kern;
    Buf<cplx> kfft;
    std::vector<Work> work;
    Plan pk, pf, pb;
};

Proj::Proj(const Map& m, const std::vector<Spectra>& data, int nwork)
    : map(m), f(sizes(m, data)), kern(falloc<double>(f.NFINE)),
      kfft(falloc<cplx>(f.NFFT)) {
    for(int i = 0; i < nwork; i++)
        work.push_back(Work{falloc<double>(f.NFINE),
                            falloc<double>(static_cast<size_t>(f.nfine)),
                            falloc<cplx>(f.NFFT)});
    pk = r2c(f.NFINE, kern.get(), kfft.get());
    pf = r2c(f.NFINE, work[0].fine.get(), work[0].fft.get());
    pb = c2r(f.NFINE, work[0].fft.get(), work[0].fine.get());
}

// Sets the blurring kernel for a dataset of resolution fwhm km/s
static void set_kernel(Proj& p, double fwhm) {
    kernel(view(p.kern, p.f.NFINE), p.f.nblurr, fwhm / p.map.vfine / EFAC);
    fftw_execute(p.pk.get());
}

// Convolves the fine array with the blurring kernel
static void blur_fine(const Proj& p, Work& wk) {
    fftw_execute_dft_r2c(p.pf.get(), wk.fine.get(), fc(wk.fft.get()));
    const auto a = view(wk.fft, p.f.NFFT);
    const auto b = view(p.kfft, p.f.NFFT);
    for(size_t k = 0; k < p.f.NFFT; k++) a[k] *= b[k];
    fftw_execute_dft_c2r(p.pb.get(), fc(wk.fft.get()), wk.fine.get());
}

// Whether any line of an image can land in any spectrum of a dataset
static bool overlaps(const Image& im, const Spectra& sp,
                     std::span<const double> w, double vmax) {
    for(size_t k = 0; k < im.wave.size(); k++) {
        for(size_t ns = 0; ns < sp.nspec; ns++) {
            const auto ws = w.subspan(ns * sp.nwave, sp.nwave);
            const double v1 =
                CKMS * (ws.front() / im.wave[k] - 1.) - im.gamma[k];
            const double v2 =
                CKMS * (ws.back() / im.wave[k] - 1.) - im.gamma[k];
            if(v1 < vmax && v2 > -vmax) return true;
        }
    }
    return false;
}

// Orbital phase at time t, inverting t = tzero + period*phi + quad*phi**2
// with two Newton-Raphson steps from the linear estimate
static double phase(const Map& map, double t) {
    double phi = (t - map.tzero) / map.period;
    for(int nc = 0; nc < 2; nc++) {
        const double corr =
            map.tzero + map.period * phi + map.quad * std::pow(phi, 2) - t;
        const double deriv = map.period + 2. * map.quad * phi;
        phi -= corr / deriv;
    }
    return phi;
}

// Time of sub-exposure nt of ntdiv, spread evenly from the start to the end
// of exposure ns. Times are mid-exposure.
static double tsub(const Spectra& sp, size_t ns, int nt, int ntdiv) {
    return sp.time[ns]
           + sp.expose[ns]
                 * (static_cast<float>(nt) - static_cast<float>(ntdiv - 1) / 2.)
                 / std::max(ntdiv - 1, 1);
}

static double itfac(Itype t, double phi) {
    const double pi = std::numbers::pi;
    switch(t) {
    case Itype::PUNIT:
        return +1.;
    case Itype::NUNIT:
        return -1.;
    case Itype::PSINE:
        return +std::sin(2. * pi * phi);
    case Itype::NSINE:
        return -std::sin(2. * pi * phi);
    case Itype::PCOSINE:
        return +std::cos(2. * pi * phi);
    case Itype::NCOSINE:
        return -std::cos(2. * pi * phi);
    case Itype::PSINE2:
        return +std::sin(4. * pi * phi);
    case Itype::NSINE2:
        return -std::sin(4. * pi * phi);
    case Itype::PCOSINE2:
        return +std::cos(4. * pi * phi);
    case Itype::NCOSINE2:
        return -std::cos(4. * pi * phi);
    }
    assert(false);
    return 0.;
}

// Weight of sub-exposure nt of ntdiv: trapezoidal so that results hardly
// depend on ntdiv, times vxy**2 so that pixel intensities are per
// (km/s)**2 whatever the pixel size.
static double weight(double itf, double vxy, int nt, int ntdiv) {
    if(ntdiv > 1 && (nt == 0 || nt == ntdiv - 1))
        return itf * std::pow(vxy, 2) / (2 * (ntdiv - 1));
    return itf * std::pow(vxy, 2) / std::max(1, ntdiv - 1);
}

// Adds the image into the fine array t at the velocities its pixels project
// to at an orbital phase with cosine cosp and sine sinp. Zero velocity is at
// the centre of t; systemic velocities are added later. 3D images are
// spaced by vz symmetrically around vz = 0. These are the innermost loops,
// so the offsets, in fine pixels, are stepped rather than recomputed.
static void project(std::span<const float> img, const Image& im, double cosp,
                    double sinp, double vfine, std::span<double> t) {
    const int nfine = static_cast<int>(t.size());
    const double pxstep = -im.vxy / vfine * cosp;
    const double pystep = +im.vxy / vfine * sinp;
    double pzoff = static_cast<double>(nfine - 1) / 2.;
    double pzstep = 0.;
    if(im.nz > 1) {
        pzoff += im.vz * static_cast<double>(im.nz + 1) / 2. / vfine;
        pzstep = -im.vz / vfine;
    }

    size_t ip = 0;
    for(size_t iz = 0; iz < im.nz; iz++) {
        pzoff += pzstep;
        double pyoff = pzoff - pystep * static_cast<double>(im.ny + 1) / 2.;
        for(size_t iy = 0; iy < im.ny; iy++) {
            pyoff += pystep;
            double pxoff = pyoff - pxstep * static_cast<double>(im.nx + 1) / 2.;
            for(size_t ix = 0; ix < im.nx; ix++, ip++) {
                pxoff += pxstep;
                const int nf = static_cast<int>(std::round(pxoff));
                if(nf >= 0 && nf < nfine) t[static_cast<size_t>(nf)] += img[ip];
            }
        }
    }
}

// Transpose of project: adds the fine array t into the image
static void backproject(std::span<float> img, const Image& im, double cosp,
                        double sinp, double vfine, std::span<const double> t) {
    const int nfine = static_cast<int>(t.size());
    const size_t nx = im.nx;
    const size_t ny = im.ny;
    const size_t nz = im.nz;
    const double pxstep = -im.vxy / vfine * cosp;
    const double pystep = im.vxy / vfine * sinp;
    const double pzstep = nz > 1 ? -im.vz / vfine : 0.;

    // parallelise over y for thin images, else over z
    if(nz < 4) {
        double pzoff = static_cast<double>(nfine - 1) / 2.
                       - pzstep * static_cast<double>(nz + 1) / 2.;
        for(size_t iz = 0; iz < nz; iz++) {
            pzoff += pzstep;
            const double pyoff0 =
                pzoff - pystep * static_cast<double>(ny - 1) / 2.;
#pragma omp parallel for
            for(size_t iy = 0; iy < ny; iy++) {
                const double pyoff = pyoff0 + pystep * static_cast<double>(iy);
                double pxoff =
                    pyoff - pxstep * static_cast<double>(nx + 1) / 2.;
                size_t ip = (iz * ny + iy) * nx;
                for(size_t ix = 0; ix < nx; ix++, ip++) {
                    pxoff += pxstep;
                    const int nf = static_cast<int>(std::round(pxoff));
                    if(nf >= 0 && nf < nfine)
                        img[ip] = static_cast<float>(
                            img[ip] + t[static_cast<size_t>(nf)]);
                }
            }
        }

    } else {
        const double pzoff0 = static_cast<double>(nfine - 1) / 2.
                              - pzstep * static_cast<double>(nz - 1) / 2.;
#pragma omp parallel for
        for(size_t iz = 0; iz < nz; iz++) {
            const double pzoff = pzoff0 + pzstep * static_cast<double>(iz);
            double pyoff = pzoff - pystep * static_cast<double>(ny + 1) / 2.;
            size_t ip = iz * ny * nx;
            for(size_t iy = 0; iy < ny; iy++) {
                pyoff += pystep;
                double pxoff =
                    pyoff - pxstep * static_cast<double>(nx + 1) / 2.;
                for(size_t ix = 0; ix < nx; ix++, ip++) {
                    pxoff += pxstep;
                    const int nf = static_cast<int>(std::round(pxoff));
                    if(nf >= 0 && nf < nfine)
                        img[ip] = static_cast<float>(
                            img[ip] + t[static_cast<size_t>(nf)]);
                }
            }
        }
    }
}

// Projects one image onto the fine array for spectrum ns, summing over the
// sub-exposures that model its finite exposure
static void fine_proj(const Proj& p, const Image& im,
                      std::span<const float> img, const Spectra& sp, size_t ns,
                      Work& wk) {
    const auto fine = view(wk.fine, p.f.NFINE);
    const auto tfine = view(wk.tfine, static_cast<size_t>(p.f.nfine));
    std::ranges::fill(fine, 0.);

    const int ntdiv = sp.nsub[ns];
    for(int nt = 0; nt < ntdiv; nt++) {
        std::ranges::fill(tfine, 0.);
        const double phi = phase(p.map, tsub(sp, ns, nt, ntdiv));
        const double pi = std::numbers::pi;
        project(img, im, std::cos(2. * pi * phi), std::sin(2. * pi * phi),
                p.map.vfine, tfine);
        const double wt = weight(itfac(im.itype, phi), im.vxy, nt, ntdiv);
        for(size_t nf = 0; nf < tfine.size(); nf++) fine[nf] += wt * tfine[nf];
    }
}

// Transpose of fine_proj: adds the fine array for spectrum ns into the image
static void fine_back(const Proj& p, const Image& im, std::span<float> img,
                      const Spectra& sp, size_t ns, Work& wk) {
    const auto fine = view(wk.fine, p.f.NFINE);
    const auto tfine = view(wk.tfine, static_cast<size_t>(p.f.nfine));

    const int ntdiv = sp.nsub[ns];
    for(int nt = 0; nt < ntdiv; nt++) {
        const double phi = phase(p.map, tsub(sp, ns, nt, ntdiv));
        const double pi = std::numbers::pi;
        const double wt = weight(itfac(im.itype, phi), im.vxy, nt, ntdiv);
        for(size_t nf = 0; nf < tfine.size(); nf++) tfine[nf] = wt * fine[nf];
        backproject(img, im, std::cos(2. * pi * phi), std::sin(2. * pi * phi),
                    p.map.vfine, tfine);
    }
}

// Extent of a data pixel in the fine array: fp1 to fp2, with whole fine
// pixels i1 <= i < i2 inside it
struct Extent {
    double fp1, fp2;
    size_t i1, i2;
};

static Extent extent(double v1, double v2, const Proj& p) {
    const int nfine = p.f.nfine;
    Extent e{};
    e.fp1 = v1 / p.map.vfine + static_cast<double>(nfine - 1) / 2.;
    e.fp2 = v2 / p.map.vfine + static_cast<double>(nfine - 1) / 2.;
    e.i1 = static_cast<size_t>(
        std::min(nfine, std::max(0, static_cast<int>(std::ceil(e.fp1 + 0.5)))));
    e.i2 = static_cast<size_t>(std::min(
        nfine, std::max(0, static_cast<int>(std::floor(e.fp2 + 0.5)))));
    return e;
}

// Calls fn(m, e, sc, dv) for each pixel m of a spectrum with wavelengths w
// that a line of the image can reach, where e is the pixel's extent in the
// fine array, sc the line's scale factor and dv the pixel's width in
// velocity divided by wavelength, which converts to flux per unit
// frequency. The first and last pixels are skipped as we can't tell their
// extent.
template <class Fn>
static void pixels(const Proj& p, const Image& im, std::span<const double> w,
                   Fn fn) {
    for(size_t k = 0; k < im.wave.size(); k++) {
        const double wv = im.wave[k];
        const float gm = im.gamma[k];
        const double sc = p.map.sfac * im.scale[k];

        // left-hand side of the first pixel
        double v1 = CKMS * ((w[0] + w[1]) / 2. / wv - 1.) - gm;
        for(size_t m = 1; m < w.size() - 1; m++) {
            const double v2 = CKMS * ((w[m] + w[m + 1]) / 2. / wv - 1.) - gm;
            if(v1 < p.f.vmax && v2 > -p.f.vmax)
                fn(m, extent(v1, v2, p), sc, std::abs(v2 - v1) / w[m]);
            v1 = v2;
        }
    }
}

// Adds the blurred fine array into the pixels of one spectrum
static void fine_to_data(const Proj& p, const Image& im,
                         std::span<const double> fine,
                         std::span<const double> w, std::span<float> d) {
    const auto nfine = static_cast<size_t>(p.f.nfine);
    pixels(p, im, w, [&](size_t m, const Extent& e, double sc, double dv) {
        double sum = 0.;
        for(size_t nf = e.i1; nf < e.i2; nf++) sum += fine[nf];
        if(e.i1 > 0)
            sum += (static_cast<double>(e.i1) - 0.5 - e.fp1) * fine[e.i1 - 1];
        if(e.i2 < nfine)
            sum += (e.fp2 - static_cast<double>(e.i2) + 0.5) * fine[e.i2];
        d[m] = static_cast<float>(d[m] + sc * sum / dv);
    });
}

// Transpose of fine_to_data
static void data_to_fine(const Proj& p, const Image& im,
                         std::span<const float> d, std::span<const double> w,
                         std::span<double> fine) {
    const auto nfine = static_cast<size_t>(p.f.nfine);
    pixels(p, im, w, [&](size_t m, const Extent& e, double sc, double dv) {
        const double add = sc * d[m] / dv;
        for(size_t nf = e.i1; nf < e.i2; nf++) fine[nf] += add;
        if(e.i1 > 0)
            fine[e.i1 - 1] += (static_cast<double>(e.i1) - 0.5 - e.fp1) * add;
        if(e.i2 < nfine)
            fine[e.i2] += (e.fp2 - static_cast<double>(e.i2) + 0.5) * add;
    });
}

void op(const Map& map, std::span<const float> image,
        const std::vector<Spectra>& data, std::span<const double> wave,
        std::span<float> flux) {
    Proj p(map, data, omp_get_max_threads());
    std::ranges::fill(flux, 0.f);

    size_t doff = 0;
    for(const Spectra& sp : data) {
        const auto w = wave.subspan(doff, sp.npix());
        const auto d = flux.subspan(doff, sp.npix());
        doff += sp.npix();
        set_kernel(p, sp.fwhm);

        size_t ioff = 0;
        for(const Image& im : map.images) {
            const auto img = image.subspan(ioff, im.npix());
            ioff += im.npix();
            if(!overlaps(im, sp, w, p.f.vmax)) continue;

#pragma omp parallel for
            for(size_t ns = 0; ns < sp.nspec; ns++) {
                Work& wk = p.work[static_cast<size_t>(omp_get_thread_num())];
                fine_proj(p, im, img, sp, ns, wk);
                blur_fine(p, wk);
                fine_to_data(p, im, view(wk.fine, p.f.NFINE),
                             w.subspan(ns * sp.nwave, sp.nwave),
                             d.subspan(ns * sp.nwave, sp.nwave));
            }
        }
    }
}

void tr(const Map& map, std::span<float> image,
        const std::vector<Spectra>& data, std::span<const double> wave,
        std::span<const float> flux) {
    // the image loops are parallelised here rather than the spectra,
    // as every spectrum adds into the same image
    Proj p(map, data, 1);
    Work& wk = p.work[0];
    const auto fine = view(wk.fine, p.f.NFINE);
    std::ranges::fill(image, 0.f);

    size_t doff = 0;
    for(const Spectra& sp : data) {
        const auto w = wave.subspan(doff, sp.npix());
        const auto d = flux.subspan(doff, sp.npix());
        doff += sp.npix();
        set_kernel(p, sp.fwhm);

        size_t ioff = 0;
        for(const Image& im : map.images) {
            const auto img = image.subspan(ioff, im.npix());
            ioff += im.npix();
            if(!overlaps(im, sp, w, p.f.vmax)) continue;

            for(size_t ns = 0; ns < sp.nspec; ns++) {
                std::ranges::fill(fine, 0.);
                data_to_fine(p, im, d.subspan(ns * sp.nwave, sp.nwave),
                             w.subspan(ns * sp.nwave, sp.nwave), fine);
                blur_fine(p, wk);
                fine_back(p, im, img, sp, ns, wk);
            }
        }
    }
}
