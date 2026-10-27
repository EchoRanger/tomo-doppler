// The Python interface to op, tr and defimage (dopp.cc) and to the MEM
// routines of memsys (cpp-mem). Most manipulation of Map and Data objects
// is better left to Python, so this just reads them into the structures of
// dopp.h and writes back images and fluxes.
//
// This must be the only file to include trm/memsys.h, since that header
// defines memsys's global variables rather than declaring them.

#include <Python.h>
#include <iostream>
#include <vector>
#include "trm/memsys.h"
#include "dopp.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>

// Thrown when a Python exception has already been set
struct PyErrSet {};

struct Decref {
    void operator()(PyObject* o) const { Py_XDECREF(o); }
};
using Ref = std::unique_ptr<PyObject, Decref>;

static Ref own(PyObject* o) {
    if(!o) throw PyErrSet();
    return Ref(o);
}

static Ref attr(PyObject* o, const char* name) {
    return own(PyObject_GetAttrString(o, name));
}

static double getf(PyObject* o, const char* name) {
    const Ref a = attr(o, name);
    const double v = PyFloat_AsDouble(a.get());
    if(v == -1. && PyErr_Occurred()) throw PyErrSet();
    return v;
}

static long geti(PyObject* o, const char* name) {
    const Ref a = attr(o, name);
    const long v = PyLong_AsLong(a.get());
    if(v == -1 && PyErr_Occurred()) throw PyErrSet();
    return v;
}

// numpy's API functions are macros that index a table of function pointers,
// which clang-tidy reports as casts and pointer arithmetic where they are
// called; hence the NOLINTs on those calls.

static PyArrayObject* arr(const Ref& a) {
    return reinterpret_cast<PyArrayObject*>(a.get());
}

// o.name as a C-contiguous array of numpy type type with nmin to nmax
// dimensions
static Ref array(PyObject* o, const char* name, int type, int nmin, int nmax) {
    const Ref a = attr(o, name);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return own(PyArray_FromAny(a.get(), PyArray_DescrFromType(type), nmin, nmax,
                               NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST,
                               nullptr));
}

template <class T> static std::span<T> values(const Ref& a) {
    return {
        static_cast<T*>(PyArray_DATA(arr(a))),
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
        static_cast<size_t>(PyArray_SIZE(arr(a)))};
}

static size_t dim(const Ref& a, int i) {
    return static_cast<size_t>(PyArray_DIM(arr(a), i));
}

// The objects in o.data: the Images of a Map or the Spectra of a Data
static std::vector<Ref> items(PyObject* o) {
    const Ref seq = attr(o, "data");
    if(!PySequence_Check(seq.get()))
        throw std::invalid_argument(
            "doppler: data attribute is not a sequence");
    const Py_ssize_t n = PySequence_Size(seq.get());
    if(n < 0) throw PyErrSet();
    std::vector<Ref> out;
    for(Py_ssize_t i = 0; i < n; i++)
        out.push_back(own(PySequence_GetItem(seq.get(), i)));
    return out;
}

static Itype to_itype(long v) {
    if(v < 1 || v > 10)
        throw std::invalid_argument("doppler: Image.itype must be 1 to 10");
    return static_cast<Itype>(v);
}

static DefOpt to_defopt(long v) {
    if(v < 1 || v > 3)
        throw std::invalid_argument("doppler: Default.option must be 1 to 3");
    return static_cast<DefOpt>(v);
}

static void read_default(PyObject* o, Image& im) {
    im.def = to_defopt(geti(o, "option"));
    im.bias = getf(o, "bias");
    if(im.bias <= 0.)
        throw std::invalid_argument("doppler: Default.bias must be > 0");

    if(im.def != DefOpt::UNIFORM) im.fwhmxy = getf(o, "fwhmxy");
    if(im.def == DefOpt::GAUSS3D) {
        im.fwhmz = getf(o, "fwhmz");
        im.squeeze = getf(o, "squeeze");
        im.sqfwhm = getf(o, "sqfwhm");
        if(im.squeeze > 0. && im.sqfwhm <= 0.)
            throw std::invalid_argument(
                "doppler: Default.sqfwhm must be > 0 if squeeze > 0");
    }
}

// Reads an Image, appending its pixels to pix
static Image read_image(PyObject* o, std::vector<float>& pix) {
    const Ref data = array(o, "data", NPY_FLOAT, 2, 3);
    const Ref wave = array(o, "wave", NPY_DOUBLE, 1, 1);
    const Ref gamma = array(o, "gamma", NPY_FLOAT, 1, 1);
    const Ref scale = array(o, "scale", NPY_FLOAT, 1, 1);

    Image im{};
    const bool is3d = PyArray_NDIM(arr(data)) == 3;
    im.nz = is3d ? dim(data, 0) : 1;
    im.ny = dim(data, is3d ? 1 : 0);
    im.nx = dim(data, is3d ? 2 : 1);
    const auto p = values<float>(data);
    pix.insert(pix.end(), p.begin(), p.end());

    const auto w = values<double>(wave);
    const auto g = values<float>(gamma);
    const auto s = values<float>(scale);
    if(g.size() != w.size() || s.size() != w.size())
        throw std::invalid_argument(
            "doppler: Image wave, gamma and scale differ in size");
    im.wave.assign(w.begin(), w.end());
    im.gamma.assign(g.begin(), g.end());
    im.scale.assign(s.begin(), s.end());

    im.itype = to_itype(geti(o, "itype"));
    im.vxy = getf(o, "vxy");
    if(im.vxy <= 0.)
        throw std::invalid_argument("doppler: Image.vxy must be > 0");
    if(is3d) {
        im.vz = getf(o, "vz");
        if(im.vz <= 0.)
            throw std::invalid_argument("doppler: Image.vz must be > 0");
    }

    const Ref def = attr(o, "default");
    read_default(def.get(), im);
    return im;
}

// Reads a Map, returning its images' pixels concatenated in pix
static Map read_map(PyObject* o, std::vector<float>& pix) {
    Map m{};
    m.tzero = getf(o, "tzero");
    m.period = getf(o, "period");
    m.quad = getf(o, "quad");
    m.vfine = getf(o, "vfine");
    m.sfac = getf(o, "sfac");
    if(m.period <= 0.)
        throw std::invalid_argument("doppler: Map.period must be > 0");
    if(m.vfine <= 0.)
        throw std::invalid_argument("doppler: Map.vfine must be > 0");

    for(const Ref& im : items(o)) m.images.push_back(read_image(im.get(), pix));
    return m;
}

// Fluxes, their errors and wavelengths of all datasets, concatenated
struct Pixels {
    std::vector<float> flux, ferr;
    std::vector<double> wave;
};

static Spectra read_spectra(PyObject* o, Pixels& px) {
    const Ref flux = array(o, "flux", NPY_FLOAT, 2, 2);
    const Ref ferr = array(o, "ferr", NPY_FLOAT, 2, 2);
    const Ref wave = array(o, "wave", NPY_DOUBLE, 2, 2);
    const Ref time = array(o, "time", NPY_DOUBLE, 1, 1);
    const Ref expose = array(o, "expose", NPY_FLOAT, 1, 1);
    const Ref nsub = array(o, "nsub", NPY_INT, 1, 1);

    Spectra sp{};
    sp.nspec = dim(flux, 0);
    sp.nwave = dim(flux, 1);
    for(const Ref* a : {&ferr, &wave})
        if(dim(*a, 0) != sp.nspec || dim(*a, 1) != sp.nwave)
            throw std::invalid_argument(
                "doppler: Spectra flux, ferr and wave differ in shape");
    for(const Ref* a : {&time, &expose, &nsub})
        if(dim(*a, 0) != sp.nspec)
            throw std::invalid_argument(
                "doppler: Spectra time, expose and nsub need one value"
                " per spectrum");
    if(sp.nwave < 3)
        throw std::invalid_argument(
            "doppler: Spectra need at least 3 pixels per spectrum");

    const auto f = values<float>(flux);
    const auto e = values<float>(ferr);
    const auto w = values<double>(wave);
    px.flux.insert(px.flux.end(), f.begin(), f.end());
    px.ferr.insert(px.ferr.end(), e.begin(), e.end());
    px.wave.insert(px.wave.end(), w.begin(), w.end());

    const auto t = values<double>(time);
    const auto x = values<float>(expose);
    const auto n = values<int>(nsub);
    sp.time.assign(t.begin(), t.end());
    sp.expose.assign(x.begin(), x.end());
    sp.nsub.assign(n.begin(), n.end());
    if(std::ranges::any_of(sp.nsub, [](int v) { return v < 1; }))
        throw std::invalid_argument("doppler: Spectra.nsub must be >= 1");

    sp.fwhm = getf(o, "fwhm");
    if(sp.fwhm <= 0.)
        throw std::invalid_argument("doppler: Spectra.fwhm must be > 0");
    return sp;
}

static std::vector<Spectra> read_data(PyObject* o, Pixels& px) {
    std::vector<Spectra> data;
    for(const Ref& sp : items(o)) data.push_back(read_spectra(sp.get(), px));
    return data;
}

// Copies vals, in sequence, into the arrays name of the objects in o.data
// (image data of a Map, or fluxes of a Data). The copy goes into the
// existing arrays, whatever their type and memory layout.
static void write(PyObject* o, const char* name, std::span<const float> vals) {
    size_t off = 0;
    for(const Ref& it : items(o)) {
        const Ref a = attr(it.get(), name);
        // NOLINTBEGIN(cppcoreguidelines-pro-type-cstyle-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
        const Ref out = own(PyArray_FromAny(
            a.get(), PyArray_DescrFromType(NPY_FLOAT), 0, 0,
            NPY_ARRAY_INOUT_ARRAY2 | NPY_ARRAY_FORCECAST, nullptr));
        // NOLINTEND(cppcoreguidelines-pro-type-cstyle-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
        const auto dst = values<float>(out);
        if(off + dst.size() > vals.size()) {
            PyArray_DiscardWritebackIfCopy(arr(out));
            throw std::invalid_argument("doppler: arrays changed size");
        }
        std::ranges::copy(vals.subspan(off, dst.size()), dst.begin());
        off += dst.size();
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast,cppcoreguidelines-pro-bounds-pointer-arithmetic)
        if(PyArray_ResolveWritebackIfCopy(arr(out)) < 0) throw PyErrSet();
    }
}

// Releases the GIL while in scope; nothing in its scope may touch Python
struct NoGil {
    NoGil() : s(PyEval_SaveThread()) {}
    ~NoGil() { PyEval_RestoreThread(s); }
    PyThreadState* s;
};

// Sets the Python exception for the C++ one being handled
static PyObject* fail() {
    try {
        throw;
    } catch(const PyErrSet&) {
    } catch(const std::bad_alloc&) {
        PyErr_NoMemory();
    } catch(const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch(const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "doppler: unknown C++ exception");
    }
    return nullptr;
}

// memsys calls opus and tropus with buffer numbers only, so they reach the
// map and data through this
struct Memctx {
    const Map* map;
    const std::vector<Spectra>* data;
    std::span<const double> wave;
    std::span<float> st;
    size_t nipix, ndpix;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static Memctx ctx;

// memsys works through globals, so only one memit can run at a time
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::mutex memlock;

static std::span<float> buffer(int k, size_t n) {
    return ctx.st.subspan(static_cast<size_t>(Mem::Gbl::kb[k]), n);
}

void Mem::opus(const int j, const int k) {
    op(*ctx.map, buffer(j, ctx.nipix), *ctx.data, ctx.wave,
       buffer(k, ctx.ndpix));
}

void Mem::tropus(const int k, const int j) {
    tr(*ctx.map, buffer(j, ctx.nipix), *ctx.data, ctx.wave,
       buffer(k, ctx.ndpix));
}

struct Memopt {
    int niter;
    float caim, tlim, rmax;
};

// Runs MEM iterations, updating image
static void mem(const Map& m, const std::vector<Spectra>& data,
                const Pixels& px, std::span<float> image, const Memopt& o) {
    const size_t nipix = image.size();
    const size_t ndpix = px.flux.size();
    const size_t mxbuf = Mem::memsize(nipix, ndpix);
    std::vector<float> buf(mxbuf);
    Mem::Gbl::st = buf.data();
    Mem::memcore(mxbuf, nipix, ndpix);
    ctx = {&m, &data, px.wave, buf, nipix, ndpix};

    try {
        std::ranges::copy(image, buffer(0, nipix).begin());
        std::ranges::copy(px.flux, buffer(20, ndpix).begin());

        // memsys wants weights 2/sigma**2/N. N is the number of data points
        // including masked ones; just the unmasked upsets bootstrapping.
        const auto wt = buffer(21, ndpix);
        for(size_t i = 0; i < ndpix; i++) {
            const float e = px.ferr[i];
            wt[i] = e > 0. ? static_cast<float>(2. / std::pow(e, 2)
                                                / static_cast<double>(ndpix))
                           : e;
        }

        // mode 30 uses the default image in buffer 19, recomputed each
        // iteration; mode 10 a uniform default
        const bool flat = std::ranges::all_of(m.images, [](const Image& im) {
            return im.def == DefOpt::UNIFORM && im.bias == 1.;
        });
        const int mode = flat ? 10 : 30;

        float c = 0.f;
        float test = 0.f;
        float cnew = 0.f;
        float s = 0.f;
        float rnew = 0.f;
        float snew = 0.f;
        float sumf = 0.f;
        for(int it = 0; it < o.niter; it++) {
            std::cerr << "\nIteration " << it + 1 << std::endl;
            if(mode == 30) defimage(m, buffer(0, nipix), buffer(19, nipix));
            Mem::memprm(mode, 20, o.caim, o.rmax, 1.f, 1.f, c, test, cnew, s,
                        rnew, snew, sumf);
            if(test < o.tlim && c <= o.caim) break;
        }
        std::ranges::copy(buffer(0, nipix), image.begin());

    } catch(...) {
        ctx = {};
        Mem::Gbl::st = nullptr;
        throw;
    }
    ctx = {};
    Mem::Gbl::st = nullptr;
}

// Now come the routines that are visible to Python

// Computes data equivalent to a map
static PyObject* doppler_comdat(PyObject*, PyObject* args, PyObject* kwords) {
    PyObject* map = nullptr;
    PyObject* data = nullptr;
    static const char* kwlist[] = {"map", "data", nullptr};
    if(!PyArg_ParseTupleAndKeywords(args, kwords, "OO",
                                    const_cast<char**>(kwlist), &map, &data))
        return nullptr;

    try {
        std::vector<float> image;
        const Map m = read_map(map, image);
        Pixels px;
        const std::vector<Spectra> sp = read_data(data, px);
        {
            const NoGil nogil;
            op(m, image, sp, px.wave, px.flux);
        }
        write(data, "flux", px.flux);
        Py_RETURN_NONE;
    } catch(...) {
        return fail();
    }
}

// Computes the default image of a map
static PyObject* doppler_comdef(PyObject*, PyObject* args, PyObject* kwords) {
    PyObject* map = nullptr;
    static const char* kwlist[] = {"map", nullptr};
    if(!PyArg_ParseTupleAndKeywords(args, kwords, "O",
                                    const_cast<char**>(kwlist), &map))
        return nullptr;

    try {
        std::vector<float> image;
        const Map m = read_map(map, image);
        std::vector<float> def(image.size());
        {
            const NoGil nogil;
            defimage(m, image, def);
        }
        write(map, "data", def);
        Py_RETURN_NONE;
    } catch(...) {
        return fail();
    }
}

// Computes the transpose of comdat
static PyObject* doppler_datcom(PyObject*, PyObject* args, PyObject* kwords) {
    PyObject* data = nullptr;
    PyObject* map = nullptr;
    static const char* kwlist[] = {"data", "map", nullptr};
    if(!PyArg_ParseTupleAndKeywords(args, kwords, "OO",
                                    const_cast<char**>(kwlist), &data, &map))
        return nullptr;

    try {
        std::vector<float> image;
        const Map m = read_map(map, image);
        Pixels px;
        const std::vector<Spectra> sp = read_data(data, px);
        {
            const NoGil nogil;
            tr(m, image, sp, px.wave, px.flux);
        }
        write(map, "data", image);
        Py_RETURN_NONE;
    } catch(...) {
        return fail();
    }
}

// Carries out mem iterations on a map given data
static PyObject* doppler_memit(PyObject*, PyObject* args, PyObject* kwords) {
    PyObject* map = nullptr;
    PyObject* data = nullptr;
    Memopt o{0, 0.f, 1.e-4f, 0.2f};
    static const char* kwlist[] = {"map",  "data", "niter", "caim",
                                   "tlim", "rmax", nullptr};
    if(!PyArg_ParseTupleAndKeywords(args, kwords, "OOif|ff",
                                    const_cast<char**>(kwlist), &map, &data,
                                    &o.niter, &o.caim, &o.tlim, &o.rmax))
        return nullptr;

    if(o.niter < 1) {
        PyErr_SetString(PyExc_ValueError, "doppler.memit: niter < 1");
        return nullptr;
    }
    if(o.caim <= 0.) {
        PyErr_SetString(PyExc_ValueError, "doppler.memit: caim <= 0");
        return nullptr;
    }
    if(o.tlim >= 0.5) {
        PyErr_SetString(PyExc_ValueError, "doppler.memit: tlim >= 0.5");
        return nullptr;
    }
    if(o.rmax >= 0.5) {
        PyErr_SetString(PyExc_ValueError, "doppler.memit: rmax >= 0.5");
        return nullptr;
    }

    try {
        std::vector<float> image;
        const Map m = read_map(map, image);
        Pixels px;
        const std::vector<Spectra> sp = read_data(data, px);
        {
            const NoGil nogil;
            const std::lock_guard lock(memlock);
            mem(m, sp, px, image, o);
        }
        write(map, "data", image);
        Py_RETURN_NONE;
    } catch(...) {
        return fail();
    }
}

// Python's C API casts every method to PyCFunction
template <class F> static PyCFunction method(F f) {
    return reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(f));
}

// The methods. This and moduledef can't be const as Python's API takes
// them by non-const pointer.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef DopplerMethods[] = {

    {"comdat", method(doppler_comdat), METH_VARARGS | METH_KEYWORDS,
     "comdat(map, data)\n\n"
     "computes the data equivalent to 'map' using 'data' as the template\n"
     "On exit, 'data' is modified.\n\n"},

    {"comdef", method(doppler_comdef), METH_VARARGS | METH_KEYWORDS,
     "comdef(map)\n\n"
     "computes the default image to 'map'\n\n"},

    {"datcom", method(doppler_datcom), METH_VARARGS | METH_KEYWORDS,
     "datcom(data, map)\n\n"
     "computes the transpose operation to comdat for testing\n\n"},

    {"memit", method(doppler_memit), METH_VARARGS | METH_KEYWORDS,
     "memit(map, data, niter, caim, tlim, rmax)\n\n"
     "carries out niter mem iterations on map given data.\n\n"
     "Arguments::\n\n"
     "  map   : input Map, changed on ouput\n"
     "  data  : input Data\n"
     "  niter : number of iterations\n"
     "  caim  : reduced chi**2 to aim for\n"
     "  tlim  : limit on TEST below which iterations cease\n"
     "  rmax  : maximum change to allow\n\n"},

    {nullptr, nullptr, 0, nullptr} /* Sentinel */
};

// New python3 stuff
struct module_state {
    PyObject* error;
};

static int myextension_traverse(PyObject* m, visitproc visit, void* arg) {
    Py_VISIT(static_cast<module_state*>(PyModule_GetState(m))->error);
    return 0;
}

static int myextension_clear(PyObject* m) {
    Py_CLEAR(static_cast<module_state*>(PyModule_GetState(m))->error);
    return 0;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,       "_doppler",        nullptr,
    sizeof(struct module_state), DopplerMethods,    nullptr,
    myextension_traverse,        myextension_clear, nullptr};

PyMODINIT_FUNC PyInit__doppler(void) {
    // import_array needed to prevent segfaults with numpy
    // array
    import_array();
    return PyModule_Create(&moduledef);
}
