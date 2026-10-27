#ifndef TRM_DOPPLER_DOPP_H
#define TRM_DOPPLER_DOPP_H

#include <cstddef>
#include <span>
#include <vector>

// Speed of light in km/s
const double CKMS = 299792.458;

// Ratio FWHM/sigma for a gaussian
const double EFAC = 2.354820045;

// Default image options. Values match Default in map.py
enum class DefOpt { UNIFORM = 1, GAUSS2D = 2, GAUSS3D = 3 };

// Image types. Values match map.py. P = +, N = -; SINE etc modulate the
// image with orbital phase, the 2 versions at twice the orbital frequency.
enum class Itype {
    PUNIT = 1,
    NUNIT,
    PSINE,
    NSINE,
    PCOSINE,
    NCOSINE,
    PSINE2,
    NSINE2,
    PCOSINE2,
    NCOSINE2
};

// One image of a map. Its pixels live elsewhere (the memsys buffer in
// memit), stored x fastest, then y, then z.
struct Image {
    size_t nx, ny, nz;
    double vxy;                      // Vx-Vy pixel size, km/s
    double vz;                       // Vz spacing, km/s, only used if nz > 1
    std::vector<double> wave;        // lines modelled by the image
    std::vector<float> gamma, scale; // systemic velocity and scale per line
    Itype itype;
    DefOpt def;
    double bias;
    double fwhmxy, fwhmz;   // GAUSS defaults, km/s
    double squeeze, sqfwhm; // GAUSS3D defaults
    size_t npix() const { return nx * ny * nz; }
};

struct Map {
    std::vector<Image> images;
    double tzero, period, quad; // ephemeris
    double vfine;               // km/s of the fine projection array
    double sfac;                // global scaling factor
};

// One dataset: nspec spectra of nwave pixels. Fluxes and wavelengths are
// held elsewhere, concatenated over datasets.
struct Spectra {
    size_t nwave, nspec;
    std::vector<double> time; // mid-exposure, same units as tzero
    std::vector<float> expose;
    std::vector<int> nsub; // sub-divisions to model each exposure
    double fwhm;           // resolution, km/s
    size_t npix() const { return nwave * nspec; }
};

// image to data transform: projects the images into flux
void op(const Map& map, std::span<const float> image,
        const std::vector<Spectra>& data, std::span<const double> wave,
        std::span<float> flux);

// transpose of op: back-projects flux into the images
void tr(const Map& map, std::span<float> image,
        const std::vector<Spectra>& data, std::span<const double> wave,
        std::span<const float> flux);

// default images for MEM, one per image of the map
void defimage(const Map& map, std::span<const float> image,
              std::span<float> def);

#endif
