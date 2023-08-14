/*
This source is published under the following 3-clause BSD license.

Copyright (c) 2012 - 2013, Lukas Hosek and Alexander Wilkie
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * None of the names of the contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* ============================================================================

This file is part of a sample implementation of the analytical skylight and
solar radiance models presented in the SIGGRAPH 2012 paper


           "An Analytic Model for Full Spectral Sky-Dome Radiance"

and the 2013 IEEE CG&A paper

       "Adding a Solar Radiance Function to the Hosek Skylight Model"

                                   both by

                       Lukas Hosek and Alexander Wilkie
                Charles University in Prague, Czech Republic


                        Version: 1.4a, February 22nd, 2013

Version history:

1.4a  February 22nd, 2013
      Removed unnecessary and counter-intuitive solar radius parameters
      from the interface of the colourspace sky dome initialisation functions.

1.4   February 11th, 2013
      Fixed a bug which caused the relative brightness of the solar disc
      and the sky dome to be off by a factor of about 6. The sun was too
      bright: this affected both normal and alien sun scenarios. The
      coefficients of the solar radiance function were changed to fix this.

1.3   January 21st, 2013 (not released to the public)
      Added support for solar discs that are not exactly the same size as
      the terrestrial sun. Also added support for suns with a different
      emission spectrum ("Alien World" functionality).

1.2a  December 18th, 2012
      Fixed a mistake and some inaccuracies in the solar radiance function
      explanations found in ArHosekSkyModel.h. The actual source code is
      unchanged compared to version 1.2.

1.2   December 17th, 2012
      Native RGB data and a solar radiance function that matches the turbidity
      conditions were added.

1.1   September 2012
      The coefficients of the spectral model are now scaled so that the output
      is given in physical units: W / (m^-2 * sr * nm). Also, the output of the
      XYZ model is now no longer scaled to the range [0...1]. Instead, it is
      the result of a simple conversion from spectral data via the CIE 2 degree
      standard observer matching functions. Therefore, after multiplication
      with 683 lm / W, the Y channel now corresponds to luminance in lm.

1.0   May 11th, 2012
      Initial release.


Please visit http://cgg.mff.cuni.cz/projects/SkylightModelling/ to check if
an updated version of this code has been published!

============================================================================ */

/*

All instructions on how to use this code are in the accompanying header file.

*/

#pragma once

#ifndef __CUDACC__
#include <cmath>

#include "arhosek_rgb_data.h"
#include "cuda_util.h"
#endif

//   Some macro definitions that occur elsewhere in ART, and that have to be
//   replicated to make this a stand-alone module.
#ifndef NIL
#define NIL 0
#endif

#ifndef MATH_PI
#define MATH_PI 3.141592653589793f
#endif

#ifndef MATH_DEG_TO_RAD
#define MATH_DEG_TO_RAD (MATH_PI / 180.0f)
#endif

#ifndef MATH_RAD_TO_DEG
#define MATH_RAD_TO_DEG (180.0f / MATH_PI)
#endif

#ifndef DEGREES
#define DEGREES *MATH_DEG_TO_RAD
#endif

#ifndef TERRESTRIAL_SOLAR_RADIUS
#define TERRESTRIAL_SOLAR_RADIUS ((0.51f DEGREES) / 2.0f)
#endif

typedef float ArHosekSkyModelConfiguration[9];

typedef struct ArHosekSkyModelState
{
    ArHosekSkyModelConfiguration configs[11];
    float radiances[11];
    float turbidity;
    float solar_radius;
    float emission_correction_factor_sky[11];
    float emission_correction_factor_sun[11];
    float albedo;
    float elevation;
} ArHosekSkyModelState;

typedef float *ArHosekSkyModel_Dataset;
typedef float *ArHosekSkyModel_Radiance_Dataset;

#ifndef __CUDACC__

inline CUDA_HOST void ArHosekSkyModel_CookConfiguration(
    ArHosekSkyModel_Dataset dataset, ArHosekSkyModelConfiguration config,
    float turbidity, float albedo, float solar_elevation)
{
    float *elev_matrix;

    int int_turbidity = (int)turbidity;
    float turbidity_rem = turbidity - (float)int_turbidity;

    solar_elevation = powf(solar_elevation / (MATH_PI / 2.0f), (1.0f / 3.0f));

    // alb 0 low turb

    elev_matrix = dataset + (9 * 6 * (int_turbidity - 1));

    for (unsigned int i = 0; i < 9; ++i)
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 *
        // A4;
        config[i] = (1.0f - albedo) * (1.0f - turbidity_rem) *
                    (powf(1.0f - solar_elevation, 5.0f) * elev_matrix[i] +
                     5.0f * powf(1.0f - solar_elevation, 4.0f) *
                         solar_elevation * elev_matrix[i + 9] +
                     10.0f * powf(1.0f - solar_elevation, 3.0f) *
                         powf(solar_elevation, 2.0f) * elev_matrix[i + 18] +
                     10.0f * powf(1.0f - solar_elevation, 2.0f) *
                         powf(solar_elevation, 3.0f) * elev_matrix[i + 27] +
                     5.0f * (1.0f - solar_elevation) *
                         powf(solar_elevation, 4.0f) * elev_matrix[i + 36] +
                     powf(solar_elevation, 5.0f) * elev_matrix[i + 45]);
    }

    // alb 1 low turb
    elev_matrix = dataset + (9 * 6 * 10 + 9 * 6 * (int_turbidity - 1));
    for (unsigned int i = 0; i < 9; ++i)
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 *
        // A4;
        config[i] += (albedo) * (1.0f - turbidity_rem) *
                     (powf(1.0f - solar_elevation, 5.0f) * elev_matrix[i] +
                      5.0f * powf(1.0f - solar_elevation, 4.0f) *
                          solar_elevation * elev_matrix[i + 9] +
                      10.0f * powf(1.0f - solar_elevation, 3.0f) *
                          powf(solar_elevation, 2.0f) * elev_matrix[i + 18] +
                      10.0f * powf(1.0f - solar_elevation, 2.0f) *
                          powf(solar_elevation, 3.0f) * elev_matrix[i + 27] +
                      5.0f * (1.0f - solar_elevation) *
                          powf(solar_elevation, 4.0f) * elev_matrix[i + 36] +
                      powf(solar_elevation, 5.0f) * elev_matrix[i + 45]);
    }

    if (int_turbidity == 10) return;

    // alb 0 high turb
    elev_matrix = dataset + (9 * 6 * (int_turbidity));
    for (unsigned int i = 0; i < 9; ++i)
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 *
        // A4;
        config[i] += (1.0f - albedo) * (turbidity_rem) *
                     (powf(1.0f - solar_elevation, 5.0f) * elev_matrix[i] +
                      5.0f * powf(1.0f - solar_elevation, 4.0f) *
                          solar_elevation * elev_matrix[i + 9] +
                      10.0f * powf(1.0f - solar_elevation, 3.0f) *
                          powf(solar_elevation, 2.0f) * elev_matrix[i + 18] +
                      10.0f * powf(1.0 - solar_elevation, 2.0f) *
                          powf(solar_elevation, 3.0f) * elev_matrix[i + 27] +
                      5.0f * (1.0f - solar_elevation) *
                          powf(solar_elevation, 4.0f) * elev_matrix[i + 36] +
                      powf(solar_elevation, 5.0f) * elev_matrix[i + 45]);
    }

    // alb 1 high turb
    elev_matrix = dataset + (9 * 6 * 10 + 9 * 6 * (int_turbidity));
    for (unsigned int i = 0; i < 9; ++i)
    {
        //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 *
        // A4;
        config[i] += (albedo) * (turbidity_rem) *
                     (powf(1.0f - solar_elevation, 5.0f) * elev_matrix[i] +
                      5.0f * powf(1.0f - solar_elevation, 4.0f) *
                          solar_elevation * elev_matrix[i + 9] +
                      10.0f * powf(1.0f - solar_elevation, 3.0f) *
                          powf(solar_elevation, 2.0f) * elev_matrix[i + 18] +
                      10.0f * powf(1.0f - solar_elevation, 2.0f) *
                          powf(solar_elevation, 3.0f) * elev_matrix[i + 27] +
                      5.0f * (1.0f - solar_elevation) *
                          powf(solar_elevation, 4.0f) * elev_matrix[i + 36] +
                      powf(solar_elevation, 5.0f) * elev_matrix[i + 45]);
    }
}

inline CUDA_HOST float ArHosekSkyModel_CookRadianceConfiguration(
    ArHosekSkyModel_Radiance_Dataset dataset, float turbidity, float albedo,
    float solar_elevation)
{
    float *elev_matrix;

    int int_turbidity = (int)turbidity;
    float turbidity_rem = turbidity - (float)int_turbidity;
    float res;
    solar_elevation = powf(solar_elevation / (MATH_PI / 2.0f), (1.0f / 3.0f));

    // alb 0 low turb
    elev_matrix = dataset + (6 * (int_turbidity - 1));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res = (1.0f - albedo) * (1.0f - turbidity_rem) *
          (powf(1.0f - solar_elevation, 5.0f) * elev_matrix[0] +
           5.0f * powf(1.0f - solar_elevation, 4.0f) * solar_elevation *
               elev_matrix[1] +
           10.0f * powf(1.0f - solar_elevation, 3.0f) *
               powf(solar_elevation, 2.0f) * elev_matrix[2] +
           10.0f * powf(1.0f - solar_elevation, 2.0f) *
               powf(solar_elevation, 3.0f) * elev_matrix[3] +
           5.0f * (1.0f - solar_elevation) * powf(solar_elevation, 4.0f) *
               elev_matrix[4] +
           powf(solar_elevation, 5.0f) * elev_matrix[5]);

    // alb 1 low turb
    elev_matrix = dataset + (6 * 10 + 6 * (int_turbidity - 1));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res += (albedo) * (1.0f - turbidity_rem) *
           (powf(1.0f - solar_elevation, 5.0f) * elev_matrix[0] +
            5.0f * powf(1.0f - solar_elevation, 4.0f) * solar_elevation *
                elev_matrix[1] +
            10.0f * pow(1.0f - solar_elevation, 3.0f) *
                powf(solar_elevation, 2.0f) * elev_matrix[2] +
            10.0f * pow(1.0f - solar_elevation, 2.0f) *
                powf(solar_elevation, 3.0f) * elev_matrix[3] +
            5.0f * (1.0f - solar_elevation) * powf(solar_elevation, 4.0f) *
                elev_matrix[4] +
            powf(solar_elevation, 5.0f) * elev_matrix[5]);
    if (int_turbidity == 10) return res;

    // alb 0 high turb
    elev_matrix = dataset + (6 * (int_turbidity));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res += (1.0f - albedo) * (turbidity_rem) *
           (powf(1.0f - solar_elevation, 5.0f) * elev_matrix[0] +
            5.0f * powf(1.0f - solar_elevation, 4.0f) * solar_elevation *
                elev_matrix[1] +
            10.0f * powf(1.0f - solar_elevation, 3.0f) *
                powf(solar_elevation, 2.0f) * elev_matrix[2] +
            10.0f * powf(1.0f - solar_elevation, 2.0f) *
                powf(solar_elevation, 3.0f) * elev_matrix[3] +
            5.0f * (1.0f - solar_elevation) * powf(solar_elevation, 4.0f) *
                elev_matrix[4] +
            powf(solar_elevation, 5.0f) * elev_matrix[5]);

    // alb 1 high turb
    elev_matrix = dataset + (6 * 10 + 6 * (int_turbidity));
    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    res += (albedo) * (turbidity_rem) *
           (powf(1.0f - solar_elevation, 5.0f) * elev_matrix[0] +
            5.0f * powf(1.0f - solar_elevation, 4.0f) * solar_elevation *
                elev_matrix[1] +
            10.0f * powf(1.0f - solar_elevation, 3.0f) *
                powf(solar_elevation, 2.0f) * elev_matrix[2] +
            10.0f * powf(1.0f - solar_elevation, 2.0f) *
                powf(solar_elevation, 3.0f) * elev_matrix[3] +
            5.0f * (1.0f - solar_elevation) * powf(solar_elevation, 4.0f) *
                elev_matrix[4] +
            powf(solar_elevation, 5.0f) * elev_matrix[5]);
    return res;
}

inline CUDA_HOST ArHosekSkyModelState arhosek_rgb_skymodelstate_alloc_init(
    const float turbidity, const float albedo, const float elevation)
{
    ArHosekSkyModelState state;

    state.solar_radius = TERRESTRIAL_SOLAR_RADIUS;
    state.turbidity = turbidity;
    state.albedo = albedo;
    state.elevation = elevation;

    for (unsigned int channel = 0; channel < 3; ++channel)
    {
        ArHosekSkyModel_CookConfiguration(datasetsRGB[channel],
                                          state.configs[channel], turbidity,
                                          albedo, elevation);

        state.radiances[channel] = ArHosekSkyModel_CookRadianceConfiguration(
            datasetsRGBRad[channel], turbidity, albedo, elevation);
    }

    return state;
}

#endif