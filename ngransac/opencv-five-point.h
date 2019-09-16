/* This file contains functionality, adapted from OpenCV (calib3d/src/five-point.cpp) */

/*  Copyright (c) 2013, Bo Li (prclibo@gmail.com), ETH Zurich
    Copyright (c) 2019, Heidelberg University, all rights reserved.
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the copyright holder nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

void compute_essential_error(cv::InputArray _m1, cv::InputArray _m2, cv::InputArray _model, cv::OutputArray _err )
{
    cv::Mat X1 = _m1.getMat(), X2 = _m2.getMat(), model = _model.getMat();
    const cv::Point2d* x1ptr = X1.ptr<cv::Point2d>();
    const cv::Point2d* x2ptr = X2.ptr<cv::Point2d>();
    int n = X1.checkVector(2);
    cv::Matx33d E(model.ptr<double>());

    _err.create(n, 1, CV_32F);
    cv::Mat err = _err.getMat();

    for (int i = 0; i < n; i++)
    {
        cv::Vec3d x1(x1ptr[i].x, x1ptr[i].y, 1.);
        cv::Vec3d x2(x2ptr[i].x, x2ptr[i].y, 1.);
        cv::Vec3d Ex1 = E * x1;
        cv::Vec3d Etx2 = E.t() * x2;
        double x2tEx1 = x2.dot(Ex1);

        double a = Ex1[0] * Ex1[0];
        double b = Ex1[1] * Ex1[1];
        double c = Etx2[0] * Etx2[0];
        double d = Etx2[1] * Etx2[1];

        err.at<float>(i) = (float)(x2tEx1 * x2tEx1 / (a + b + c + d));
    }
}