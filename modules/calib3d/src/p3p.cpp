#include <cstring>
#include <cmath>
#include <iostream>

#include "polynom_solver.h"
#include "p3p.h"

void p3p_old::init_inverse_parameters()
{
    inv_fx = 1. / fx;
    inv_fy = 1. / fy;
    cx_fx = cx / fx;
    cy_fy = cy / fy;
}

p3p_old::p3p_old(cv::Mat cameraMatrix)
{
    if (cameraMatrix.depth() == CV_32F)
        init_camera_parameters<float>(cameraMatrix);
    else
        init_camera_parameters<double>(cameraMatrix);
    init_inverse_parameters();
}

p3p_old::p3p_old(double _fx, double _fy, double _cx, double _cy)
{
    fx = _fx;
    fy = _fy;
    cx = _cx;
    cy = _cy;
    init_inverse_parameters();
}

bool p3p_old::solve(cv::Mat& R, cv::Mat& tvec, const cv::Mat& opoints, const cv::Mat& ipoints)
{
    CV_INSTRUMENT_REGION();

    double rotation_matrix[3][3] = {}, translation[3] = {};
    std::vector<double> points;
    if (opoints.depth() == ipoints.depth())
    {
        if (opoints.depth() == CV_32F)
            extract_points<cv::Point3f,cv::Point2f>(opoints, ipoints, points);
        else
            extract_points<cv::Point3d,cv::Point2d>(opoints, ipoints, points);
    }
    else if (opoints.depth() == CV_32F)
        extract_points<cv::Point3f,cv::Point2d>(opoints, ipoints, points);
    else
        extract_points<cv::Point3d,cv::Point2f>(opoints, ipoints, points);

    bool result = solve(rotation_matrix, translation,
                        points[0], points[1], points[2], points[3], points[4],
                        points[5], points[6], points[7], points[8], points[9],
                        points[10], points[11], points[12], points[13], points[14],
                        points[15], points[16], points[17], points[18], points[19]);
    cv::Mat(3, 1, CV_64F, translation).copyTo(tvec);
    cv::Mat(3, 3, CV_64F, rotation_matrix).copyTo(R);
    return result;
}

int p3p_old::solve(std::vector<cv::Mat>& Rs, std::vector<cv::Mat>& tvecs, const cv::Mat& opoints, const cv::Mat& ipoints)
{
    CV_INSTRUMENT_REGION();

    double rotation_matrix[4][3][3] = {}, translation[4][3] = {};
    std::vector<double> points;
    if (opoints.depth() == ipoints.depth())
    {
        if (opoints.depth() == CV_32F)
            extract_points<cv::Point3f,cv::Point2f>(opoints, ipoints, points);
        else
            extract_points<cv::Point3d,cv::Point2d>(opoints, ipoints, points);
    }
    else if (opoints.depth() == CV_32F)
        extract_points<cv::Point3f,cv::Point2d>(opoints, ipoints, points);
    else
        extract_points<cv::Point3d,cv::Point2f>(opoints, ipoints, points);

    const bool p4p = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F)) == 4;
    int solutions = solve(rotation_matrix, translation,
                          points[0], points[1], points[2], points[3], points[4],
                          points[5], points[6], points[7], points[8], points[9],
                          points[10], points[11], points[12], points[13], points[14],
                          points[15], points[16], points[17], points[18], points[19],
                          p4p);

    for (int i = 0; i < solutions; i++) {
        cv::Mat R, tvec;
        cv::Mat(3, 1, CV_64F, translation[i]).copyTo(tvec);
        cv::Mat(3, 3, CV_64F, rotation_matrix[i]).copyTo(R);

        Rs.push_back(R);
        tvecs.push_back(tvec);
    }

    return solutions;
}

bool p3p_old::solve(double R[3][3], double t[3],
    double mu0, double mv0,   double X0, double Y0, double Z0,
    double mu1, double mv1,   double X1, double Y1, double Z1,
    double mu2, double mv2,   double X2, double Y2, double Z2,
    double mu3, double mv3,   double X3, double Y3, double Z3)
{
    double Rs[4][3][3] = {}, ts[4][3] = {};

    const bool p4p = true;
    int n = solve(Rs, ts, mu0, mv0, X0, Y0, Z0,  mu1, mv1, X1, Y1, Z1, mu2, mv2, X2, Y2, Z2, mu3, mv3, X3, Y3, Z3, p4p);

    if (n == 0)
        return false;

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++)
            R[i][j] = Rs[0][i][j];
        t[i] = ts[0][i];
    }

    return true;
}

int p3p_old::solve(double R[4][3][3], double t[4][3],
               double mu0, double mv0,   double X0, double Y0, double Z0,
               double mu1, double mv1,   double X1, double Y1, double Z1,
               double mu2, double mv2,   double X2, double Y2, double Z2,
               double mu3, double mv3,   double X3, double Y3, double Z3,
               bool p4p)
{
    double mk0, mk1, mk2;
    double norm;

    mu0 = inv_fx * mu0 - cx_fx;
    mv0 = inv_fy * mv0 - cy_fy;
    norm = sqrt(mu0 * mu0 + mv0 * mv0 + 1);
    mk0 = 1. / norm; mu0 *= mk0; mv0 *= mk0;

    mu1 = inv_fx * mu1 - cx_fx;
    mv1 = inv_fy * mv1 - cy_fy;
    norm = sqrt(mu1 * mu1 + mv1 * mv1 + 1);
    mk1 = 1. / norm; mu1 *= mk1; mv1 *= mk1;

    mu2 = inv_fx * mu2 - cx_fx;
    mv2 = inv_fy * mv2 - cy_fy;
    norm = sqrt(mu2 * mu2 + mv2 * mv2 + 1);
    mk2 = 1. / norm; mu2 *= mk2; mv2 *= mk2;

    mu3 = inv_fx * mu3 - cx_fx;
    mv3 = inv_fy * mv3 - cy_fy;

    double distances[3];
    distances[0] = sqrt( (X1 - X2) * (X1 - X2) + (Y1 - Y2) * (Y1 - Y2) + (Z1 - Z2) * (Z1 - Z2) );
    distances[1] = sqrt( (X0 - X2) * (X0 - X2) + (Y0 - Y2) * (Y0 - Y2) + (Z0 - Z2) * (Z0 - Z2) );
    distances[2] = sqrt( (X0 - X1) * (X0 - X1) + (Y0 - Y1) * (Y0 - Y1) + (Z0 - Z1) * (Z0 - Z1) );

    // Calculate angles
    double cosines[3];
    cosines[0] = mu1 * mu2 + mv1 * mv2 + mk1 * mk2;
    cosines[1] = mu0 * mu2 + mv0 * mv2 + mk0 * mk2;
    cosines[2] = mu0 * mu1 + mv0 * mv1 + mk0 * mk1;

    double lengths[4][3] = {};
    int n = solve_for_lengths(lengths, distances, cosines);

    int nb_solutions = 0;
    double reproj_errors[4];
    for(int i = 0; i < n; i++) {
        double M_orig[3][3];

        M_orig[0][0] = lengths[i][0] * mu0;
        M_orig[0][1] = lengths[i][0] * mv0;
        M_orig[0][2] = lengths[i][0] * mk0;

        M_orig[1][0] = lengths[i][1] * mu1;
        M_orig[1][1] = lengths[i][1] * mv1;
        M_orig[1][2] = lengths[i][1] * mk1;

        M_orig[2][0] = lengths[i][2] * mu2;
        M_orig[2][1] = lengths[i][2] * mv2;
        M_orig[2][2] = lengths[i][2] * mk2;

        if (!align(M_orig, X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2, R[nb_solutions], t[nb_solutions]))
            continue;

        if (p4p) {
            double X3p = R[nb_solutions][0][0] * X3 + R[nb_solutions][0][1] * Y3 + R[nb_solutions][0][2] * Z3 + t[nb_solutions][0];
            double Y3p = R[nb_solutions][1][0] * X3 + R[nb_solutions][1][1] * Y3 + R[nb_solutions][1][2] * Z3 + t[nb_solutions][1];
            double Z3p = R[nb_solutions][2][0] * X3 + R[nb_solutions][2][1] * Y3 + R[nb_solutions][2][2] * Z3 + t[nb_solutions][2];
            double mu3p = X3p / Z3p;
            double mv3p = Y3p / Z3p;
            reproj_errors[nb_solutions] = (mu3p - mu3) * (mu3p - mu3) + (mv3p - mv3) * (mv3p - mv3);
        }

        nb_solutions++;
    }

    if (p4p) {
        //sort the solutions
        for (int i = 1; i < nb_solutions; i++) {
            for (int j = i; j > 0 && reproj_errors[j-1] > reproj_errors[j]; j--) {
                std::swap(reproj_errors[j], reproj_errors[j-1]);
                std::swap(R[j], R[j-1]);
                std::swap(t[j], t[j-1]);
            }
        }
    }

    return nb_solutions;
}

/// Given 3D distances between three points and cosines of 3 angles at the apex, calculates
/// the lengths of the line segments connecting projection center (P) and the three 3D points (A, B, C).
/// Returned distances are for |PA|, |PB|, |PC| respectively.
/// Only the solution to the main branch.
/// Reference : X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang; "Complete Solution Classification for the Perspective-Three-Point Problem"
/// IEEE Trans. on PAMI, vol. 25, No. 8, August 2003
/// \param lengths Lengths of line segments up to four solutions.
/// \param distances Distance between 3D points in pairs |BC|, |AC|, |AB|.
/// \param cosines Cosine of the angles /_BPC, /_APC, /_APB.
/// \returns Number of solutions.
/// WARNING: NOT ALL THE DEGENERATE CASES ARE IMPLEMENTED

int p3p_old::solve_for_lengths(double lengths[4][3], double distances[3], double cosines[3])
{
    double p = cosines[0] * 2;
    double q = cosines[1] * 2;
    double r = cosines[2] * 2;

    double inv_d22 = 1. / (distances[2] * distances[2]);
    double a = inv_d22 * (distances[0] * distances[0]);
    double b = inv_d22 * (distances[1] * distances[1]);

    double a2 = a * a, b2 = b * b, p2 = p * p, q2 = q * q, r2 = r * r;
    double pr = p * r, pqr = q * pr;

    // Check reality condition (the four points should not be coplanar)
    if (p2 + q2 + r2 - pqr - 1 == 0)
        return 0;

    double ab = a * b, a_2 = 2*a;

    double A = -2 * b + b2 + a2 + 1 + ab*(2 - r2) - a_2;

    // Check reality condition
    if (A == 0) return 0;

    double a_4 = 4*a;

    double B = q*(-2*(ab + a2 + 1 - b) + r2*ab + a_4) + pr*(b - b2 + ab);
    double C = q2 + b2*(r2 + p2 - 2) - b*(p2 + pqr) - ab*(r2 + pqr) + (a2 - a_2)*(2 + q2) + 2;
    double D = pr*(ab-b2+b) + q*((p2-2)*b + 2 * (ab - a2) + a_4 - 2);
    double E = 1 + 2*(b - a - ab) + b2 - b*p2 + a2;

    double temp = (p2*(a-1+b) + r2*(a-1-b) + pqr - a*pqr);
    double b0 = b * temp * temp;
    // Check reality condition
    if (b0 == 0)
        return 0;

    double real_roots[4];
    int n = solve_deg4(A, B, C, D, E,  real_roots[0], real_roots[1], real_roots[2], real_roots[3]);

    if (n == 0)
        return 0;

    int nb_solutions = 0;
    double r3 = r2*r, pr2 = p*r2, r3q = r3 * q;
    double inv_b0 = 1. / b0;

    // For each solution of x
    for(int i = 0; i < n; i++) {
        double x = real_roots[i];

        // Check reality condition
        if (x <= 0)
            continue;

        double x2 = x*x;

        double b1 =
            ((1-a-b)*x2 + (q*a-q)*x + 1 - a + b) *
            (((r3*(a2 + ab*(2 - r2) - a_2 + b2 - 2*b + 1)) * x +

            (r3q*(2*(b-a2) + a_4 + ab*(r2 - 2) - 2) + pr2*(1 + a2 + 2*(ab-a-b) + r2*(b - b2) + b2))) * x2 +

            (r3*(q2*(1-2*a+a2) + r2*(b2-ab) - a_4 + 2*(a2 - b2) + 2) + r*p2*(b2 + 2*(ab - b - a) + 1 + a2) + pr2*q*(a_4 + 2*(b - ab - a2) - 2 - r2*b)) * x +

            2*r3q*(a_2 - b - a2 + ab - 1) + pr2*(q2 - a_4 + 2*(a2 - b2) + r2*b + q2*(a2 - a_2) + 2) +
            p2*(p*(2*(ab - a - b) + a2 + b2 + 1) + 2*q*r*(b + a_2 - a2 - ab - 1)));

        // Check reality condition
        if (b1 <= 0)
            continue;

        double y = inv_b0 * b1;
        double v = x2 + y*y - x*y*r;

        if (v <= 0)
            continue;

        double Z = distances[2] / sqrt(v);
        double X = x * Z;
        double Y = y * Z;

        lengths[nb_solutions][0] = X;
        lengths[nb_solutions][1] = Y;
        lengths[nb_solutions][2] = Z;

        nb_solutions++;
    }

    return nb_solutions;
}

bool p3p_old::align(double M_end[3][3],
    double X0, double Y0, double Z0,
    double X1, double Y1, double Z1,
    double X2, double Y2, double Z2,
    double R[3][3], double T[3])
{
    // Centroids:
    double C_start[3] = {}, C_end[3] = {};
    for(int i = 0; i < 3; i++) C_end[i] = (M_end[0][i] + M_end[1][i] + M_end[2][i]) / 3;
    C_start[0] = (X0 + X1 + X2) / 3;
    C_start[1] = (Y0 + Y1 + Y2) / 3;
    C_start[2] = (Z0 + Z1 + Z2) / 3;

    // Covariance matrix s:
    double s[3 * 3] = {};
    for(int j = 0; j < 3; j++) {
        s[0 * 3 + j] = (X0 * M_end[0][j] + X1 * M_end[1][j] + X2 * M_end[2][j]) / 3 - C_end[j] * C_start[0];
        s[1 * 3 + j] = (Y0 * M_end[0][j] + Y1 * M_end[1][j] + Y2 * M_end[2][j]) / 3 - C_end[j] * C_start[1];
        s[2 * 3 + j] = (Z0 * M_end[0][j] + Z1 * M_end[1][j] + Z2 * M_end[2][j]) / 3 - C_end[j] * C_start[2];
    }

    double Qs[16] = {}, evs[4] = {}, U[16] = {};

    Qs[0 * 4 + 0] = s[0 * 3 + 0] + s[1 * 3 + 1] + s[2 * 3 + 2];
    Qs[1 * 4 + 1] = s[0 * 3 + 0] - s[1 * 3 + 1] - s[2 * 3 + 2];
    Qs[2 * 4 + 2] = s[1 * 3 + 1] - s[2 * 3 + 2] - s[0 * 3 + 0];
    Qs[3 * 4 + 3] = s[2 * 3 + 2] - s[0 * 3 + 0] - s[1 * 3 + 1];

    Qs[1 * 4 + 0] = Qs[0 * 4 + 1] = s[1 * 3 + 2] - s[2 * 3 + 1];
    Qs[2 * 4 + 0] = Qs[0 * 4 + 2] = s[2 * 3 + 0] - s[0 * 3 + 2];
    Qs[3 * 4 + 0] = Qs[0 * 4 + 3] = s[0 * 3 + 1] - s[1 * 3 + 0];
    Qs[2 * 4 + 1] = Qs[1 * 4 + 2] = s[1 * 3 + 0] + s[0 * 3 + 1];
    Qs[3 * 4 + 1] = Qs[1 * 4 + 3] = s[2 * 3 + 0] + s[0 * 3 + 2];
    Qs[3 * 4 + 2] = Qs[2 * 4 + 3] = s[2 * 3 + 1] + s[1 * 3 + 2];

    jacobi_4x4(Qs, evs, U);

    // Looking for the largest eigen value:
    int i_ev = 0;
    double ev_max = evs[i_ev];
    for(int i = 1; i < 4; i++)
        if (evs[i] > ev_max)
            ev_max = evs[i_ev = i];

    // Quaternion:
    double q[4];
    for(int i = 0; i < 4; i++)
        q[i] = U[i * 4 + i_ev];

    double q02 = q[0] * q[0], q12 = q[1] * q[1], q22 = q[2] * q[2], q32 = q[3] * q[3];
    double q0_1 = q[0] * q[1], q0_2 = q[0] * q[2], q0_3 = q[0] * q[3];
    double q1_2 = q[1] * q[2], q1_3 = q[1] * q[3];
    double q2_3 = q[2] * q[3];

    R[0][0] = q02 + q12 - q22 - q32;
    R[0][1] = 2. * (q1_2 - q0_3);
    R[0][2] = 2. * (q1_3 + q0_2);

    R[1][0] = 2. * (q1_2 + q0_3);
    R[1][1] = q02 + q22 - q12 - q32;
    R[1][2] = 2. * (q2_3 - q0_1);

    R[2][0] = 2. * (q1_3 - q0_2);
    R[2][1] = 2. * (q2_3 + q0_1);
    R[2][2] = q02 + q32 - q12 - q22;

    for(int i = 0; i < 3; i++)
        T[i] = C_end[i] - (R[i][0] * C_start[0] + R[i][1] * C_start[1] + R[i][2] * C_start[2]);

    return true;
}

bool p3p_old::jacobi_4x4(double * A, double * D, double * U)
{
    double B[4] = {}, Z[4] = {};
    double Id[16] = {1., 0., 0., 0.,
        0., 1., 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.};

    memcpy(U, Id, 16 * sizeof(double));

    B[0] = A[0]; B[1] = A[5]; B[2] = A[10]; B[3] = A[15];
    memcpy(D, B, 4 * sizeof(double));

    for(int iter = 0; iter < 50; iter++) {
        double sum = fabs(A[1]) + fabs(A[2]) + fabs(A[3]) + fabs(A[6]) + fabs(A[7]) + fabs(A[11]);

        if (sum == 0.0)
            return true;

        double tresh =  (iter < 3) ? 0.2 * sum / 16. : 0.0;
        for(int i = 0; i < 3; i++) {
            double * pAij = A + 5 * i + 1;
            for(int j = i + 1 ; j < 4; j++) {
                double Aij = *pAij;
                double eps_machine = 100.0 * fabs(Aij);

                if ( iter > 3 && fabs(D[i]) + eps_machine == fabs(D[i]) && fabs(D[j]) + eps_machine == fabs(D[j]) )
                    *pAij = 0.0;
                else if (fabs(Aij) > tresh) {
                    double hh = D[j] - D[i], t;
                    if (fabs(hh) + eps_machine == fabs(hh))
                        t = Aij / hh;
                    else {
                        double theta = 0.5 * hh / Aij;
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
                        if (theta < 0.0) t = -t;
                    }

                    hh = t * Aij;
                    Z[i] -= hh;
                    Z[j] += hh;
                    D[i] -= hh;
                    D[j] += hh;
                    *pAij = 0.0;

                    double c = 1.0 / sqrt(1 + t * t);
                    double s = t * c;
                    double tau = s / (1.0 + c);
                    for(int k = 0; k <= i - 1; k++) {
                        double g = A[k * 4 + i], h = A[k * 4 + j];
                        A[k * 4 + i] = g - s * (h + g * tau);
                        A[k * 4 + j] = h + s * (g - h * tau);
                    }
                    for(int k = i + 1; k <= j - 1; k++) {
                        double g = A[i * 4 + k], h = A[k * 4 + j];
                        A[i * 4 + k] = g - s * (h + g * tau);
                        A[k * 4 + j] = h + s * (g - h * tau);
                    }
                    for(int k = j + 1; k < 4; k++) {
                        double g = A[i * 4 + k], h = A[j * 4 + k];
                        A[i * 4 + k] = g - s * (h + g * tau);
                        A[j * 4 + k] = h + s * (g - h * tau);
                    }
                    for(int k = 0; k < 4; k++) {
                        double g = U[k * 4 + i], h = U[k * 4 + j];
                        U[k * 4 + i] = g - s * (h + g * tau);
                        U[k * 4 + j] = h + s * (g - h * tau);
                    }
                }
                pAij++;
            }
        }

        for(int i = 0; i < 4; i++) B[i] += Z[i];
        memcpy(D, B, 4 * sizeof(double));
        memset(Z, 0, 4 * sizeof(double));
    }

    return false;
}





using namespace cv;

namespace
{
Matx33d rotVec2RotMat(const Vec3d &v) {
    const double phi = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    const double x = v[0] / phi, y = v[1] / phi, z = v[2] / phi;
    const double a = std::sin(phi), b = std::cos(phi);
    // R = I + sin(phi) * skew(v) + (1 - cos(phi) * skew(v)^2
    return {(b - 1)*y*y + (b - 1)*z*z + 1, -a*z - x*y*(b - 1), a*y - x*z*(b - 1),
        a*z - x*y*(b - 1), (b - 1)*x*x + (b - 1)*z*z + 1, -a*x - y*z*(b - 1),
        -a*y - x*z*(b - 1), a*x - y*z*(b - 1), (b - 1)*x*x + (b - 1)*y*y + 1};
}

Vec3d rotMat2RotVec(const Matx33d &R) {
    // https://math.stackexchange.com/questions/83874/efficient-and-accurate-numerical-implementation-of-the-inverse-rodrigues-rotatio?rq=1
    Vec3d rot_vec;
    const double trace = R(0,0)+R(1,1)+R(2,2);
    if (trace >= 3 - FLT_EPSILON) {
        rot_vec = (0.5 * (trace-3)/12)*Vec3d(R(2,1)-R(1,2),
                                             R(0,2)-R(2,0),
                                             R(1,0)-R(0,1));
    } else if (3 - FLT_EPSILON > trace && trace > -1 + FLT_EPSILON) {
        double theta = std::acos((trace - 1) / 2);
        rot_vec = (theta / (2 * std::sin(theta))) * Vec3d(R(2,1)-R(1,2),
                                                     R(0,2)-R(2,0),
                                                     R(1,0)-R(0,1));
    } else {
        int a;
        if (R(0,0) > R(1,1))
            a = R(0,0) > R(2,2) ? 0 : 2;
        else
            a = R(1,1) > R(2,2) ? 1 : 2;
        Vec3d v;
        int b = (a + 1) % 3, c = (a + 2) % 3;
        double s = sqrt(R(a,a) - R(b,b) - R(c,c) + 1);
        v[a] = s / 2;
        v[b] = (R(b,a) + R(a,b)) / (2 * s);
        v[c] = (R(c,a) + R(a,c)) / (2 * s);
        rot_vec = M_PI * v / norm(v);
    }
    return rot_vec;
}
}

void p3p::calibrateAndNormalizePointsPnP(const Mat &opoints_, const Mat &ipoints_) {
    Mat ipoints = ipoints_.clone(), opoints = opoints_.clone();

    auto convertPoints = [] (Mat &points, int pt_dim) {
        points.convertTo(points, CV_64F); // convert points to have float precision
        if (points.channels() > 1)
            points = points.reshape(1, (int)points.total()); // convert point to have 1 channel
        if (points.rows < points.cols)
            transpose(points, points); // transpose so points will be in rows
        CV_CheckGE(points.cols, pt_dim, "Invalid dimension of point");
        if (points.cols != pt_dim) // in case when image points are 3D convert them to 2D
            points = points.colRange(0, pt_dim);
    };

    convertPoints(ipoints, 2);
    convertPoints(opoints, 3);

    points_mat(0, 0) = opoints.at<double>(0, 0);
    points_mat(0, 1) = opoints.at<double>(0, 1);
    points_mat(0, 2) = opoints.at<double>(0, 2);

    points_mat(1, 0) = opoints.at<double>(1, 0);
    points_mat(1, 1) = opoints.at<double>(1, 1);
    points_mat(1, 2) = opoints.at<double>(1, 2);

    points_mat(2, 0) = opoints.at<double>(2, 0);
    points_mat(2, 1) = opoints.at<double>(2, 1);
    points_mat(2, 2) = opoints.at<double>(2, 2);

    for (int i = 0; i < ipoints.rows; i++) {
        const double k_inv_u = ipoints.at<double>(i, 0);
        const double k_inv_v = ipoints.at<double>(i, 1);
        const double norm = 1.f / sqrtf(k_inv_u*k_inv_u + k_inv_v*k_inv_v + 1);
        calib_norm_points_mat(i, 0) = k_inv_u * norm;
        calib_norm_points_mat(i, 1) = k_inv_v * norm;
        calib_norm_points_mat(i, 2) =           norm;
    }
}

p3p::p3p() :
    points_mat(), calib_norm_points_mat()
{
}

int p3p::estimate(std::vector<Mat>& Rs, std::vector<Mat>& ts, const cv::Mat &opoints, const cv::Mat &ipoints) {
   /*
    * The description of this solution can be found here:
    * http://cmp.felk.cvut.cz/~pajdla/gvg/GVG-2016-Lecture.pdf
    * pages: 51-59
    */
    CV_INSTRUMENT_REGION();
    calibrateAndNormalizePointsPnP(opoints, ipoints);

    // std::cout << "ipoints:\n" << ipoints << std::endl;

    const Vec3d X1 (points_mat(0, 0), points_mat(0, 1), points_mat(0, 2));
    const Vec3d X2 (points_mat(1, 0), points_mat(1, 1), points_mat(1, 2));
    const Vec3d X3 (points_mat(2, 0), points_mat(2, 1), points_mat(2, 2));

    // find distance between world points d_ij = ||Xi - Xj||
    const double d12 = norm(X1 - X2);
    const double d23 = norm(X2 - X3);
    const double d31 = norm(X3 - X1);

    const double VAL_THR = 1e-4;
    if (d12 < VAL_THR || d23 < VAL_THR || d31 < VAL_THR)
        return 0;

    const Vec3d cx1 (calib_norm_points_mat(0, 0), calib_norm_points_mat(0, 1), calib_norm_points_mat(0, 2));
    const Vec3d cx2 (calib_norm_points_mat(1, 0), calib_norm_points_mat(1, 1), calib_norm_points_mat(1, 2));
    const Vec3d cx3 (calib_norm_points_mat(2, 0), calib_norm_points_mat(2, 1), calib_norm_points_mat(2, 2));

    // find cosine angles, cos(x1,x2) = K^-1 x1.dot(K^-1 x2) / (||K^-1 x1|| * ||K^-1 x2||)
    // calib_norm_points are already K^-1 x / ||K^-1 x||, so we perform only dot product
    const double c12 = cx1(0)*cx2(0) + cx1(1)*cx2(1) + cx1(2)*cx2(2);
    const double c23 = cx2(0)*cx3(0) + cx2(1)*cx3(1) + cx2(2)*cx3(2);
    const double c31 = cx3(0)*cx1(0) + cx3(1)*cx1(1) + cx3(2)*cx1(2);

    Matx33d Z, Zw;
    auto * z = Z.val, * zw = Zw.val;

    // find coefficients of polynomial a4 x^4 + ... + a0 = 0
    const double c12_p2 = c12*c12, c23_p2 = c23*c23, c31_p2 = c31*c31;
    const double d12_p2 = d12*d12, d12_p4 = d12_p2*d12_p2;
    const double d23_p2 = d23*d23, d23_p4 = d23_p2*d23_p2, d23_p6 = d23_p2*d23_p4, d23_p8 = d23_p4*d23_p4;
    const double d31_p2 = d31*d31, d31_p4 = d31_p2*d31_p2;
    const double a4 = -4*d23_p4*d12_p2*d31_p2*c23_p2+d23_p8-2*d23_p6*d12_p2-2*d23_p6*d31_p2+d23_p4*d12_p4+2*d23_p4*d12_p2*d31_p2+d23_p4*d31_p4;
    const double a3 = 8*d23_p4*d12_p2*d31_p2*c12*c23_p2+4*d23_p6*d12_p2*c31*c23-4*d23_p4*d12_p4*c31*c23+4*d23_p4*d12_p2*d31_p2*c31*c23-4*d23_p8*c12+4*d23_p6*d12_p2*c12+8*d23_p6*d31_p2*c12-4*d23_p4*d12_p2*d31_p2*c12-4*d23_p4*d31_p4*c12;
    const double a2 = -8*d23_p6*d12_p2*c31*c12*c23-8*d23_p4*d12_p2*d31_p2*c31*c12*c23+4*d23_p8*c12_p2-4*d23_p6*d12_p2*c31_p2-8*d23_p6*d31_p2*c12_p2+4*d23_p4*d12_p4*c31_p2+4*d23_p4*d12_p4*c23_p2-4*d23_p4*d12_p2*d31_p2*c23_p2+4*d23_p4*d31_p4*c12_p2+2*d23_p8-4*d23_p6*d31_p2-2*d23_p4*d12_p4+2*d23_p4*d31_p4;
    const double a1 = 8*d23_p6*d12_p2*c31_p2*c12+4*d23_p6*d12_p2*c31*c23-4*d23_p4*d12_p4*c31*c23+4*d23_p4*d12_p2*d31_p2*c31*c23-4*d23_p8*c12-4*d23_p6*d12_p2*c12+8*d23_p6*d31_p2*c12+4*d23_p4*d12_p2*d31_p2*c12-4*d23_p4*d31_p4*c12;
    const double a0 = -4*d23_p6*d12_p2*c31_p2+d23_p8-2*d23_p4*d12_p2*d31_p2+2*d23_p6*d12_p2+d23_p4*d31_p4+d23_p4*d12_p4-2*d23_p6*d31_p2;

    double roots[4] = {0};
    int num_roots = solve_deg4(a4, a3, a2, a1, a0, roots[0], roots[1], roots[2], roots[3]);
    // std::cout << "num_roots=" << num_roots << std::endl;

    Rs.reserve(num_roots);
    ts.reserve(num_roots);
    int nb_solutions = 0;
    for (double root : roots) {
        if (root <= 0) continue;

        const double n12 = root, n12_p2 = n12 * n12;
        const double n13 = (d12_p2*(d23_p2-d31_p2*n12_p2)+(d23_p2-d31_p2)*(d23_p2*(1+n12_p2-2*n12*c12)-d12_p2*n12_p2))
                            / (2*d12_p2*(d23_p2*c31 - d31_p2*c23*n12) + 2*(d31_p2-d23_p2)*d12_p2*c23*n12);
        const double n1 = d12 / sqrt(1 + n12_p2 - 2*n12*c12); // 1+n12^2-2n12c12 is always > 0
        const double n2 = n1 * n12;
        const double n3 = n1 * n13;

        if (n1 <= 0 || n2 <= 0 || n3 <= 0)
            continue;
        // compute and check errors
        if (fabs((sqrt(n1*n1 + n2*n2 - 2*n1*n2*c12) - d12) / d12) > VAL_THR ||
            fabs((sqrt(n2*n2 + n3*n3 - 2*n2*n3*c23) - d23) / d23) > VAL_THR ||
            fabs((sqrt(n3*n3 + n1*n1 - 2*n3*n1*c31) - d31) / d31) > VAL_THR)
            continue;

        const Vec3d nX1 = n1 * cx1;
        Vec3d Z2 = n2 * cx2 - nX1; Z2 /= norm(Z2);
        Vec3d Z3 = n3 * cx3 - nX1; Z3 /= norm(Z3);
        Vec3d Z1 = Z2.cross(Z3);   Z1 /= norm(Z1);
        const Vec3d Z3crZ1 = Z3.cross(Z1);

        z[0] = Z1(0);     z[3] = Z1(1);     z[6] = Z1(2);
        z[1] = Z2(0);     z[4] = Z2(1);     z[7] = Z2(2);
        z[2] = Z3crZ1(0); z[5] = Z3crZ1(1); z[8] = Z3crZ1(2);

        Vec3d Zw2 = (X2 - X1) / d12;
        Vec3d Zw3 = (X3 - X1) / d31;
        Vec3d Zw1 = Zw2.cross(Zw3); Zw1 /= norm(Zw1);
        const Vec3d Z3crZ1w = Zw3.cross(Zw1);

        zw[0] = Zw1(0);     zw[3] = Zw1(1);     zw[6] = Zw1(2);
        zw[1] = Zw2(0);     zw[4] = Zw2(1);     zw[7] = Zw2(2);
        zw[2] = Z3crZ1w(0); zw[5] = Z3crZ1w(1); zw[8] = Z3crZ1w(2);

        const Matx33d R = rotVec2RotMat(rotMat2RotVec(Z * Zw.inv()));
        Rs.push_back(cv::Mat(R));
        ts.push_back(cv::Mat(-R * (X1 - R.t() * nX1)));
        // std::cout << "ts: " << cv::Mat(-R * (X1 - R.t() * nX1)).t() << std::endl;
        nb_solutions++;
    }

    return nb_solutions;
}
