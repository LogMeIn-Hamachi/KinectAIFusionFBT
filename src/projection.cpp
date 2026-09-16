#include "core.hpp"
namespace kf {
V2 ColorProjection::project(V3 v) const {
    double denominator = p[8] * v.x + p[9] * v.y + v.z + p[10];
    if (std::abs(denominator) < 1e-6)
        return {-1e6, -1e6};
    return {(p[0] * v.x + p[1] * v.y + p[2] * v.z + p[3]) / denominator * width,
            (p[4] * v.x + p[5] * v.y + p[6] * v.z + p[7]) / denominator * height};
}
V3 ColorProjection::correction(V3 point, V2 target) const {
    if (!valid)
        return {};
    V2 current = project(point);
    double residual[2]{target.x - current.x, target.y - current.y};
    double error = std::hypot(residual[0], residual[1]);
    if (!std::isfinite(error) || error > 100)
        return {};
    double j[2][3];
    for (int axis = 0; axis < 3; ++axis) {
        V3 d{};
        if (axis == 0)
            d.x = .001;
        else if (axis == 1)
            d.y = .001;
        else
            d.z = .001;
        auto next = project(point + d);
        j[0][axis] = (next.x - current.x) / .001;
        j[1][axis] = (next.y - current.y) / .001;
    }
    // Minimum-norm 2D reprojection update; no spurious metric-depth observation.
    double a = 4, b = 0, c = 4;
    for (int k = 0; k < 3; ++k) {
        a += j[0][k] * j[0][k];
        b += j[0][k] * j[1][k];
        c += j[1][k] * j[1][k];
    }
    double det = a * c - b * b;
    if (det < 1e-9)
        return {};
    double u = (c * residual[0] - b * residual[1]) / det, v = (a * residual[1] - b * residual[0]) / det;
    double weight = std::min(1.0, 12 / std::max(error, 1e-9));
    return bounded(
        V3{j[0][0] * u + j[1][0] * v, j[0][1] * u + j[1][1] * v, j[0][2] * u + j[1][2] * v} * weight, .025);
}
ColorProjection fitColorProjection(const Frame &f) {
    if(f.colorProjection)return *f.colorProjection;
    ColorProjection result;
    result.width=f.width;result.height=f.height;
    double matrix[11][12]{};
    unsigned count = 0;
    size_t stride = std::max(size_t(1), f.mapping.size() / 1200);
    for (size_t i = 0; i < f.mapping.size(); i += stride) {
        auto m = f.mapping[i];
        if (m.z < (f.sensorVersion==2?.5:.8) || m.z > (f.sensorVersion==2?4.5:4.) || m.u < 0 || m.u >= f.width || m.v < 0 || m.v >= f.height || !finite({m.x, m.y, m.z}))
            continue;
        double u = double(m.u) / f.width, v = double(m.v) / f.height;
        double rows[2][11]{{m.x, m.y, m.z, 1, 0, 0, 0, 0, -u * m.x, -u * m.y, -u},
                           {0, 0, 0, 0, m.x, m.y, m.z, 1, -v * m.x, -v * m.y, -v}};
        double targets[2]{u * m.z, v * m.z};
        for (int r = 0; r < 2; ++r)
            for (int a = 0; a < 11; ++a) {
                for (int b = 0; b < 11; ++b)
                    matrix[a][b] += rows[r][a] * rows[r][b];
                matrix[a][11] += rows[r][a] * targets[r];
            }
        ++count;
    }
    if (count < 100)
        return result;
    for (int col = 0; col < 11; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 11; ++row)
            if (std::abs(matrix[row][col]) > std::abs(matrix[pivot][col]))
                pivot = row;
        if (std::abs(matrix[pivot][col]) < 1e-6)
            return result;
        for (int j = col; j < 12; ++j)
            std::swap(matrix[pivot][j], matrix[col][j]);
        double scale = matrix[col][col];
        for (int j = col; j < 12; ++j)
            matrix[col][j] /= scale;
        for (int row = 0; row < 11; ++row)
            if (row != col) {
                double a = matrix[row][col];
                for (int j = col; j < 12; ++j)
                    matrix[row][j] -= a * matrix[col][j];
            }
    }
    for (int i = 0; i < 11; ++i)
        result.p[i] = matrix[i][11];
    double error = 0;
    count = 0;
    for (size_t i = stride / 2; i < f.mapping.size(); i += stride) {
        auto m = f.mapping[i];
        if (m.z < (f.sensorVersion==2?.5:.8) || m.z > (f.sensorVersion==2?4.5:4.) || m.u < 0 || m.u >= f.width || m.v < 0 || m.v >= f.height)
            continue;
        auto uv = result.project({m.x, m.y, m.z});
        double e = std::hypot(uv.x - m.u, uv.y - m.v);
        if (!std::isfinite(e))
            return result;
        error += e * e;
        ++count;
    }
    result.rms = count ? std::sqrt(error / count) : 100;
    result.valid = count >= 100 && result.rms < 1.5 * std::max(1., f.width / 640.);
    return result;
}
} // namespace kf
